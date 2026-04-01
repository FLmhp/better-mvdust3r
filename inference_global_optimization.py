import argparse
import copy
import os
import sys
import tempfile
import time

import matplotlib.pyplot as pl
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.runtime_utils import (
    adapt_conf_drop_percentile,
    confidence_keep_mask,
    mean_quality_score,
    min_keep_points_from_hw,
    raw_scene_confidence_threshold,
    resolve_scene_graph_policy,
)
from dust3r.utils.device import to_numpy
from dust3r.utils.image import load_images, rgb
from dust3r.viz import CAM_COLORS, OPENGL, add_scene_cam, cat_meshes, pts3d_to_trimesh

pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True


def default_runtime_options():
    return {
        "runtime_profile": "lite_robust_v1",
        "amp": True,
        "pair_batch_size": 4,
        "oom_retry": True,
        "scene_graph_policy": "auto",
        "go_schedule": "linear",
        "go_lr": 0.01,
        "go_min_iter": 40,
        "go_max_iter": 300,
        "go_check_every": 5,
        "go_rel_tol": 5e-4,
        "go_patience": 3,
        "conf_percentile": 3.0,
        "preprocess_profile": "lite_robust_v1",
        "quality_adaptive": True,
    }


def resolve_runtime_options(args=None, runtime_options=None):
    options = default_runtime_options()
    if args is not None:
        if args.runtime_profile == "default":
            options["runtime_profile"] = "default"
            options["amp"] = False
            options["pair_batch_size"] = 1
            options["oom_retry"] = False
            options["scene_graph_policy"] = "complete"
            options["preprocess_profile"] = "none"
            options["quality_adaptive"] = False
            options["go_min_iter"] = args.go_max_iter
            options["go_check_every"] = 1
            options["go_rel_tol"] = 0.0
            options["go_patience"] = 0
        else:
            options["runtime_profile"] = args.runtime_profile

        if args.amp is not None:
            options["amp"] = args.amp
        if args.pair_batch_size is not None:
            options["pair_batch_size"] = max(1, args.pair_batch_size)
        if args.oom_retry is not None:
            options["oom_retry"] = args.oom_retry
        if args.scene_graph_policy is not None:
            options["scene_graph_policy"] = args.scene_graph_policy
        if args.go_schedule is not None:
            options["go_schedule"] = args.go_schedule
        if args.go_lr is not None:
            options["go_lr"] = args.go_lr
        if args.go_min_iter is not None:
            options["go_min_iter"] = args.go_min_iter
        if args.go_max_iter is not None:
            options["go_max_iter"] = args.go_max_iter
        if args.go_check_every is not None:
            options["go_check_every"] = args.go_check_every
        if args.go_rel_tol is not None:
            options["go_rel_tol"] = args.go_rel_tol
        if args.go_patience is not None:
            options["go_patience"] = args.go_patience
        if args.conf_percentile is not None:
            options["conf_percentile"] = args.conf_percentile
        if args.preprocess_profile is not None:
            options["preprocess_profile"] = args.preprocess_profile
        if args.quality_adaptive is not None:
            options["quality_adaptive"] = args.quality_adaptive

    if runtime_options:
        options.update(runtime_options)
    return options


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action="store_true", default=False)
    parser_url.add_argument("--server_name", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=224, choices=[512, 224])
    parser.add_argument("--server_port", type=int, default=None)
    parser_weights = parser.add_mutually_exclusive_group(required=True)
    parser_weights.add_argument("--weights", type=str, default=None)
    parser_weights.add_argument(
        "--model_name",
        type=str,
        choices=[
            "DUSt3R_ViTLarge_BaseDecoder_512_dpt",
            "DUSt3R_ViTLarge_BaseDecoder_512_linear",
            "DUSt3R_ViTLarge_BaseDecoder_224_linear",
        ],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tmp_dir", type=str, default=None)
    parser.add_argument("--silent", action="store_true", default=False)
    parser.add_argument("--runtime_profile", type=str, default="lite_robust_v1", choices=["default", "lite_robust_v1"])
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pair_batch_size", type=int, default=None)
    parser.add_argument("--oom_retry", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--scene_graph_policy", type=str, default=None, choices=["auto", "complete", "swin", "oneref", "anchorlocal"])
    parser.add_argument("--go_schedule", type=str, default=None, choices=["linear", "cosine"])
    parser.add_argument("--go_lr", type=float, default=None)
    parser.add_argument("--go_min_iter", type=int, default=None)
    parser.add_argument("--go_max_iter", type=int, default=300)
    parser.add_argument("--go_check_every", type=int, default=None)
    parser.add_argument("--go_rel_tol", type=float, default=None)
    parser.add_argument("--go_patience", type=int, default=None)
    parser.add_argument("--conf_percentile", type=float, default=None)
    parser.add_argument("--preprocess_profile", type=str, default=None, choices=["none", "lite_robust_v1"])
    parser.add_argument("--quality_adaptive", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("inputfiles", nargs="*")
    return parser


def _convert_scene_output_to_glb(
    outdir,
    imgs,
    pts3d,
    mask,
    focals,
    cams2world,
    cam_size=0.05,
    cam_color=None,
    as_pointcloud=False,
    transparent_cams=False,
    silent=False,
):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pointcloud = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pointcloud)
    else:
        meshes = [pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]) for i in range(len(imgs))]
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(
            scene,
            pose_c2w,
            camera_edge_color,
            None if transparent_cams else imgs[i],
            focals[i],
            imsize=imgs[i].shape[1::-1],
            screen_width=cam_size,
        )

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler("y", np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = os.path.join(outdir, "scene.glb")
    if not silent:
        print("(exporting 3D scene to", outfile, ")")
    scene.export(file_obj=outfile)
    return outfile


def _effective_conf_percentile(quality_score, runtime_options):
    return adapt_conf_drop_percentile(
        runtime_options["conf_percentile"],
        quality_score,
        runtime_options["quality_adaptive"],
    )


def _configure_scene_confidence(scene, quality_score, runtime_options):
    scene.min_conf_thr = raw_scene_confidence_threshold(
        scene,
        _effective_conf_percentile(quality_score, runtime_options),
    )


def _run_scene_reconstruction(imgs, model, device, silent, runtime_options):
    scene_graph = resolve_scene_graph_policy(imgs, runtime_options["scene_graph_policy"])
    if scene_graph == "swin":
        scene_graph = "swin-1"
    elif scene_graph == "oneref":
        scene_graph = "oneref-0"

    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)

    t = [time.time()]
    torch.cuda.synchronize()
    output = inference(
        pairs,
        model,
        device,
        batch_size=runtime_options["pair_batch_size"],
        verbose=not silent,
        use_amp=runtime_options["amp"],
        oom_retry=runtime_options["oom_retry"],
    )
    torch.cuda.synchronize()
    t.append(time.time())

    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    quality_score = mean_quality_score(imgs) or 0.0
    _configure_scene_confidence(scene, quality_score, runtime_options)

    torch.cuda.synchronize()
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        scene.compute_global_alignment(
            init="mst",
            niter=runtime_options["go_max_iter"],
            schedule=runtime_options["go_schedule"],
            lr=runtime_options["go_lr"],
            min_iter=runtime_options["go_min_iter"],
            check_every=runtime_options["go_check_every"],
            rel_tol=runtime_options["go_rel_tol"],
            patience=runtime_options["go_patience"],
        )
    torch.cuda.synchronize()
    t.append(time.time())

    if not silent:
        print("scene_graph", scene_graph)
        print("test net inference time", t[1] - t[0], "GO time", t[2] - t[1])
        if hasattr(scene, "last_alignment_iterations"):
            print("GO iterations", scene.last_alignment_iterations)

    return scene, t[1] - t[0], t[2] - t[1], quality_score


def get_3D_model_from_scene(
    outdir,
    silent,
    scene,
    min_conf_thr=3,
    as_pointcloud=False,
    mask_sky=False,
    clean_depth=False,
    transparent_cams=False,
    cam_size=0.05,
):
    if scene is None:
        return None
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    scene.min_conf_thr = raw_scene_confidence_threshold(scene, min_conf_thr)
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    pts3d = to_numpy(scene.get_pts3d())
    masks = to_numpy(scene.get_masks())
    return _convert_scene_output_to_glb(
        outdir,
        rgbimg,
        pts3d,
        masks,
        focals,
        cams2world,
        as_pointcloud=as_pointcloud,
        transparent_cams=transparent_cams,
        cam_size=cam_size,
        silent=silent,
    )


def get_reconstructed_scene(outdir, model, device, silent, image_size, filelist, runtime_options=None):
    runtime_options = resolve_runtime_options(runtime_options=runtime_options)
    imgs = load_images(
        filelist,
        size=image_size,
        verbose=not silent,
        preprocess_profile=runtime_options["preprocess_profile"],
    )
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]["idx"] = 1

    scene, _, _, quality_score = _run_scene_reconstruction(imgs, model, device, silent, runtime_options)
    outfile = get_3D_model_from_scene(outdir, silent, scene, runtime_options["conf_percentile"], as_pointcloud=True)

    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    cmap = pl.get_cmap("jet")
    depths_max = max([d.max() for d in depths])
    depths = [d / depths_max for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d / confs_max) for d in confs]

    gallery = []
    for i in range(len(rgbimg)):
        gallery.append(rgbimg[i])
        gallery.append(rgb(depths[i]))
        gallery.append(rgb(confs[i]))

    return scene, outfile, gallery, quality_score


def _reorder_views_for_demo(imgs):
    imgs = [copy.deepcopy(img) for img in imgs]
    if len(imgs) < 12:
        if len(imgs) > 3:
            imgs[1], imgs[3] = imgs[3], imgs[1]
        if len(imgs) > 6:
            imgs[2], imgs[6] = imgs[6], imgs[2]
    else:
        change_id = len(imgs) // 4 + 1
        imgs[1], imgs[change_id] = imgs[change_id], imgs[1]
        change_id = (len(imgs) * 2) // 4 + 1
        imgs[2], imgs[change_id] = imgs[change_id], imgs[2]
        change_id = (len(imgs) * 3) // 4 + 1
        imgs[3], imgs[change_id] = imgs[change_id], imgs[3]
    return imgs


def Rt(transform, points):
    return transform[:3, 3] + points @ transform[:3, :3].T


def inference_global_optimization(model, device, silent, img_tensors, first_view_c2w, runtime_options=None):
    del first_view_c2w
    runtime_options = resolve_runtime_options(runtime_options=runtime_options)
    imgs = []
    for img_id, img in enumerate(img_tensors):
        imgs.append(
            dict(
                img=img[None],
                true_shape=np.int32([img.shape[-2:]]),
                idx=img_id,
                instance=str(img_id),
                quality_score=np.float32(1.0),
                raw_quality_score=np.float32(1.0),
            )
        )
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]["idx"] = 1

    scene, net_time, go_time, _ = _run_scene_reconstruction(imgs, model, device, silent, runtime_options)

    pts3d = scene.get_pts3d()
    conf = scene.get_conf()
    all_c2w = scene.get_im_poses()
    intrinsics = scene.get_intrinsics()

    output_pcd = []
    original_first_w2c = torch.linalg.inv(all_c2w[0])
    for pcd in pts3d:
        original_shape = pcd.shape
        pcd_canonical = Rt(original_first_w2c, pcd.reshape(-1, 3))
        output_pcd.append(pcd_canonical.reshape(*original_shape))

    return output_pcd, all_c2w, intrinsics, conf, net_time, go_time


def loss_of_one_batch_go_mv(
    batch,
    model,
    criterion,
    device,
    symmetrize_batch=False,
    use_amp=False,
    ret=None,
    runtime_options=None,
):
    del symmetrize_batch, use_amp
    runtime_options = resolve_runtime_options(runtime_options=runtime_options)
    views = batch
    view1, view2s = views[0], views[1:]
    for view in batch:
        for name in "img pts3d valid_mask camera_pose camera_intrinsics F_matrix corres".split():
            if name in view:
                view[name] = view[name].to(device, non_blocking=True)

    t1s = []
    t2s = []
    n_v_real = 1
    for view2 in view2s:
        if view2["only_render"][0].item():
            break
        n_v_real += 1

    view2s_all = view2s
    view2s = view2s[: n_v_real - 1]
    views = [view1] + view2s
    n_v = len(view2s) + 1
    bs = view1["img"].shape[0]
    preds = [{"pts3d": [], "conf": [], "c2ws_pred": [], "intrinsics_pred": []} for _ in range(n_v)]

    for i in range(bs):
        pts3ds, c2ws, intrinsics, confs, t1, t2 = inference_global_optimization(
            model,
            device,
            False,
            [view1["img"][i]] + [view2["img"][i] for view2 in view2s],
            view1["camera_pose"][i],
            runtime_options=runtime_options,
        )
        t1s.append(t1)
        t2s.append(t2)
        for j in range(n_v):
            preds[j]["pts3d"].append(pts3ds[j])
            preds[j]["conf"].append(confs[j])
            preds[j]["c2ws_pred"].append(c2ws[j])
            preds[j]["intrinsics_pred"].append(intrinsics[j])

    for pred, view in zip(preds, views):
        pred["pts3d"] = torch.stack(pred["pts3d"], dim=0).detach()
        pred["conf"] = torch.stack(pred["conf"], dim=0).detach()
        pred["c2ws_pred"] = torch.stack(pred["c2ws_pred"], dim=0).detach()
        pred["intrinsics_pred"] = torch.stack(pred["intrinsics_pred"], dim=0).detach()
        pred["rgb"] = view["img"].permute(0, 2, 3, 1)
        pred["opacity"] = torch.ones_like(pred["rgb"][:, :, :, 0:1])
        min_keep = min_keep_points_from_hw(pred["conf"].shape[-2], pred["conf"].shape[-1])
        for batch_idx in range(bs):
            conf_mask = ~confidence_keep_mask(
                pred["conf"][batch_idx],
                runtime_options["conf_percentile"],
                min_keep=min_keep,
            )
            pred["opacity"][batch_idx][conf_mask] = 0
        pred["scale"] = torch.ones_like(pred["rgb"]) * 1e-3 * 2
        pred["rotation"] = torch.ones_like(pred["rgb"][:, :, :, 0:1].repeat(1, 1, 1, 4))

    for pred in preds[1:]:
        pred["pts3d_in_other_view"] = pred.pop("pts3d")
    pred1, pred2s = preds[0], preds[1:]

    loss = None
    if criterion is not None:
        with torch.cuda.amp.autocast(enabled=False):
            loss = criterion(view1, view2s_all, pred1, pred2s, log=True)

    view2s = batch[1:]
    result = dict(view1=view1, view2s=view2s, pred1=pred1, pred2s=pred2s, loss=loss)
    result = result[ret] if ret else result
    return result, float(np.mean(t1s)), float(np.mean(t2s)), n_v_real


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    if args.tmp_dir is not None:
        os.makedirs(args.tmp_dir, exist_ok=True)
        tempfile.tempdir = args.tmp_dir

    if not args.inputfiles:
        parser.print_help()
        return

    weights_path = args.weights if args.weights is not None else "naver/" + args.model_name
    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(args.device)
    model.eval()
    runtime_options = resolve_runtime_options(args=args)

    with tempfile.TemporaryDirectory(suffix="dust3r_go") as tmpdirname:
        if not args.silent:
            print("Outputing stuff in", tmpdirname)
        _, outfile, _, quality_score = get_reconstructed_scene(
            tmpdirname,
            model,
            args.device,
            args.silent,
            args.image_size,
            args.inputfiles,
            runtime_options=runtime_options,
        )
        if not args.silent:
            print("saved glb", outfile)
            print("mean quality score", quality_score)


if __name__ == "__main__":
    main()
