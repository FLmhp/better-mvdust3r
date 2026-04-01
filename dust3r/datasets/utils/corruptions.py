import cv2
import numpy as np


_RGB_OPS = (
    "gaussian_blur",
    "motion_blur",
    "gaussian_noise",
    "jpeg",
    "downup",
    "exposure",
    "contrast",
    "occlusion",
)

_GEOM_OPS = (
    "depth_dropout",
    "depth_quantization",
    "focal_perturbation",
    "pose_perturbation",
)

_RGB_SEVERITY = {
    "mild": {
        "num_ops": (1, 1),
        "gaussian_sigma": (0.35, 0.9),
        "motion_kernel": (3, 5),
        "noise_std": (4.0, 10.0),
        "jpeg_quality": (40, 62),
        "downsample_scale": (0.55, 0.75),
        "exposure_scale": (0.85, 1.15),
        "contrast_scale": (0.82, 1.18),
        "occlusion_area": (0.04, 0.10),
    },
    "moderate": {
        "num_ops": (1, 2),
        "gaussian_sigma": (0.8, 1.8),
        "motion_kernel": (5, 9),
        "noise_std": (10.0, 22.0),
        "jpeg_quality": (18, 40),
        "downsample_scale": (0.35, 0.6),
        "exposure_scale": (0.7, 1.3),
        "contrast_scale": (0.65, 1.35),
        "occlusion_area": (0.08, 0.18),
    },
}

_GEOM_SEVERITY = {
    "mild": {
        "num_ops": (1, 1),
        "depth_dropout": (0.02, 0.06),
        "depth_quant_levels": (96, 192),
        "focal_scale": (0.97, 1.03),
        "translation_std": (0.005, 0.015),
        "rotation_deg": (0.5, 1.5),
    },
    "moderate": {
        "num_ops": (1, 2),
        "depth_dropout": (0.05, 0.12),
        "depth_quant_levels": (48, 112),
        "focal_scale": (0.94, 1.06),
        "translation_std": (0.01, 0.03),
        "rotation_deg": (1.0, 3.0),
    },
}


def _normalize_rgb_profile(profile):
    if profile in (None, "none"):
        return "none"
    if profile != "robust_v1":
        raise ValueError(f"unsupported RGB corruption profile: {profile}")
    return profile


def _normalize_geom_profile(profile):
    if profile in (None, "none"):
        return "none"
    if profile in ("pose_depth_v1", "robust_v1"):
        return "pose_depth_v1"
    raise ValueError(f"unsupported geometry noise profile: {profile}")


def _normalize_severity(severity):
    if severity not in _RGB_SEVERITY:
        raise ValueError(f"unsupported corruption severity: {severity}")
    return severity


def _clip_probability(probability):
    return float(np.clip(float(probability), 0.0, 1.0))


def _randint_inclusive(rng, low, high):
    return int(rng.integers(int(low), int(high) + 1))


def _sample_num_ops(rng, config):
    lo, hi = config["num_ops"]
    return _randint_inclusive(rng, lo, hi)


def _sample_signed_uniform(rng, magnitude_range):
    magnitude = float(rng.uniform(float(magnitude_range[0]), float(magnitude_range[1])))
    return magnitude if rng.random() >= 0.5 else -magnitude


def _sample_unit_vector(rng):
    axis = rng.normal(size=3)
    axis_norm = np.linalg.norm(axis) + 1e-8
    return (axis / axis_norm).astype(np.float32)


def _axis_angle_to_matrix(axis, angle):
    x, y, z = axis
    skew = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float32,
    )
    eye = np.eye(3, dtype=np.float32)
    return eye + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _ensure_rgb_uint8(image):
    image_np = np.asarray(image)
    if image_np.ndim == 2:
        image_np = np.repeat(image_np[..., None], 3, axis=2)
    if image_np.shape[2] == 4:
        image_np = image_np[..., :3]
    if image_np.dtype != np.uint8:
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    return image_np


def _apply_gaussian_blur(image, rng, config):
    sigma = float(rng.uniform(*config["gaussian_sigma"]))
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return blurred, f"gaussian_blur@{sigma:.2f}"


def _apply_motion_blur(image, rng, config):
    kernel_size = _randint_inclusive(rng, *config["motion_kernel"])
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    direction = int(rng.integers(0, 4))
    if direction == 0:
        kernel[kernel_size // 2, :] = 1.0
    elif direction == 1:
        kernel[:, kernel_size // 2] = 1.0
    elif direction == 2:
        np.fill_diagonal(kernel, 1.0)
    else:
        np.fill_diagonal(np.fliplr(kernel), 1.0)
    kernel /= np.maximum(kernel.sum(), 1e-8)
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred, f"motion_blur@{kernel_size}"


def _apply_gaussian_noise(image, rng, config):
    std = float(rng.uniform(*config["noise_std"]))
    noisy = image.astype(np.float32) + rng.normal(0.0, std, size=image.shape).astype(np.float32)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy, f"gaussian_noise@{std:.1f}"


def _apply_jpeg(image, rng, config):
    quality = _randint_inclusive(rng, *config["jpeg_quality"])
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image, "jpeg@skip"
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return decoded, f"jpeg@{quality}"


def _apply_downup(image, rng, config):
    scale = float(rng.uniform(*config["downsample_scale"]))
    height, width = image.shape[:2]
    small_width = max(16, int(round(width * scale)))
    small_height = max(16, int(round(height * scale)))
    down = cv2.resize(image, (small_width, small_height), interpolation=cv2.INTER_AREA)
    up = cv2.resize(down, (width, height), interpolation=cv2.INTER_LINEAR)
    return up, f"downup@{scale:.2f}"


def _apply_exposure(image, rng, config):
    scale = float(rng.uniform(*config["exposure_scale"]))
    shifted = np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return shifted, f"exposure@{scale:.2f}"


def _apply_contrast(image, rng, config):
    scale = float(rng.uniform(*config["contrast_scale"]))
    mean = image.astype(np.float32).mean(axis=(0, 1), keepdims=True)
    contrasted = np.clip((image.astype(np.float32) - mean) * scale + mean, 0, 255).astype(np.uint8)
    return contrasted, f"contrast@{scale:.2f}"


def _apply_occlusion(image, rng, config):
    height, width = image.shape[:2]
    area_ratio = float(rng.uniform(*config["occlusion_area"]))
    box_area = max(1, int(round(height * width * area_ratio)))
    aspect_ratio = float(rng.uniform(0.5, 2.0))
    box_h = int(np.clip(np.sqrt(box_area / aspect_ratio), 1, height))
    box_w = int(np.clip(box_area / max(box_h, 1), 1, width))
    top = int(rng.integers(0, max(1, height - box_h + 1)))
    left = int(rng.integers(0, max(1, width - box_w + 1)))
    fill = image.mean(axis=(0, 1), keepdims=True).astype(np.uint8)
    occluded = image.copy()
    occluded[top:top + box_h, left:left + box_w] = fill
    return occluded, f"occlusion@{area_ratio:.2f}"


_RGB_APPLIERS = {
    "gaussian_blur": _apply_gaussian_blur,
    "motion_blur": _apply_motion_blur,
    "gaussian_noise": _apply_gaussian_noise,
    "jpeg": _apply_jpeg,
    "downup": _apply_downup,
    "exposure": _apply_exposure,
    "contrast": _apply_contrast,
    "occlusion": _apply_occlusion,
}


def build_corruption_state(
    rng,
    *,
    corruption_profile="none",
    corruption_severity="mild",
    corruption_prob=0.0,
    geom_noise_profile="none",
    geom_noise_prob=0.0,
    geom_loss_weight=0.3,
):
    rgb_profile = _normalize_rgb_profile(corruption_profile)
    geom_profile = _normalize_geom_profile(geom_noise_profile)
    severity = _normalize_severity(corruption_severity)

    rgb_prob = _clip_probability(corruption_prob) if rgb_profile != "none" else 0.0
    geom_prob = _clip_probability(geom_noise_prob) if geom_profile != "none" else 0.0

    total_prob = rgb_prob + geom_prob
    if total_prob > 1.0:
        rgb_prob /= total_prob
        geom_prob /= total_prob
        total_prob = 1.0

    draw = float(rng.random())
    mode = "clean"
    if draw < geom_prob:
        mode = "geometry"
    elif draw < geom_prob + rgb_prob:
        mode = "rgb"

    return {
        "mode": mode,
        "severity": severity,
        "corruption_profile": rgb_profile,
        "geom_noise_profile": geom_profile,
        "rgb_corrupted": mode == "rgb",
        "geom_corrupted": mode == "geometry",
        "geometry_loss_weight": np.float32(geom_loss_weight if mode == "geometry" else 1.0),
        "rgb_corruption_name": "none",
        "geom_noise_name": "none",
        "rgb_prob": np.float32(rgb_prob),
        "geom_prob": np.float32(geom_prob),
    }


def apply_rgb_corruption(image, rng, *, profile="robust_v1", severity="mild"):
    profile = _normalize_rgb_profile(profile)
    severity = _normalize_severity(severity)
    if profile == "none":
        return image, "none"

    image_np = _ensure_rgb_uint8(image)
    config = _RGB_SEVERITY[severity]
    num_ops = min(len(_RGB_OPS), _sample_num_ops(rng, config))
    op_names = list(rng.choice(np.array(_RGB_OPS), size=num_ops, replace=False))

    output = image_np
    applied = []
    for op_name in op_names:
        output, op_label = _RGB_APPLIERS[str(op_name)](output, rng, config)
        applied.append(op_label)

    return output, "+".join(applied)


def _apply_depth_dropout(depthmap, rng, config):
    depth = np.asarray(depthmap, dtype=np.float32).copy()
    valid_mask = depth > 0.0
    drop_ratio = float(rng.uniform(*config["depth_dropout"]))
    if valid_mask.any():
        drop_mask = rng.random(depth.shape) < drop_ratio
        depth[valid_mask & drop_mask] = 0.0
    return depth, f"depth_dropout@{drop_ratio:.3f}"


def _apply_depth_quantization(depthmap, rng, config):
    depth = np.asarray(depthmap, dtype=np.float32).copy()
    valid_mask = depth > 0.0
    if not valid_mask.any():
        return depth, "depth_quantization@skip"
    levels = _randint_inclusive(rng, *config["depth_quant_levels"])
    valid_depth = depth[valid_mask]
    min_depth = float(valid_depth.min())
    max_depth = float(valid_depth.max())
    if max_depth <= min_depth + 1e-6:
        return depth, f"depth_quantization@{levels}"
    normalized = (valid_depth - min_depth) / (max_depth - min_depth)
    quantized = np.round(normalized * (levels - 1)) / max(levels - 1, 1)
    depth[valid_mask] = quantized * (max_depth - min_depth) + min_depth
    return depth.astype(np.float32), f"depth_quantization@{levels}"


def _apply_focal_perturbation(intrinsics, rng, config):
    intrinsics_out = np.asarray(intrinsics, dtype=np.float32).copy()
    fx_scale = 1.0 + _sample_signed_uniform(rng, (0.0, max(abs(config["focal_scale"][0] - 1.0), abs(config["focal_scale"][1] - 1.0))))
    fy_scale = 1.0 + _sample_signed_uniform(rng, (0.0, max(abs(config["focal_scale"][0] - 1.0), abs(config["focal_scale"][1] - 1.0))))
    fx_scale = float(np.clip(fx_scale, *config["focal_scale"]))
    fy_scale = float(np.clip(fy_scale, *config["focal_scale"]))
    intrinsics_out[0, 0] *= fx_scale
    intrinsics_out[1, 1] *= fy_scale
    return intrinsics_out, f"focal_perturbation@{fx_scale:.3f},{fy_scale:.3f}"


def _apply_pose_perturbation(camera_pose, rng, config):
    pose_out = np.asarray(camera_pose, dtype=np.float32).copy()
    axis = _sample_unit_vector(rng)
    angle_deg = _sample_signed_uniform(rng, config["rotation_deg"])
    angle_rad = np.deg2rad(angle_deg).astype(np.float32)
    rotation_delta = _axis_angle_to_matrix(axis, angle_rad)
    translation_std = float(rng.uniform(*config["translation_std"]))
    translation_delta = rng.normal(0.0, translation_std, size=3).astype(np.float32)
    pose_out[:3, :3] = rotation_delta @ pose_out[:3, :3]
    pose_out[:3, 3] = pose_out[:3, 3] + translation_delta
    return pose_out, f"pose_perturbation@{angle_deg:.2f}deg,{translation_std:.4f}m"


_GEOM_APPLIERS = {
    "depth_dropout": _apply_depth_dropout,
    "depth_quantization": _apply_depth_quantization,
    "focal_perturbation": _apply_focal_perturbation,
    "pose_perturbation": _apply_pose_perturbation,
}


def apply_geometry_corruption(depthmap, intrinsics, camera_pose, rng, *, profile="pose_depth_v1", severity="mild"):
    profile = _normalize_geom_profile(profile)
    severity = _normalize_severity(severity)
    if profile == "none":
        return depthmap, intrinsics, camera_pose, "none"

    config = _GEOM_SEVERITY[severity]
    num_ops = min(len(_GEOM_OPS), _sample_num_ops(rng, config))
    op_names = list(rng.choice(np.array(_GEOM_OPS), size=num_ops, replace=False))

    depth_out = np.asarray(depthmap, dtype=np.float32).copy()
    intrinsics_out = np.asarray(intrinsics, dtype=np.float32).copy()
    pose_out = np.asarray(camera_pose, dtype=np.float32).copy()
    applied = []

    for op_name in op_names:
        op_name = str(op_name)
        if op_name in ("depth_dropout", "depth_quantization"):
            depth_out, op_label = _GEOM_APPLIERS[op_name](depth_out, rng, config)
        elif op_name == "focal_perturbation":
            intrinsics_out, op_label = _GEOM_APPLIERS[op_name](intrinsics_out, rng, config)
        else:
            pose_out, op_label = _GEOM_APPLIERS[op_name](pose_out, rng, config)
        applied.append(op_label)

    return depth_out.astype(np.float32), intrinsics_out.astype(np.float32), pose_out.astype(np.float32), "+".join(applied)
