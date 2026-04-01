import math

import cv2
import numpy as np
import PIL.Image
import torch


def is_oom_error(exc):
    message = str(exc).lower()
    return isinstance(exc, RuntimeError) and "out of memory" in message


def build_batch_size_candidates(initial_batch_size, oom_retry):
    batch_size = max(1, int(initial_batch_size))
    if not oom_retry:
        return [batch_size]

    candidates = []
    current = batch_size
    while True:
        if current not in candidates:
            candidates.append(current)
        if current == 1:
            break
        current = max(1, current // 2)
    return candidates


def confidence_threshold(confidence, drop_percentile, min_keep=1):
    flat_confidence = confidence.reshape(-1)
    if flat_confidence.numel() == 0:
        raise ValueError("confidence tensor is empty")

    drop_percentile = float(np.clip(drop_percentile, 0.0, 99.9))
    min_keep = max(1, int(min_keep))
    sorted_confidence = flat_confidence.sort()[0]
    max_drop = max(0, sorted_confidence.numel() - min_keep)
    drop_index = min(int(sorted_confidence.numel() * drop_percentile * 0.01), max_drop)
    return sorted_confidence[drop_index]


def confidence_keep_mask(confidence, drop_percentile, min_keep=1):
    threshold = confidence_threshold(confidence, drop_percentile, min_keep=min_keep)
    return confidence >= threshold


def pil_to_numpy_rgb(image):
    if isinstance(image, PIL.Image.Image):
        return np.asarray(image.convert("RGB"))
    return np.asarray(image)


def compute_image_quality_score(image):
    image_np = pil_to_numpy_rgb(image)
    if image_np.ndim != 3 or image_np.shape[2] != 3:
        raise ValueError("expected an RGB image")

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    contrast = float(gray.std())
    clipping_ratio = float(((gray <= 4) | (gray >= 251)).mean())

    sharpness_score = min(laplacian_variance / 300.0, 1.0)
    contrast_score = min(contrast / 64.0, 1.0)
    exposure_score = max(0.0, 1.0 - (clipping_ratio * 2.0))
    score = 0.5 * sharpness_score + 0.3 * contrast_score + 0.2 * exposure_score
    return float(np.clip(score, 0.0, 1.0))


def adapt_conf_drop_percentile(base_percentile, quality_score, quality_adaptive):
    percentile = float(base_percentile)
    if not quality_adaptive or quality_score is None:
        return percentile
    if quality_score < 0.35:
        return max(1.0, percentile * 0.5)
    if quality_score > 0.6:
        return min(15.0, percentile * 1.5)
    return percentile


def mean_quality_score(images):
    quality_scores = [float(img.get("quality_score", 0.0)) for img in images if isinstance(img, dict)]
    if not quality_scores:
        return None
    return float(sum(quality_scores) / len(quality_scores))


def preprocess_pil_image(image, profile="none", quality_score=None):
    if not isinstance(image, PIL.Image.Image):
        image = PIL.Image.fromarray(pil_to_numpy_rgb(image))

    if quality_score is None:
        quality_score = compute_image_quality_score(image)

    if profile in (None, "none"):
        return image, quality_score
    if profile != "lite_robust_v1":
        raise ValueError(f"unsupported preprocess profile: {profile}")

    image_np = pil_to_numpy_rgb(image)
    processed = image_np

    if quality_score < 0.55:
        processed = cv2.fastNlMeansDenoisingColored(processed, None, 3, 3, 5, 15)

    lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clip_limit = 1.5 if quality_score < 0.35 else 1.25
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    processed = cv2.cvtColor(cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2RGB)

    if quality_score < 0.25:
        blurred = cv2.GaussianBlur(processed, (0, 0), 1.0)
        processed = cv2.addWeighted(processed, 1.35, blurred, -0.35, 0)

    return PIL.Image.fromarray(processed), quality_score


def resolve_scene_graph_policy(imgs, scene_graph_policy):
    policy = scene_graph_policy or "auto"
    if policy == "auto":
        return "complete" if len(imgs) <= 8 else "anchorlocal-2"
    if policy == "anchorlocal":
        return "anchorlocal-2"
    return policy


def raw_scene_confidence_threshold(scene, drop_percentile, min_keep_per_view=256):
    confidences = [conf.reshape(-1) for conf in scene.im_conf]
    if not confidences:
        raise ValueError("scene has no confidences")
    flat_confidence = torch.cat(confidences, dim=0)
    min_keep = min(flat_confidence.numel(), max(1, int(min_keep_per_view)) * len(confidences))
    return float(confidence_threshold(flat_confidence, drop_percentile, min_keep=min_keep))


def min_keep_points_from_hw(height, width, keep_fraction=0.01, floor=256):
    total_points = int(height) * int(width)
    return min(total_points, max(floor, int(math.ceil(total_points * keep_fraction))))
