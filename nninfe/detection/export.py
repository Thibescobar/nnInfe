"""Detection export helpers."""

import json
from typing import Dict, Tuple

import numpy as np
import SimpleITK as sitk


def detections_to_mask(
    detection: Dict[str, np.ndarray],
    image_shape_zyx: Tuple[int, ...],
    reference_image: sitk.Image,
) -> sitk.Image:
    """Build a connected-component label map from detected boxes."""
    mask = np.zeros(image_shape_zyx, dtype=np.uint8)
    for box in detection["boxes"]:
        d0a, d1a, d0b, d1b, d2a, d2b = box.astype(int)
        d0a, d0b = max(0, d0a), min(image_shape_zyx[0], d0b)
        d1a, d1b = max(0, d1a), min(image_shape_zyx[1], d1b)
        d2a, d2b = max(0, d2a), min(image_shape_zyx[2], d2b)
        mask[d0a:d0b, d1a:d1b, d2a:d2b] = 1

    mask_sitk = sitk.GetImageFromArray(mask)
    mask_sitk.CopyInformation(reference_image)
    return sitk.ConnectedComponent(mask_sitk)


def _rescale_boxes_to_ref(
    boxes: np.ndarray,
    ref_spacing_xyz: Tuple[float, float, float],
    current_meta: dict = None,
) -> np.ndarray:
    if current_meta is None or len(boxes) == 0:
        return boxes
    cur_spacing_xyz = current_meta["spacing_xyz"]
    if cur_spacing_xyz == ref_spacing_xyz:
        return boxes
    scale_d0 = cur_spacing_xyz[2] / ref_spacing_xyz[2]
    scale_d1 = cur_spacing_xyz[1] / ref_spacing_xyz[1]
    scale_d2 = cur_spacing_xyz[0] / ref_spacing_xyz[0]
    scale = np.array([scale_d0, scale_d1, scale_d0, scale_d1, scale_d2, scale_d2], dtype=np.float32)
    return boxes * scale


def export_detections_json(
    detection: Dict[str, np.ndarray],
    output_path: str,
    ref_meta: dict,
    current_meta: dict = None,
) -> None:
    ref_spacing_xyz = ref_meta["spacing_xyz"]
    boxes = _rescale_boxes_to_ref(detection["boxes"], ref_spacing_xyz, current_meta)

    data = {
        "pred_boxes": boxes.tolist(),
        "pred_scores": detection["scores"].tolist(),
        "pred_labels": detection["labels"].tolist(),
        "restore": True,
        "original_size_of_raw_data": list(reversed(ref_meta["size_xyz"])),
        "itk_origin": list(ref_meta["origin"]),
        "itk_spacing": list(ref_spacing_xyz),
        "itk_direction": list(ref_meta["direction"]),
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


def export_detections_pkl(
    detection: Dict[str, np.ndarray],
    output_path: str,
    ref_meta: dict,
    current_meta: dict = None,
) -> None:
    ref_spacing_xyz = ref_meta["spacing_xyz"]
    boxes = _rescale_boxes_to_ref(detection["boxes"], ref_spacing_xyz, current_meta)

    data = {
        "pred_boxes": np.asarray(boxes, dtype=np.float32),
        "pred_scores": np.asarray(detection["scores"], dtype=np.float32),
        "pred_labels": np.asarray(detection["labels"], dtype=np.int64),
        "restore": True,
        "original_size_of_raw_data": np.array(list(reversed(ref_meta["size_xyz"])), dtype=np.int64),
        "itk_origin": [float(x) for x in ref_meta["origin"]],
        "itk_spacing": [float(x) for x in ref_spacing_xyz],
        "itk_direction": [float(x) for x in ref_meta["direction"]],
    }
    with open(output_path, "wb") as f:
        import pickle

        pickle.dump(data, f)


def export_detections_csv(
    detection: Dict[str, np.ndarray],
    output_path: str,
    image_name: str,
    ref_meta: dict,
    current_meta: dict = None,
) -> None:
    ref_spacing_xyz = ref_meta["spacing_xyz"]
    origin = ref_meta["origin"]
    boxes = _rescale_boxes_to_ref(detection["boxes"], ref_spacing_xyz, current_meta)

    header = (
        "image_name,detection_id,label,score,"
        "z_min,y_min,x_min,z_max,y_max,x_max,"
        "center_x_mm,center_y_mm,center_z_mm,"
        "size_x_mm,size_y_mm,size_z_mm,volume_mm3\n"
    )
    with open(output_path, "w") as f:
        f.write(header)
        for i, (box, score, label) in enumerate(zip(boxes, detection["scores"], detection["labels"])):
            d0a, d1a, d0b, d1b, d2a, d2b = box
            cz = (d0a + d0b) / 2.0
            cy = (d1a + d1b) / 2.0
            cx = (d2a + d2b) / 2.0
            wx = origin[0] + cx * ref_spacing_xyz[0]
            wy = origin[1] + cy * ref_spacing_xyz[1]
            wz = origin[2] + cz * ref_spacing_xyz[2]
            sz_x = (d2b - d2a) * ref_spacing_xyz[0]
            sz_y = (d1b - d1a) * ref_spacing_xyz[1]
            sz_z = (d0b - d0a) * ref_spacing_xyz[2]
            vol = sz_x * sz_y * sz_z
            f.write(
                f"{image_name},{i + 1},{int(label)},{score:.4f},"
                f"{d0a:.1f},{d1a:.1f},{d2a:.1f},{d0b:.1f},{d1b:.1f},{d2b:.1f},"
                f"{wx:.1f},{wy:.1f},{wz:.1f},"
                f"{sz_x:.1f},{sz_y:.1f},{sz_z:.1f},{vol:.1f}\n"
            )
