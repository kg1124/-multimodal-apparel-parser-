import json
import os
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from shapely.geometry import Polygon

from apparel_parser.common.constants import CATEGORY_ID_TO_TARGET_INDEX


def polygons_to_yolo_lines(polygons: list, target_class: int, img_w: int, img_h: int) -> list:
    """
    把同一个实例的所有多边形组件合并成尽量少的YOLO-seg行。
    做法：把所有子多边形画到同一张掩码上，再用findContours重新提取轮廓——
    这样如果子多边形本来是同一个物体被分成的两块（比如两条裤腿），
    只要它们在图上有重叠或接壤，就会被合并成一个连续轮廓，不会被错误地
    拆成两个独立实例。
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    valid_any = False
    for polygon in polygons:
        if len(polygon) < 6:
            continue
        pts = np.array(polygon, dtype=np.float64).reshape(-1, 2)
        pts[:, 0] = np.clip(pts[:, 0], 0, img_w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, img_h - 1)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
        valid_any = True

    if not valid_any:
        return []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        if len(cnt) < 3 or cv2.contourArea(cnt) < 4:
            continue
        normalized = []
        pts = []
        for point in cnt.reshape(-1, 2):
            x = min(max(point[0] / img_w, 0.0), 1.0)
            y = min(max(point[1] / img_h, 0.0), 1.0)
            normalized.append(f"{x:.6f}")
            normalized.append(f"{y:.6f}")
            pts.append((x, y))
        if not Polygon(pts).is_valid:
            continue
        lines.append(f"{target_class} " + " ".join(normalized))
    return lines


def annotation_to_yolo_lines(data: dict, img_w: int, img_h: int) -> list:
    """
    核心转换逻辑（纯函数，不涉及文件读写，方便单元测试）。
    输入：一张图的原始标注数据（dict）+ 图片宽高
    输出：YOLO-seg格式的标注行列表，每行格式为 "class_id x1 y1 x2 y2 ..."（坐标已归一化到0-1）
    """
    lines = []
    for key, item in data.items():
        if not key.startswith("item"):
            continue

        category_id = item.get("category_id")
        if category_id not in CATEGORY_ID_TO_TARGET_INDEX:
            continue

        target_class = CATEGORY_ID_TO_TARGET_INDEX[category_id]
        polygons = item.get("segmentation", [])
        lines.extend(polygons_to_yolo_lines(polygons, target_class, img_w, img_h))

    return lines


def convert_one_file(json_path: Path, image_path: Path) -> list:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with Image.open(image_path) as img:
        img_w, img_h = img.size
    return annotation_to_yolo_lines(data, img_w, img_h)


def convert_split(deepfashion2_root: str, split: str, output_root: str) -> None:
    image_dir = Path(deepfashion2_root) / split / "image"
    annos_dir = Path(deepfashion2_root) / split / "annos"

    out_split = "train" if split == "train" else "val"
    out_image_dir = Path(output_root) / "images" / out_split
    out_label_dir = Path(output_root) / "labels" / out_split
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(annos_dir.glob("*.json"))
    print(f"[{split}] 共 {len(json_files)} 个标注文件")

    kept = 0
    for json_path in json_files:
        image_path = image_dir / (json_path.stem + ".jpg")
        if not image_path.exists():
            continue

        lines = convert_one_file(json_path, image_path)
        if not lines:
            continue

        out_image_path = out_image_dir / image_path.name
        if not out_image_path.exists():
            os.symlink(image_path.resolve(), out_image_path)

        out_label_path = out_label_dir / (json_path.stem + ".txt")
        with open(out_label_path, "w") as f:
            f.write("\n".join(lines))

        kept += 1

    print(f"[{split}] 转换完成，保留 {kept} 张有效图片")
