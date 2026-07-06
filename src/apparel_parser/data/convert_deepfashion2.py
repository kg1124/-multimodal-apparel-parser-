import json
import os
from pathlib import Path
from PIL import Image

from apparel_parser.common.constants import CATEGORY_ID_TO_TARGET_INDEX


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

        for polygon in item.get("segmentation", []):
            if len(polygon) < 6:
                continue

            normalized = []
            for i in range(0, len(polygon), 2):
                x = min(max(polygon[i] / img_w, 0.0), 1.0)
                y = min(max(polygon[i + 1] / img_h, 0.0), 1.0)
                normalized.append(f"{x:.6f}")
                normalized.append(f"{y:.6f}")

            lines.append(f"{target_class} " + " ".join(normalized))

    return lines


def convert_one_file(json_path: Path, image_path: Path) -> list:
    """读取单个json+图片文件，调用核心转换逻辑"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with Image.open(image_path) as img:
        img_w, img_h = img.size
    return annotation_to_yolo_lines(data, img_w, img_h)


def convert_split(deepfashion2_root: str, split: str, output_root: str) -> None:
    """转换一个数据集划分（train 或 validation）下的所有标注"""
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
