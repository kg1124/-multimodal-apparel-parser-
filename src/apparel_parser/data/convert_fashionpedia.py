import json
import os
from collections import defaultdict
from pathlib import Path

from shapely.geometry import Polygon

from apparel_parser.common.constants import FASHIONPEDIA_NAME_TO_TARGET_INDEX


def segmentation_to_yolo_lines(segmentation, target_class: int, img_w: int, img_h: int) -> list:
    """
    核心转换逻辑（纯函数，方便单元测试）。
    segmentation 是COCO polygon格式：list of [x1,y1,x2,y2,...]
    自相交/无效的多边形会被跳过，不写入结果。
    """
    lines = []
    if not isinstance(segmentation, list):
        return lines  # RLE等其他格式暂不支持，先跳过

    for polygon in segmentation:
        if len(polygon) < 6:
            continue
        normalized = []
        points = []
        for i in range(0, len(polygon), 2):
            x = min(max(polygon[i] / img_w, 0.0), 1.0)
            y = min(max(polygon[i + 1] / img_h, 0.0), 1.0)
            normalized.append(f"{x:.6f}")
            normalized.append(f"{y:.6f}")
            points.append((x, y))

        if not Polygon(points).is_valid:
            continue

        lines.append(f"{target_class} " + " ".join(normalized))
    return lines


def build_category_mapping(categories: list) -> dict:
    """根据json里的categories列表(id+name)，动态生成 category_id -> 目标类别索引 的映射"""
    mapping = {}
    for cat in categories:
        name = cat["name"]
        if name in FASHIONPEDIA_NAME_TO_TARGET_INDEX:
            mapping[cat["id"]] = FASHIONPEDIA_NAME_TO_TARGET_INDEX[name]
    return mapping


def convert_fashionpedia_split(
    annotation_json_path: str,
    image_dir: str,
    output_root: str,
    output_split: str,
    filename_prefix: str = "fp_",
) -> None:
    with open(annotation_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    category_id_to_target = build_category_mapping(data["categories"])
    print(f"匹配到的鞋包配饰类别: {category_id_to_target}")

    images_by_id = {img["id"]: img for img in data["images"]}

    annos_by_image = defaultdict(list)
    for anno in data["annotations"]:
        if anno["category_id"] in category_id_to_target:
            annos_by_image[anno["image_id"]].append(anno)

    print(f"共 {len(annos_by_image)} 张图片包含目标类别（鞋子/包包/配饰）")

    out_image_dir = Path(output_root) / "images" / output_split
    out_label_dir = Path(output_root) / "labels" / output_split
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    for image_id, annos in annos_by_image.items():
        img_info = images_by_id.get(image_id)
        if img_info is None:
            continue

        file_name = img_info["file_name"]
        img_w, img_h = img_info["width"], img_info["height"]
        src_image_path = Path(image_dir) / file_name
        if not src_image_path.exists():
            continue

        lines = []
        for anno in annos:
            target_class = category_id_to_target[anno["category_id"]]
            lines.extend(segmentation_to_yolo_lines(anno["segmentation"], target_class, img_w, img_h))

        if not lines:
            continue

        new_name = f"{filename_prefix}{file_name}"
        out_image_path = out_image_dir / new_name
        if not out_image_path.exists():
            os.symlink(src_image_path.resolve(), out_image_path)

        out_label_path = out_label_dir / (Path(new_name).stem + ".txt")
        with open(out_label_path, "w") as f:
            f.write("\n".join(lines))

        kept += 1

    print(f"[{output_split}] 转换完成，保留 {kept} 张有效图片")
