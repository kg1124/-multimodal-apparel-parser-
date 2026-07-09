import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from apparel_parser.common.constants import TARGET_CLASSES

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
]


def draw_labels_on_image(image_path: Path, label_path: Path, output_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"无法读取图片: {image_path}")
        return
    h, w = img.shape[:2]

    if not label_path.exists():
        print(f"没有对应的标签文件: {label_path}")
        return

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        class_idx = int(parts[0])
        coords = [float(x) for x in parts[1:]]
        points = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * w)
            y = int(coords[i + 1] * h)
            points.append([x, y])
        points_arr = np.array([points], dtype=np.int32)
        color = COLORS[class_idx % len(COLORS)]
        cv2.polylines(img, points_arr, isClosed=True, color=color, thickness=2)
        class_name = TARGET_CLASSES.get(class_idx, str(class_idx))
        cv2.putText(img, class_name, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(str(output_path), img)
    print(f"已保存: {output_path}")


def visualize_random_samples(dataset_root: str, split: str, num_samples: int, output_dir: str):
    image_dir = Path(dataset_root) / "images" / split
    label_dir = Path(dataset_root) / "labels" / split
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    image_files = list(image_dir.glob("*.jpg"))
    samples = random.sample(image_files, min(num_samples, len(image_files)))

    for image_path in samples:
        label_path = label_dir / (image_path.stem + ".txt")
        output_path = output_dir_path / image_path.name
        draw_labels_on_image(image_path, label_path, output_path)


def visualize_samples_for_class(dataset_root: str, split: str, target_class: int, num_samples: int, output_dir: str):
    """只挑选标签文件里包含指定类别的样本，专项排查某一类的问题（找够数量就提前停止，不用扫描全部文件）"""
    image_dir = Path(dataset_root) / "images" / split
    label_dir = Path(dataset_root) / "labels" / split
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    matching_labels = []
    target_str = str(target_class)
    for label_path in label_dir.glob("*.txt"):
        with open(label_path, "r") as f:
            first_col = [line.split()[0] for line in f if line.strip()]
        if target_str in first_col:
            matching_labels.append(label_path)
        if len(matching_labels) >= num_samples * 5:
            break

    print(f"扫描到 {len(matching_labels)} 个包含类别 {target_class} 的标签文件（提前停止，不代表总数）")
    samples = random.sample(matching_labels, min(num_samples, len(matching_labels)))

    for label_path in samples:
        image_path = image_dir / (label_path.stem + ".jpg")
        if not image_path.exists():
            continue
        output_path = output_dir_path / image_path.name
        draw_labels_on_image(image_path, label_path, output_path)


if __name__ == "__main__":
    visualize_samples_for_class(
        dataset_root="/root/autodl-tmp/project/yolo_dataset",
        split="train",
        target_class=2,
        num_samples=10,
        output_dir="/root/autodl-tmp/project/vis_samples_pants",
    )
