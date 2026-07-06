import random
from pathlib import Path

import cv2
import numpy as np

from apparel_parser.common.constants import TARGET_CLASSES


def draw_labels_on_image(image_path: Path, label_path: Path, output_path: Path):
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]

    lines = []
    if label_path.exists():
        with open(label_path) as f:
            lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        class_id = int(parts[0])
        coords = [float(v) for v in parts[1:]]
        points = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * w)
            y = int(coords[i + 1] * h)
            points.append((x, y))

        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        label_name = TARGET_CLASSES.get(class_id, str(class_id))
        if points:
            cv2.putText(img, label_name, points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

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


if __name__ == "__main__":
    visualize_random_samples(
        dataset_root="/root/autodl-tmp/project/yolo_dataset",
        split="train",
        num_samples=5,
        output_dir="/root/autodl-tmp/project/vis_samples",
    )
