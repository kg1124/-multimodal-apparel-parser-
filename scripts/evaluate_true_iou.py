import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from apparel_parser.common.constants import TARGET_CLASSES

def polygon_label_to_mask(label_path: Path, img_w: int, img_h: int):
    instances = []
    if not label_path.exists():
        return instances
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * img_w)
                y = int(coords[i + 1] * img_h)
                points.append([x, y])
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 1)
            instances.append((cls, mask))
    return instances

def mask_iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return intersection / union if union > 0 else 0.0

def evaluate_true_iou(model_path, dataset_root, split="val", match_threshold=0.5, max_images=10000):
    model = YOLO(model_path)
    image_dir = Path(dataset_root) / "images" / split
    label_dir = Path(dataset_root) / "labels" / split

    import random; image_files = sorted(image_dir.glob("*.jpg")); random.seed(42); random.shuffle(image_files)
    if max_images:
        image_files = image_files[:max_images]

    print(f"共 {len(image_files)} 张图片参与评估")

    per_class_ious = defaultdict(list)
    total_gt = 0
    total_matched = 0

    for idx, image_path in enumerate(image_files):
        if idx % 200 == 0:
            print(f"进度: {idx}/{len(image_files)}")

        label_path = label_dir / (image_path.stem + ".txt")
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        gt_instances = polygon_label_to_mask(label_path, img_w, img_h)
        if not gt_instances:
            continue

        result = model.predict(source=str(image_path), verbose=False)[0]
        pred_instances = []
        if result.masks is not None:
            pred_classes = result.boxes.cls.cpu().numpy().astype(int)
            pred_masks = result.masks.data.cpu().numpy()
            for cls, mask in zip(pred_classes, pred_masks):
                mask_resized = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                pred_instances.append((cls, mask_resized.astype(np.uint8)))

        for gt_cls, gt_mask in gt_instances:
            total_gt += 1
            best_iou = 0.0
            for pred_cls, pred_mask in pred_instances:
                if pred_cls != gt_cls:
                    continue
                iou = mask_iou(gt_mask, pred_mask)
                if iou > best_iou:
                    best_iou = iou
            if best_iou >= match_threshold:
                total_matched += 1
                per_class_ious[gt_cls].append(best_iou)

    print("\n===== 真实平均IoU（PRD定义：分割IoU≥0.85）=====")
    all_ious = []
    for cls in sorted(per_class_ious.keys()):
        ious = per_class_ious[cls]
        avg = sum(ious) / len(ious)
        all_ious.extend(ious)
        print(f"{TARGET_CLASSES[cls]}: 匹配到 {len(ious)} 个实例，平均IoU = {avg:.4f}")

    if all_ious:
        overall_avg = sum(all_ious) / len(all_ious)
        print(f"\n整体平均IoU = {overall_avg:.4f}（PRD目标 ≥0.85）")
    print(f"匹配率: {total_matched}/{total_gt} = {total_matched/total_gt*100:.1f}%")

if __name__ == "__main__":
    evaluate_true_iou(
        model_path="/root/autodl-tmp/project/-multimodal-apparel-parser-/runs/segment/train_final_8class/weights/best.pt",
        dataset_root="/root/autodl-tmp/project/yolo_dataset",
        split="val",
        max_images=10000,
    )
