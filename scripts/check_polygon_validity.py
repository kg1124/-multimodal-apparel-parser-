import sys
from pathlib import Path
from collections import defaultdict

from shapely.geometry import Polygon
from shapely.validation import explain_validity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from apparel_parser.common.constants import TARGET_CLASSES


def check_labels_for_class(dataset_root: str, split: str, target_class: int):
    label_dir = Path(dataset_root) / "labels" / split
    total = 0
    invalid = 0
    invalid_files = []

    for label_path in label_dir.glob("*.txt"):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts or int(parts[0]) != target_class:
                    continue
                total += 1
                coords = [float(x) for x in parts[1:]]
                points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
                if len(points) < 3:
                    invalid += 1
                    invalid_files.append((label_path.name, "点数不足3个"))
                    continue
                poly = Polygon(points)
                if not poly.is_valid:
                    invalid += 1
                    invalid_files.append((label_path.name, explain_validity(poly)))

    print(f"类别 {target_class}({TARGET_CLASSES[target_class]}): 共 {total} 个实例，{invalid} 个自相交/无效（占比 {invalid/total*100:.1f}%）")
    print("前10个问题文件示例：")
    for name, reason in invalid_files[:10]:
        print(f"  {name}: {reason}")
    return invalid_files


if __name__ == "__main__":
    for cls in [1, 2]:  # 外套=1, 裤子=2
        check_labels_for_class("/root/autodl-tmp/project/yolo_dataset", "train", cls)
        print()
