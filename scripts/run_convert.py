import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apparel_parser.data.convert_deepfashion2 import convert_split

DEEPFASHION2_ROOT = "/root/autodl-tmp/project/deepfashion2.zip"
OUTPUT_ROOT = "/root/autodl-tmp/project/yolo_dataset"

if __name__ == "__main__":
    convert_split(DEEPFASHION2_ROOT, "train", OUTPUT_ROOT)
    convert_split(DEEPFASHION2_ROOT, "validation", OUTPUT_ROOT)
    