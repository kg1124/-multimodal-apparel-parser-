import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apparel_parser.data.convert_fashionpedia import convert_fashionpedia_split

FASHIONPEDIA_ROOT = "/root/autodl-tmp/project/fashionpedia"
OUTPUT_ROOT = "/root/autodl-tmp/project/yolo_dataset"

if __name__ == "__main__":
    convert_fashionpedia_split(
        annotation_json_path=f"{FASHIONPEDIA_ROOT}/instances_attributes_train2020.json",
        image_dir=f"{FASHIONPEDIA_ROOT}/train",
        output_root=OUTPUT_ROOT,
        output_split="train",
    )
    convert_fashionpedia_split(
        annotation_json_path=f"{FASHIONPEDIA_ROOT}/instances_attributes_val2020.json",
        image_dir=f"{FASHIONPEDIA_ROOT}/test",
        output_root=OUTPUT_ROOT,
        output_split="val",
    )