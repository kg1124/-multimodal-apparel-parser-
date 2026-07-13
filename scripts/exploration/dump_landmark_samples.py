import json
import random
from pathlib import Path
from collections import defaultdict

import cv2

annos_dir = Path("/root/autodl-tmp/project/deepfashion2.zip/train/annos")
image_dir = Path("/root/autodl-tmp/project/deepfashion2.zip/train/image")
out_dir = Path("/root/autodl-tmp/project/landmark_samples")
out_dir.mkdir(exist_ok=True)

TARGET_CATEGORIES = ["short_sleeve_top", "trousers", "skirt"]
NEED_PER_CATEGORY = 5

json_files = list(annos_dir.glob("*.json"))
random.seed(2)
random.shuffle(json_files)

collected = defaultdict(int)

for json_path in json_files:
    if all(collected[c] >= NEED_PER_CATEGORY for c in TARGET_CATEGORIES):
        break
    with open(json_path) as f:
        data = json.load(f)
    for key, item in data.items():
        if not key.startswith("item"):
            continue
        cat = item.get("category_name")
        if cat not in TARGET_CATEGORIES or collected[cat] >= NEED_PER_CATEGORY:
            continue
        if item.get("viewpoint") != 2:  # 只要正面视角
            continue

        image_path = image_dir / (json_path.stem + ".jpg")
        img = cv2.imread(str(image_path))
        if img is None:
            continue

        landmarks = item.get("landmarks", [])
        for i in range(0, len(landmarks), 3):
            x, y, v = landmarks[i], landmarks[i+1], landmarks[i+2]
            idx = i // 3
            if v == 0:
                continue
            color = (0, 255, 0) if v == 2 else (0, 165, 255)
            cv2.circle(img, (int(x), int(y)), 4, color, -1)
            cv2.putText(img, str(idx), (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out_path = out_dir / f"{cat}_{collected[cat]}_{json_path.stem}.jpg"
        cv2.imwrite(str(out_path), img)
        collected[cat] += 1

for cat in TARGET_CATEGORIES:
    print(f"{cat}: 保存了 {collected[cat]} 张")
