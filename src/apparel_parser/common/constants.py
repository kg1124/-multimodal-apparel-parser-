# DeepFashion2 原始13类别（category_id: 官方英文名）
DEEPFASHION2_CATEGORIES = {
    1: "short_sleeve_top",
    2: "long_sleeve_top",
    3: "short_sleeve_outwear",
    4: "long_sleeve_outwear",
    5: "vest",
    6: "sling",
    7: "shorts",
    8: "trousers",
    9: "skirt",
    10: "short_sleeve_dress",
    11: "long_sleeve_dress",
    12: "vest_dress",
    13: "sling_dress",
}

# PRD要求的8大类，YOLO训练用的类别索引（0开始）
TARGET_CLASSES = {
    0: "上衣",
    1: "外套",
    2: "裤子",
    3: "裙子",
    4: "连衣裙",
    5: "鞋子",
    6: "包包",
    7: "配饰",
}

# DeepFashion2 的 category_id -> 目标类别索引 的映射表
CATEGORY_ID_TO_TARGET_INDEX = {
    1: 0,
    2: 0,
    5: 0,
    6: 0,
    3: 1,
    4: 1,
    7: 2,
    8: 2,
    9: 3,
    10: 4,
    11: 4,
    12: 4,
    13: 4,
}

# Fashionpedia 类别名称 -> 目标类别索引 的映射表
# 按名称匹配（不写死数字id），因为具体id要从下载的json里的categories列表动态读取
FASHIONPEDIA_NAME_TO_TARGET_INDEX = {
    "shoe": 5,
    "bag, wallet": 6,
    "glasses": 7,
    "hat": 7,
    "headband, head covering, hair accessory": 7,
    "tie": 7,
    "glove": 7,
    "watch": 7,
    "belt": 7,
    "leg warmer": 7,
    "tights, stockings": 7,
    "sock": 7,
    "scarf": 7,
    "umbrella": 7,
}