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
# 注意：鞋子、包包、配饰这3类 DeepFashion2 不包含，暂不训练，后续用 Fashionpedia 数据补充
TARGET_CLASSES = {
    0: "上衣",
    1: "外套",
    2: "裤子",
    3: "裙子",
    4: "连衣裙",
    # 5: "鞋子",   # 缺数据，待补充
    # 6: "包包",   # 缺数据，待补充
    # 7: "配饰",   # 缺数据，待补充
}

# DeepFashion2 的 category_id -> 目标类别索引 的映射表
CATEGORY_ID_TO_TARGET_INDEX = {
    1: 0,   # short_sleeve_top -> 上衣
    2: 0,   # long_sleeve_top  -> 上衣
    5: 0,   # vest             -> 上衣
    6: 0,   # sling            -> 上衣
    3: 1,   # short_sleeve_outwear -> 外套
    4: 1,   # long_sleeve_outwear  -> 外套
    7: 2,   # shorts    -> 裤子
    8: 2,   # trousers  -> 裤子
    9: 3,   # skirt     -> 裙子
    10: 4,  # short_sleeve_dress -> 连衣裙
    11: 4,  # long_sleeve_dress  -> 连衣裙
    12: 4,  # vest_dress         -> 连衣裙
    13: 4,  # sling_dress        -> 连衣裙
}