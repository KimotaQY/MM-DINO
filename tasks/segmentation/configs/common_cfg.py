MS_ROOT_DIR = "/home/yyyjvm"


def get_labels(dataset_name=None):
    if dataset_name is None:
        raise ValueError("Please specify a dataset")

    if dataset_name == "Vaihingen" or dataset_name == "Potsdam":
        labels = [
            "roads", "buildings", "low veg.", "trees", "cars", "clutter"
        ]  # Label names
    elif dataset_name == "YYYJ":
        labels = [
            "地基建设", "基础结构建设", "封顶厂房", "封顶楼房", "施工道路", "硬化道路", "风电施工", "风电",
            "光伏", "推填土", "体育场地", "临时棚房", "自建房", "专属设施", "未定义"
        ]
    elif dataset_name == "EarthMiss":
        labels = [
            "Background", "Building", "Road", "Water", "Barren", "Forest",
            "Agricultural", "Playground"
        ]
    elif dataset_name == "WHU":
        labels = ["农田", "城市", "村庄", "水体", "森林", "道路", "其他"]

    return labels
