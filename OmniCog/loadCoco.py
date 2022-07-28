from .manager import CocoDataManager
def load_coco(path):
    filename = get_data_path(path)
    with Path(filename).open() as f:
        coco_data = json.load(f)
    return CocoDataManager(coco_data)
