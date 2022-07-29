from .manager import CocoDataManager
from cogworks_data.language import get_data_path
from pathlib import Path
import json

def load_coco(path, resnet):
    filename = get_data_path(path)
    with Path(filename).open() as f:
        coco_data = json.load(f)
    return CocoDataManager(coco_data, resnet)
