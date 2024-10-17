from mmengine.registry import Registry
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS


MODELS = Registry(
    'model', parent=MMENGINE_MODELS, locations=['src.models'])
DATASETS = Registry(
    'dataset', parent=MMENGINE_DATASETS, locations=['src.datasets'])
METRICS = Registry(
    'metric', parent=MMENGINE_METRICS, locations=['src.utils'])
TRANSFORMS = Registry(
    'transform', parent=MMENGINE_TRANSFORMS, locations=['src.datasets.transforms'])
