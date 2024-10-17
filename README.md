# Gesture Recognition using GNN

## Model: ST_GCN

Train with mediapipe keypoints: 
```
python train.py configs/stgcn_mediapipe_keypoint.py 
```

Train with mmpose keypoints (coco format)
```
python train.py configs/stgcn_mmpose_keypoint.py
```

Test with checkpoint:
```
python test.py [CONFIG_PATH] [CHECKPOINT_PATH]
```

Want to update graph or GCN module?
==> See file `src/models/backbones/utils/gcn_utils.py`  and `src/models/backbones/utils/graph.py`
