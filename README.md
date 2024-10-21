# Gesture Recognition using GNN

## Model: ST_GCN

Train with seed: 
```
python train.py [CONFIG_PATH] --cfg-options randomness.seed=[SEED]
```
For example, 
- Train with mediapipe keypoints: 
    ```
    python train.py configs/stgcn_mediapipe_keypoint.py --cfg-options randomness.seed=42
    ```
- Train with mmpose keypoints (coco format)
    ```
    python train.py configs/stgcn_mmpose_keypoint.py --cfg-options randomness.seed=42
    ```

Test with checkpoint:
```
python test.py [CONFIG_PATH] [CHECKPOINT_PATH]
```

## Modification

- Want to update graph or GCN module?

    ==> See file `src/models/backbones/utils/gcn_utils.py`  and `src/models/backbones/utils/graph.py`

- Want to update/add new model architecture

    ==> See files in folder `src/models/backbones` and `src/models/heads`
