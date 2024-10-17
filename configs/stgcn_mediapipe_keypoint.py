default_scope = "src"

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

default_hooks = dict(
    runtime_info=dict(type="RuntimeInfoHook"),
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=100, ignore_last=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook", interval=1, save_best="auto", max_keep_ckpts=3
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    sync_buffers=dict(type="SyncBuffersHook"),
)
log_processor = dict(type="LogProcessor", window_size=20, by_epoch=True)

vis_backends = [
    dict(type="LocalVisBackend"),
    # dict(
    #     type='WandbVisBackend',
    #     init_kwargs={
    #         'project': 'gesture-recognition',
    #         'entity': 'tungbui198-hust',
    #         'name': 'stgcn_mediapipe_keypoint'
    #     },
    # ),
]
visualizer = dict(type="Visualizer", vis_backends=vis_backends)

log_level = "INFO"
load_from = None
resume = False

model = dict(
    type="RecognizerGCN",
    backbone=dict(
        type="STGCN", 
        graph_cfg=dict(
            layout=dict(
                num_node=25,
                # inward = [
                #     (0, 1), (1, 2), (2, 3), (3, 7),
                #     (0, 4), (4, 5), (5, 6), (6, 9),
                #     (9, 10), (11, 12), (11, 23), (12, 24), (23, 24),
                #     (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                #     (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20)
                # ],
                inward = [
                    (0, 1), (1, 2), (2, 3), (3, 7),
                    (0, 4), (4, 5), (5, 6), (6, 9),
                    (9, 10), (11, 12), (11, 23), (12, 24), (23, 24),
                    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20)
                ],
                center = 0
            ), 
            mode="stgcn_spatial"
        )
    ),
    cls_head=dict(type="GCNHead", num_classes=10, in_channels=256),
)

dataset_type = "PoseDataset"
train_ann_file = "data/annotations/mediapipe_train.pkl"
test_ann_file = "data/annotations/mediapipe_test.pkl"
train_pipeline = [
    dict(type="PreNormalize2D"),
    dict(type="GenSkeFeat"),
    dict(type="UniformSampleFrames", clip_len=100),
    dict(type="PoseDecode"),
    dict(type="FormatGCNInput", num_person=1),
    dict(type="PackActionInputs"),
]
val_pipeline = [
    dict(type="PreNormalize2D"),
    dict(type="GenSkeFeat"),
    dict(type="UniformSampleFrames", clip_len=100, num_clips=1, test_mode=True),
    dict(type="PoseDecode"),
    dict(type="FormatGCNInput", num_person=1),
    dict(type="PackActionInputs"),
]
test_pipeline = [
    dict(type="PreNormalize2D"),
    dict(type="GenSkeFeat"),
    dict(type="UniformSampleFrames", clip_len=100, num_clips=5, test_mode=True),
    dict(type="PoseDecode"),
    dict(type="FormatGCNInput", num_person=1),
    dict(type="PackActionInputs"),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="RepeatDataset",
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=train_ann_file,
            pipeline=train_pipeline,
        ),
    ),
)
val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type, ann_file=test_ann_file, pipeline=val_pipeline, test_mode=True
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=test_ann_file,
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

val_evaluator = [dict(type="AccMetric")]
test_evaluator = val_evaluator

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=16, val_begin=1, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

param_scheduler = [
    dict(
        type="CosineAnnealingLR",
        eta_min=0,
        T_max=16,
        by_epoch=True,
        convert_to_iter_based=True,
    )
]

optim_wrapper = dict(
    optimizer=dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=128)
