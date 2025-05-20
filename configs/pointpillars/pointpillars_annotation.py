_base_ = [
    '../_base_/models/pointpillars_voxel_size_0.25_32annotation.py',
    '../_base_/datasets/annotation_dataset.py',
    '../_base_/schedules/schedule_wxwc_mutilr.py', '../_base_/default_runtime.py'
]





default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1))