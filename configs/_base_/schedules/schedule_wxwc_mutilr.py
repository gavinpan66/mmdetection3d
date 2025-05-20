# optimizer

lr = 0.01
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    # max_norm=10 is better for SECOND
    clip_grad=dict(max_norm=35, norm_type=2))


train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=70, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 1000,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='MultiStepLR',
        begin=0,
        end=70,
        by_epoch=True,
        milestones=[25,40,50,60,65],
        gamma=0.6)
]

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=32)
