# Config for running pre-trained COCO MaskFormer on the valid dataset (Option 2)
# Uses COCO's 133 classes but evaluates on valid dataset
# Only COCO-matching categories in valid dataset will contribute to metrics

_base_ = [
    '../_base_/default_runtime.py',
]

# Model config - MUST match pre-trained checkpoint (133 classes = 80 things + 53 stuff)
num_things_classes = 80
num_stuff_classes = 53
num_classes = num_things_classes + num_stuff_classes

model = dict(
    type='MaskFormer',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=True,
        seg_pad_value=255),
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth')),
    panoptic_head=dict(
        type='MaskFormerHead',
        in_channels=[192, 384, 768, 1536],
        feat_channels=256,
        out_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=100,
        pixel_decoder=dict(
            type='PixelDecoder',
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU')),
        enforce_decoder_input_project=True,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(
            num_layers=6,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256, num_heads=8, dropout=0.1, batch_first=True),
                cross_attn_cfg=dict(
                    embed_dims=256, num_heads=8, dropout=0.1, batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=dict(type='ReLU', inplace=True))),
            return_intermediate=True),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=20.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=1.0)),
    panoptic_fusion_head=dict(
        type='MaskFormerFusionHead',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_panoptic=None,
        init_cfg=None),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='ClassificationCost', weight=1.0),
                dict(type='FocalLossCost', weight=20.0, binary_input=True),
                dict(type='DiceCost', weight=1.0, pred_act=True, eps=1.0)
            ]),
        sampler=dict(type='MaskPseudoSampler')),
    test_cfg=dict(
        panoptic_on=True,
        semantic_on=False,
        instance_on=False,
        max_per_image=100,
        iou_thr=0.8,
        object_mask_thr=0.8,
        filter_low_score=False),
    init_cfg=None)

# Dataset config - point to valid dataset but use CocoPanopticDataset (no custom metainfo)
# Since valid uses COCO category IDs, the default COCO class mapping will work
data_root = 'data/valid/'
backend_args = None

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadPanopticAnnotations', backend_args=backend_args),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoPanopticDataset',
        data_root=data_root,
        ann_file='annotations/panoptic_val_coco_format.json',  # Full COCO categories, valid data
        data_prefix=dict(img='val/', seg='annotations/panoptic_val/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

# Evaluator - uses annotation with full COCO category definitions
test_evaluator = dict(
    type='CocoPanopticMetric',
    ann_file='data/valid/annotations/panoptic_val_coco_format.json',
    seg_prefix='data/valid/annotations/panoptic_val/',
    backend_args=backend_args)

# Disable visualization hook to avoid version issues
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'))

test_cfg = dict(type='TestLoop')

