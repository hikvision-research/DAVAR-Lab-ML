"""
##################################################################################################
# Filename       :    fasterrcnn_coco_config.py
# Abstract       :    Config file of Raster RCNN model for coco dataset.

# Current Version:    1.0.0
# Date           :    2022-03-03
##################################################################################################
"""

"""
1. model settings
"""
root_path = '/your/project/path/datalists/'
model = dict(
    type='KD_Two_Stage_Detector',
    pretrained=None,
    student= dict(
        pretrained=root_path + 'checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth',
        backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)))),
    teacher = dict(
        pretrained=root_path + 'checkpoints/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth',
        backbone=dict(
            type='ResNet',
            depth=101,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch'),
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5),
        rpn_head=dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[8],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        roi_head=dict(
            type='StandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0)))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)),
    # knowledge distillation cfg
    kd_cfg=dict(
        with_kd = True, # whether to use knowledge distillation
        st_neck_channels=[256,256,256,256,256], # channels of student's neck (FPN)
        tc_neck_channels=[256,256,256,256,256], # channels of teacher's neck (FPN)
        tea_mode='eval',
        adpt_activation = 'relu',
        prototype_each_epoch = 1, # frequence of update the prototypes
        strategy=dict(
            kd_loss=dict(type='TwoStageDetStrategy', # kd loss for object detection
                start_epoch=0,    # the start epoch for using kd
                distributed=True,
                kd_attenuation = True, # whether to use kd attenuation
                use_soft_label = True, # whether to use soft label kd
                use_point_kd = True, # whether to use the point-wise kd
                point_strategy = 'decouple', # fgfi or decouple
                proposal_replace_gt = True, # whether to use proposals to replace gt bboxes when generating mask
                use_pair_kd = True, # whether to usw similarity (pair-wise) kd loss
                # pair_wise choices are ['use_pgm', 'use_rkd', 'robustness_for_point']
                pair_wise = 'use_pgm', # use which pair-wise kd function
                kd_param=[[1, 0], [1], [4, 1]], # kd loss factor of [[neck point-wise, backbone(not used)],[soft target]]
                neck_kd_lambda=(1, 1, 1, 1, 1), # kd loss factor of each neck layer
                kd_t=2, # temperature of soft target
                pair_wise_factor=5) # kd loss factor of pair_wise
            )
    )
)


"""
2. dataset settings
"""
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (1333, 800)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='KDLoadAnnotations', #'LoadAnnotations',
         with_bbox=True,            # Bounding Rect
         with_label=True,           # Bboxes' labels
         label_start_index=0
         ),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data_root=root_path + 'datalists/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='KDDataset',
        ann_file=data_root + 'instances_train2017.json',
        img_prefix='/root/path/of/image/',
        classes_config=data_root + 'classes_config.json',
        pipeline=train_pipeline),
    val=dict(
        type='CocoDataset',
        ann_file=data_root + 'instances_val2017.json',
        img_prefix='/root/path/of/image/',
        pipeline=test_pipeline),
    test=dict(
        type='CocoDataset',
        ann_file=data_root + 'instances_val2017.json',
        img_prefix='/root/path/of/image/',
        pipeline=test_pipeline))


"""
3. trainning settings
"""
evaluation = dict(interval=1, metric='bbox')
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])

total_epochs = 24


"""
4. running settings
"""
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHookKD'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]


"""
5. prototype generation module cfg
"""
pgm_cfg = dict(
    strategy_cfg=dict(
        type='PrototypeGenerationModule',
        temp_dir=None,
        silent_mode=False,
        out_dir=None,
        save_ckpt_min=20,
    ),
    runtime_cfg=dict(
        model_cfg_paths=[root_path + 'configs/faster_rcnn_coco_101.py',  # teacher model config
                         root_path + 'configs/faster_rcnn_coco_50.py'],  # student model config
        checkpoints=[root_path + 'checkpoints/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth',  # teacher model checkpoint
                     root_path + 'checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'],  # student model checkpoint
        pseudolabel_path=None,
        cls_names=[],  # classes to generate prototypes, will reset in pl_gen.py
        conf_thr=0.90,  # pgm will filter the bboxes whose conf bigger than conf_thr as the candidates of prototype
        img_prefix='/root/path/of/image/',
        max_samples=800,  # max number of candidates 
        max_num_prototypes=10,  # max number of prototypes
        alpha=10,  # trade-off weight controlling the regularization term in seleting prototypes
        vis_prototype=False,  # whether to save the images of prototypes
        save_path='./',  # the path of saving prototypes' features, will change before every epoch
        pgm_dataset={},  # dataset information to generate prototypes
        alter_path='./'  # alternative path to load alternative prototypes' features when generate prototypes failed, will reset in pl_gen.py
    )
)