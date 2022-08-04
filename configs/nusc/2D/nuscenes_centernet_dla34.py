import itertools
import logging

from det3d.utils.config_tool import get_downsample_factor

tasks = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=1, class_names=["barrier"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)

# model settings
model = dict(
    type="DddHead",
    tasks=tasks,
    backbone=dict(
        type="DLASegv2",
        base_name='dla34',
        pretrained=True,
        heads={'hm': 10,'dep': 1,'wh': 2,'reg': 2,'rot': 8,'dim': 3},
        down_ratio=4,
        final_kernel=1,
        last_level=5,
        head_conv=256,),
)

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=4,
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    cam_down_ratio=model['backbone']['down_ratio'],
)


train_cfg = dict(assigner=assigner)


test_cfg = dict(
    post_center_limit_range=[-80, -80, -10.0, 80, 80, 10.0],
    nms=dict(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=4096,
        nms_post_max_size=500,
        nms_iou_threshold=0.7,
    ),
    score_threshold=0.1,
    pc_range=[-75.2, -75.2],
    out_size_factor=4,
    voxel_size=[0.1, 0.1],
)


# dataset settings
dataset_type = "NuScenesDataset"
nsweeps = 10
data_root = "/data_nuscenes"

db_sampler = dict(
    type="GT-AUG",
    enable=False,
    db_info_path="/nuscenes_data/dbinfos_train_10sweeps_withvelo.pkl",
    sample_groups=[
        dict(car=2),
        dict(truck=3),
        dict(construction_vehicle=7),
        dict(bus=4),
        dict(trailer=6),
        dict(barrier=2),
        dict(motorcycle=6),
        dict(bicycle=6),
        dict(pedestrian=2),
        dict(traffic_cone=2),
    ],
    db_prep_steps=[
        dict(
            filter_by_min_num_points=dict(
                car=5,
                truck=5,
                bus=5,
                trailer=5,
                construction_vehicle=5,
                traffic_cone=5,
                barrier=5,
                motorcycle=5,
                bicycle=5,
                pedestrian=5,
            )
        ),
        dict(filter_by_difficulty=[-1],),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
) 

train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    global_rot_noise=[-0.78539816, 0.78539816],
    global_scale_noise=[0.95, 1.05],
    db_sampler=db_sampler,
    class_names=class_names,
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
)

voxel_generator = dict(
    range=[-75.2, -75.2, -2, 75.2, 75.2, 4],
    voxel_size=[0.1, 0.1, 0.15],
    max_points_in_voxel=5,
    max_voxel_num=150000,
    out_size_factor=4,
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

train_anno = "/nuscenes_data/infos_train_10sweeps_withvelo_filter_True.pkl"
val_anno = "/nuscenes_data/infos_val_10sweeps_withvelo_filter_True.pkl"
test_anno = None

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        ann_file=val_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        ann_file=test_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)



optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# optimizer
optimizer = dict(
    type="Adam", lr=5e-5,amsgrad=False, weight_decay=0.01,
)
lr_config = dict(
    type="manual_stepping"
)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# runtime settings
total_epochs = 6
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None 
resume_from = None  
workflow = [('train', 1)]
drop_step = [2000]