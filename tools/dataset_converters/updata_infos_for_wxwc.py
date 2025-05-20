import mmengine
from pathlib import Path
import numpy as np
import copy
from os import path as osp
#instance 空白模版
def get_empty_instance():
    """Empty annotation for single instance."""
    instance = dict(
        # (list[float], required): list of 4 numbers representing
        # the bounding box of the instance, in (x1, y1, x2, y2) order.
        bbox=None,
        # (int, required): an integer in the range
        # [0, num_categories-1] representing the category label.
        bbox_label=None,
        #  (list[float], optional): list of 7 (or 9) numbers representing
        #  the 3D bounding box of the instance,
        #  in [x, y, z, w, h, l, yaw]
        #  (or [x, y, z, w, h, l, yaw, vx, vy]) order.
        bbox_3d=None,
        # (bool, optional): Whether to use the
        # 3D bounding box during training.
        bbox_3d_isvalid=None,
        # (int, optional): 3D category label
        # (typically the same as label).
        bbox_label_3d=None,
        # (float, optional): Projected center depth of the
        # 3D bounding box compared to the image plane.
        depth=None,
        #  (list[float], optional): Projected
        #  2D center of the 3D bounding box.
        center_2d=None,
        # (int, optional): Attribute labels
        # (fine-grained labels such as stopping, moving, ignore, crowd).
        attr_label=None,
        # (int, optional): The number of LiDAR
        # points in the 3D bounding box.
        num_lidar_pts=None,
        # (int, optional): The number of Radar
        # points in the 3D bounding box.
        num_radar_pts=None,
        # (int, optional): Difficulty level of
        # detecting the 3D bounding box.
        difficulty=None,
        #trcking id(int, optional): The tracking id of the instance.
        tracking_id=None,
        unaligned_bbox_3d=None)
    return instance

def get_empty_img_info():
    img_info = dict(
        # (str, required): the path to the image file.
        img_path=None,
        # (int) The height of the image.
        height=None,
        # (int) The width of the image.
        width=None,
        # (str, optional): Path of the depth map file
        depth_map=None,
        # (list[list[float]], optional) : Transformation
        # matrix from camera to image with
        # shape [3, 3], [3, 4] or [4, 4].
        cam2img=None,
        # (list[list[float]]): Transformation matrix from lidar
        # or depth to image with shape [4, 4].
        lidar2img=None,
        # (list[list[float]], optional) : Transformation
        # matrix from camera to ego-vehicle
        # with shape [4, 4].
        cam2ego=None)
    return img_info

def get_singel_image_sweep(camera_types):
    single_image_sweep = dict(
        timestamp=None,
        ego2global=None)
    images = dict()
    for cam_type in camera_types:
        images[cam_type] = get_empty_img_info()
    single_image_sweep['images'] = images
    return single_image_sweep

def get_empty_standard_data_info(
        camera_types=['front', 'front_6mm', 'front_left', 'front_right', 'rear_left','rear_right']):    
    data_info = dict(
        sample_idx=None,
        token=None,
        **get_singel_image_sweep(camera_types),
        lidar_points= dict(
            num_pts_feats=None,
            lidar_path=None,
            lidar2ego=None,
        ),
        instances=[])
    return data_info


def updata_infos_for_wxwc(pkl_path, out_dir):
    camera_types = ['front', 'front_6mm', 'front_left', 'front_right', 'rear_left','rear_right']
    print(f'{pkl_path} will be modified.')

    METAINFO = {'classes': ('Car', 'Truck','Pedestrian', 'Cyclist', 'Corner','EngineeringTruck')}
    data_list = mmengine.load(pkl_path)
    converted_list = []
    for ori_info_dict in mmengine.track_iter_progress(data_list):
        #获取数据格式的空白模版
        temp_data_info = get_empty_standard_data_info(camera_types)
        ori_info_dict['sample_idx'] = ori_info_dict['point_cloud']['pc_path']

        #TODO:添加calib matrix

        #image path
        for cam_idx, cam_key in enumerate(camera_types):
            temp_data_info['images'][cam_key]['img_path']= ori_info_dict['image'][cam_key]
        
        #TODO:for the sweeps part:

        #for annotations part:
        anns = ori_info_dict.get('annos', None)
        if anns is not None:
            num_instances = len(anns)
            isinstance_list = []
            for instance_id in range(num_instances):
                #bbox x,y,z,w,h,l,row,pitch,yaw
                empty_instance = get_empty_instance()
                empty_instance['bbox'] = anns[instance_id]['bbox']

                #bbox_label
                if anns[instance_id]['label'] in METAINFO['classes']:
                    empty_instance['bbox_label'] = METAINFO['classes'].index(anns[instance_id]['label'])
                else:
                    empty_instance['bbox_label'] = -1

                #bbox_3d
                # loc = anns[instance_id]['bbox'][0:3]
                # dims = anns[instance_id]['bbox'][3:6]
                # rots = anns[instance_id]['bbox'][-1]
                loc = anns[instance_id]['bbox']
                empty_instance['bbox_3d'] = np.array([loc[0],loc[1],loc[2],loc[3],loc[4],loc[5],loc[-1]]).astype(np.float32).tolist()
                empty_instance['bbox_label_3d'] = copy.deepcopy(empty_instance['bbox_label'])

                #tracking_id
                empty_instance['tracking_id'] = int(anns[instance_id]['track_id'])

                #truncated
                empty_instance['truncated'] = int(anns[instance_id]['truncation'])

                isinstance_list.append(empty_instance)
            temp_data_info['instances'] = isinstance_list
        converted_list.append(temp_data_info)

    pkl_name = Path(pkl_path).name
    out_path = osp.join(out_dir, pkl_name)
    print(f'Writing to output file: {out_path}.')

    # dataset metainfo
    metainfo = dict()
    metainfo['categories'] = {k: i for i, k in enumerate(METAINFO['classes'])}
    metainfo['dataset'] = 'wxwc'
    metainfo['version'] = '1.1'
    metainfo['info_version'] = '1.1'

    converted_data_info = dict(metainfo=metainfo, data_list=converted_list)

    mmengine.dump(converted_data_info, out_path, 'pkl')





