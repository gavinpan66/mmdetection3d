metainfo = {
                'categories': {
                                'Pedestrian': 0,
                                'Cyclist': 1,
                                'Car': 2,
                                'Truck': 3,
                                'EngineeringTruck': 4,
                                'Corner':5,
                                'DontCare': -1
                            },
                'dataset': 'kitti',
                'info_version': '1.1'
}

import pickle
import cv2
from PIL import Image
import os
import json
import math
import numpy as np
import sys
import argparse


def get_trans_info(src_calib):
    # open json file
    with open(src_calib, 'r') as file:
        calib_data = json.load(file)
    front_6mm = calib_data['front_6mm']
    images_wh = [front_6mm['width'],front_6mm['height']]
    intrinsics = np.array(front_6mm['Intrinsics']).reshape((4,4))
    extrinsics = np.array(front_6mm['Extrinsics']).reshape((4,4))
    cam2img = intrinsics
    lidar2cam = extrinsics
    lidar2img = np.dot(cam2img,lidar2cam)
    imu2lidar = np.eye(4)
    return images_wh,cam2img.tolist(),lidar2img.tolist(),lidar2cam.tolist(),imu2lidar.tolist()



def main():
    calib_file = "data/wxwc/calib/camera.json"
    annotation_file = "data/wxwc/3d_label/val_data.json"
    pkl_path = "./annotation_val.pkl"
    

    images_hw,cam2img, lidar2img, lidar2cam, imu2lidar=get_trans_info(calib_file)
    data_list = []

    with open(annotation_file, 'r') as file:
        anno_datas = json.load(file)['data']
    for id_name in anno_datas:
        anno = anno_datas[id_name]
        data = {}
        data['sample_idx'] = anno['pc_path']
        images = {}
        R0_rect = np.eye(4)
        images['R0_rect'] = R0_rect.tolist()

        CAM2 = {}
        CAM2['img_path'] = os.path.join(anno['pc_path'] + ".jpg")
        CAM2['height'] = images_hw[0]
        CAM2['width'] = images_hw[1]
        CAM2['cam2img'] = cam2img
        CAM2['lidar2img'] = lidar2img
        CAM2['lidar2cam'] = lidar2cam
        images['CAM2'] = CAM2

        lidar_points = {}
        lidar_points['num_pts_feats'] = 4
        lidar_points['lidar_path'] = os.path.join(anno['pc_path']+ ".bin")
        #lidar_points['Tr_velo_to_cam'] = (np.dot(np.linalg.inv(R0_rect), lidar2cam)).tolist()
        lidar_points['Tr_velo_to_cam'] = lidar2cam
        lidar_points['Tr_imu_to_velo'] = imu2lidar

        instances = []
        bboxes = anno['3D_instances']
        index = 0
        for bbox in bboxes:
            obj_info = {}
            obj_type = bbox['label']
            label = metainfo['categories'][obj_type]
            box = bbox['bbox']
            #滤除超出范围的bbox
            if(box[0]>70 or box[0]<-70 or box[1]>70 or box[1]<-70):
                label = -1
            obj_info['bbox'] = [0,0,50,50]
            obj_info['bbox_label'] = label
            point = np.dot(np.array(lidar2cam),np.array([box[0],box[1],box[2]-(box[5]/2),1]))
            c_x, c_y, c_z = point[:3]
            yaw = -math.pi/2 - box[8]
            obj_info['bbox_3d'] = [round(val,2) for val in [c_x, c_y, c_z, box[3],box[5],box[4],yaw]]
            obj_info['bbox_label_3d'] = label
            obj_info['depth'] = c_z
            obj_info['center_2d'] = [25, 25] 
            obj_info['num_lidar_pts'] = 0
            obj_info['difficulty'] = bbox['truncation']
            obj_info['truncated'] = bbox['truncation']
            obj_info['occluded'] = 0
            alpha = (np.arctan2(c_z, c_x) - 0.5 * np.pi + yaw)
            obj_info['alpha'] = box[8]#alpha
            obj_info['score'] = 0.0 
            obj_info['index'] = index
            obj_info['group_id'] = index
            index += 1
            instances.append(obj_info)
        data['images'] = images
        data['lidar_points'] = lidar_points
        data['instances'] = instances
        data_list.append(data)

    kitti_data = {}
    kitti_data['metainfo'] = metainfo
    kitti_data['data_list'] = data_list

    with open(pkl_path, 'wb') as pkl_file:
        pickle.dump(kitti_data, pkl_file)



    

if __name__ == "__main__":
    main()

    