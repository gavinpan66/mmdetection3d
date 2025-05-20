import mmengine
import numpy as np
import json
from pathlib import Path
from updata_infos_for_wxwc import updata_infos_for_wxwc
from os import path as osp




def get_wxwc_info(data_path,json_file):
    result = []
    with open(json_file,'r') as f:
        datas = json.load(f)

    datas = datas['data']
    keys = datas.keys()
    for key in keys:
        info = {}
        data = datas[key]
        
        #pc_info
        pc_info = {}
        pc_info['num_features'] = data['num_pts_feats']
        pc_info['pc_path'] = key+'.bin'
        info['point_cloud'] = pc_info

        #images
        images_info ={}
        images_info['front'] = data['images']['front']['image_path']
        images_info['front_6mm'] = data['images']['front_6mm']['image_path']
        images_info['front_left'] = data['images']['front_left']['image_path']
        images_info['front_right'] = data['images']['front_right']['image_path']
        images_info['rear_left'] = data['images']['rear_left']['image_path']
        images_info['rear_right'] = data['images']['rear_right']['image_path']
        info['image'] = images_info

        #annotation
        annotation_info=[]
        if(len(data['3D_instances'])):
            for instance in data['3D_instances']:
                object = {}
                object['bbox'] = instance['bbox']
                object['label'] = instance['label']
                object['track_id'] = instance['track_id']
                object['truncation'] = instance['truncation']
                annotation_info.append(object)
        info['annos'] = annotation_info
        result.append(info)
    return result



def create_wxwc_info_file(data_path,
                          out_dir = None,
                          pkl_prefix='wxwc',
                          relative_path=True):
    """Create info file of wxwc dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str, optional): Prefix of the info file to be generated.
            Default: 'wxwc'.
        save_path (str, optional): Path to save the info file.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
    """
    train_json = Path(data_path) / 'train.json'
    val_json = Path(data_path) / 'val.json'
    print("generate info. it may take several minutes.")

    wxwc_info_train = get_wxwc_info(data_path,train_json)
    filename = Path(data_path) / f'{pkl_prefix}_infos_train.pkl'
    mmengine.dump(wxwc_info_train,filename)
    updata_infos_for_wxwc(filename,out_dir)



    wxwc_info_val = get_wxwc_info(data_path,val_json)
    filename = Path(data_path) / f'{pkl_prefix}_infos_val.pkl'
    mmengine.dump(wxwc_info_val,filename)
    updata_infos_for_wxwc(filename,out_dir)

    


