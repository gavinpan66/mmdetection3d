# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
from argparse import ArgumentParser

from mmengine.logging import print_log

from mmdet3d.apis import LidarDet3DInferencer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('model', help='Config file')
    parser.add_argument('weights', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.3,
        help='bbox score threshold')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs',
        help='Output directory of prediction and visualization results.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show online visualization results')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=-1,
        help='The interval of show (s). Demo will be blocked in showing'
        'results, if wait_time is -1. Defaults to -1.')
    parser.add_argument(
        '--no-save-vis',
        action='store_true',
        help='Do not save detection visualization results')
    parser.add_argument(
        '--no-save-pred',
        action='store_true',
        help='Do not save detection prediction results')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    call_args = vars(parser.parse_args())

    call_args['inputs'] = dict(points=call_args.pop('pcd'))

    if call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''

    init_kws = ['model', 'weights', 'device']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    # NOTE: If your operating environment does not have a display device,
    # (e.g. a remote server), you can save the predictions and visualize
    # them in local devices.
    if os.environ.get('DISPLAY') is None and call_args['show']:
        print_log(
            'Display device not found. `--show` is forced to False',
            logger='current',
            level=logging.WARNING)
        call_args['show'] = False

    return init_args, call_args


def main():
    # TODO: Support inference of point cloud numpy file.
    #init_args, call_args = parse_args()
    call_args = {'pred_score_thr': 0.4, 'out_dir': 'lidar_bin_result', 'show': False, 'wait_time': -1, 'no_save_vis': True, 'no_save_pred': False, 'print_result': False, 'inputs': ""}
    init_args = {'model': 'work_dirs/pointpillars_voxel_size_0.25_64_shedule_wxwc_mutilr/pointpillars_voxel_size_0.25_64_shedule_wxwc_mutilr.py', 'weights': 'work_dirs/pointpillars_voxel_size_0.25_64_shedule_wxwc_mutilr/epoch_54.pth', 'device': 'cuda:0'}
    pcd_dir = "lidar_bin"
    
    

    inferencer = LidarDet3DInferencer(**init_args)

    # 新增代码：迭代出 pcd_dir 下的所有文件并加上路径
    for root, dirs, files in os.walk(pcd_dir):
        for file in files:
            file_path = os.path.join(root, file)
            # 处理每个文件
            call_args['inputs'] = dict(points=file_path)
            inferencer(**call_args)

    if call_args['out_dir'] != '' and not (call_args['no_save_vis']
                                           and call_args['no_save_pred']):
        print_log(
            f'results have been saved at {call_args["out_dir"]}',
            logger='current')


if __name__ == '__main__':
    main()
