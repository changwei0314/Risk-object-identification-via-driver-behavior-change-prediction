import os.path as osp
import os
from collections import OrderedDict
import socket
import getpass
import json
import PIL.Image as Image
import numpy as np

machine_name = socket.gethostname()
username = getpass.getuser()

__all__ = ['parse_args']


def parse_args(parser):
    parser.add_argument(
        '--data_root', default=osp.expanduser('/home/william/HDD'), type=str)
    parser.add_argument(
        '--save_path', default=osp.expanduser('/home/william/exp_trn'), type=str)
    parser.add_argument('--phases', default=['train', 'test'], type=list)
    parser.add_argument('--test_interval', default=1, type=int)
    parser.add_argument('--width', default=1280, type=int)
    parser.add_argument('--height', default=720, type=int)

    args = parser.parse_args()
    args.data = osp.basename(osp.normpath(args.data_root))


    args.data_root = '/data/william/interactive'
    args.save_path = '/home/william/risk assessment via GAT/save'

    args.class_index = list(data_info[args.data]['class_info'].keys())
    args.class_weight = list(data_info[args.data]['class_info'].values())

    args.test_session_set = []
    args.train_session_set = []

    
    # open scenario_list.json to choose selected scenario
    f = open('/home/william/risk-assessment-via-GAT/config/scenario_list.json')
    scenario_list = json.load(f)
    interactive_list = scenario_list["interactive"][0]

    for basic_scene in os.listdir(args.data_root):
        basic_scene_path = osp.join(args.data_root, basic_scene, 'variant_scenario')

        for var_scene in os.listdir(basic_scene_path):
            # check if the current scene is in the scenario_list.json
            scene_info = [basic_scene, var_scene]
            if scene_info not in interactive_list:
                continue

            var_scene_path = osp.join(basic_scene_path, var_scene)
            
            if basic_scene[:2] == '10':
                args.test_session_set.append(var_scene_path)
            else:
                args.train_session_set.append(var_scene_path)




    
    args.num_classes = len(args.class_index)

    return args


data_info = OrderedDict()
data_info['HDD'] = OrderedDict()
'''
data_info['HDD']['class_info'] = OrderedDict([
    ('background',               1.0),
    ('intersection passing',     1.0),
    ('left turn',                1.0),
    ('right turn',               1.0),
    ('left lane change',         1.0),
    ('right lane change',        1.0),
    ('left lane branch',         1.0),
    ('right lane branch',        1.0),
    ('crosswalk passing',        1.0),
    ('railroad passing',         1.0),
    ('merge',                    1.0),
    ('U-turn',                   1.0),
])
'''
data_info['HDD']['class_info'] = OrderedDict([
    ('go',       1.0),
    ('stop',     1.0),
])

