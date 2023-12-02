import pickle
import numpy as np
import mmcv
import cv2
import sys
import pdb
import os
from templates.existing import get_existence_text
from templates.counting import get_counting_text
from templates.topology import get_traffic_topology_text
from utils.scene_graph import get_scene_graph
from utils.vis import show_image, show_image_anno, show_image_raw
from tqdm import tqdm
import json 
from json import JSONEncoder
import math
import copy
from main_test import generate_question
from pathlib import Path


from itertools import islice

data_path = '/mnt/disk01/nuscenes/'
language_path = 'language'

data_info_all_train = pickle.load(open(os.path.join(data_path, "nuscenes_infos_temporal_val_xjw.pkl"), "rb"))
topo_info_all_train = pickle.load(open(os.path.join(data_path, "openlane_v2_nus/data_dict_subset_B_val_new.pkl"), "rb"))

inverse_map_ts_index = {}

for i in tqdm(range(len(data_info_all_train)), desc ="build success"):
    data_info = data_info_all_train[i]
    timestamp = data_info["cams"]["CAM_FRONT"]["data_path"].split("__")[-1][:-4]
    scene_token = data_info['scene_token']

    inverse_map_ts_index[timestamp] = (i, scene_token)

mmcv.dump(inverse_map_ts_index, os.path.join(data_path, language_path, "inverse_map_ts_idx_st_val.pkl"))




