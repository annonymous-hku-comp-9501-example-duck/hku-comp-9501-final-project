import json
import random
import os
from tqdm import tqdm
from pathlib import Path
import pickle as pkl
from PIL import Image, ImageDraw

import torch
import cv2
import numpy as np
from skimage.transform import resize

from finetuning_vlm.tokenizer import ActionTokenizer

from hydra.utils import get_original_cwd, to_absolute_path
from transformers import (
    AutoProcessor,
)
# from line_profiler import LineProfiler

# def profile(func):
#     def wrapper(*args, **kwargs):
#         profiler = LineProfiler()
#         profiler.add_function(func)
#         profiler.runcall(func, *args, **kwargs)
#         profiler.print_stats()
#     return wrapper


cv2.setNumThreads(0)

# TODO: options for visual sanity checks
SANITY_CHECKS = {
    "VIZ_GETITEM": False,
}


class DriveLM(torch.utils.data.Dataset):
    # @profile
    def __init__(
        self,
        cfg,
        data_path,
        # sample="question", # "image" or "question"
        training_data="QA", # QA, action, both, action_full
        QA_version="v12", # "v11" or "v0"
        QA_type="all", # "Perception" or "Prediction" or "Planning" or "all"
        QA_reference="description", # 'tags', description, description_answer, description_visual, coordinates, description_visual_tags
        filter="front_view", # front_view, none
        split="train",
        discretize="linear", # 'linear', 'square', 'nonlinear'
        num_bins=256,
        action_command_input=False,
        test=False,
        max_samples=-1,
        baseline='None',
        test_split='None',
        **kwargs,
    ):
        super().__init__()

        self.data_path = to_absolute_path(data_path)
        if training_data.startswith('action'):
            sample = 'image'
        elif training_data.startswith('QA') or training_data == 'both':
            sample = 'question'
        else:
            raise NotImplementedError

        if action_command_input:
            assert training_data == 'action' or training_data == 'action_QA', "action_command_input only works with training_data='action'"

        
        
        self.cfg = cfg
        self.test = test
        self.test_split = test_split
        self.training_data = training_data
        self.action_command_input = action_command_input
        self.filter = filter
        self.baseline = baseline
        self.command_type = self.cfg.dataset.get("command_type", "None")
        model_type_name = self.cfg.model.name.split('/')[-1]


        if self.command_type == 'action_template_9':
            self.get_action_template = self.get_action_template_9
        elif self.command_type == 'action_template_7':
            self.get_action_template = self.get_action_template_7
        elif self.command_type == 'action_template_5':
            self.get_action_template = self.get_action_template_5
        elif self.command_type == 'action_template_3':
            self.get_action_template = self.get_action_template_3
        elif self.command_type == 'action_template_11':
            self.get_action_template = self.get_action_template_11

        
        action_tokenizer = ActionTokenizer(model_type_name, discretize, num_bins, past_traj=self.cfg.dataset.add_past_traj)

        num_wps = 6 # 3sec

        if baseline != 'None':
            training_data = 'action'
            if baseline == 'random':
                self.all_train_actions = []
            elif baseline == 'command_random' or baseline == 'command_mean' or baseline == 'mean_action_template':
                self.all_train_actions = {}
            elif baseline == 'nearest_neighbor':
                self.train_features = pkl.load(open(f'data/DriveLM_v2/img_features/{model_type_name}/train/img_features_command.pkl', "rb"))
                self.val_features = pkl.load(open(f'data/DriveLM_v2/img_features/{model_type_name}/val/img_features.pkl', "rb"))
            elif baseline == 'nearest_neighbor_action_template':
                self.train_features = pkl.load(open(f'data/DriveLM_v2/img_features/{model_type_name}/train/img_features_command_action_template.pkl', "rb"))
                self.val_features = pkl.load(open(f'data/DriveLM_v2/img_features/{model_type_name}/val/img_features.pkl', "rb"))
            else:
                raise NotImplementedError

            QA_pairs = json.load(open(os.path.join(self.data_path, f"QA_pair_{QA_version}_train.json"), "r"))
            if test_split != 'None':
                QA_pairs = json.load(open(os.path.join(self.data_path, f"subset_{test_split}_train.json"), "r"))
            data_info_all = pkl.load(open(os.path.join(self.data_path, f"nuscenes_infos_temporal_train_xjw_w_annos_wo_mm.pkl"), "rb"))
            for data_idx in tqdm(range(len(data_info_all)), desc ="loop success"):
                data_info = data_info_all[data_idx]
                timestamp = data_info["cams"]["CAM_FRONT"]["data_path"].split("__")[-1][:-4]    # this is the key to find QA pair and topo info
                scene_token = data_info['scene_token']                                          # this is to find prev/next frames of current frame
                ego_traj = data_info['annos']['gt_sdc_fut_traj'].squeeze()[:num_wps]                # this is the ego trajectory of current frame
                command = data_info['annos']['command']                                         # this is the command of current frame

                if scene_token not in QA_pairs:
                    continue
                if timestamp in QA_pairs[scene_token]["key_frame"].keys():
                    continue

                if baseline == 'random':
                    self.all_train_actions.append(ego_traj)
                elif baseline == 'command_random' or baseline == 'command_mean':
                    if command == 0:
                        command = "go right"
                    elif command == 1:
                        command = "go left"
                    elif command == 2:
                        command = "go straight"

                    if command not in self.all_train_actions:
                        self.all_train_actions[command] = []
                    self.all_train_actions[command].append(ego_traj)

                elif baseline == 'mean_action_template':
                    action = torch.tensor(ego_traj)
                    # get average distance in x and y direction
                    x_dist_mean = torch.mean(action[1:, 0] - action[:-1, 0])
                    y_dist_mean = torch.mean(action[1:, 1] - action[:-1, 1])

                    action_template, category = self.get_action_template(x_dist_mean, y_dist_mean)
                    category = str(category)
                    if category not in self.all_train_actions:
                        self.all_train_actions[category] = []
                    self.all_train_actions[category].append(ego_traj)

            
            if baseline == 'command_mean' or baseline == 'mean_action_template':
                for command in self.all_train_actions:
                    self.all_train_actions[command] = np.mean(np.stack(self.all_train_actions[command]), axis=0)
        

        all_data = []
        miss_topo = 0
        miss_QA = 0
        num_QAs = 0

        # load json from path {data_path}/QA_pair_train.json
        if QA_version != "v0":
            QA_pairs = json.load(open(os.path.join(self.data_path, f"QA_pair_{QA_version}_{split}.json"), "r"))
        elif QA_version == "v0":
            QA_pairs = json.load(open(os.path.join(self.data_path, f"QA_pair_{split}.json"), "r"))
        else:
            raise NotImplementedError


        if test_split != 'None':
            QA_pairs = json.load(open(os.path.join(self.data_path, f"subset_{test_split}_train.json"), "r"))
        
        data_info_all = pkl.load(open(os.path.join(self.data_path, f"nuscenes_infos_temporal_{split}_xjw_w_annos_wo_mm.pkl"), "rb"))
        past_data = pkl.load(open(os.path.join(self.data_path, f"data_nuscene.pkl"), "rb"))
        # topo_info_all = pkl.load(open(os.path.join(self.data_path, f"data_dict_subset_B_{split}_new.pkl"), "rb"))
        # plan_info_all = pkl.load(open(os.path.join(self.data_path, f"nuscenes_planning_gt_{split}.pkl"), "rb"))
        # this is used when you loop QA pair json and want to find the corresponding frame in data_info_all
        inverse_map_ts_index = pkl.load(open(os.path.join(self.data_path, f"inverse_map_ts_idx_st_{split}.pkl"), "rb"))

        for data_idx in tqdm(range(len(data_info_all)), desc ="loop success"):
            data_info = data_info_all[data_idx]
            timestamp = data_info["cams"]["CAM_FRONT"]["data_path"].split("__")[-1][:-4]    # this is the key to find QA pair and topo info
            scene_token = data_info['scene_token']                                          # this is to find prev/next frames of current frame

            cams_info = data_info["cams"]                           # this is all info about images of current frame
            cam_front_path = cams_info["CAM_FRONT"]["data_path"]    # relative path, i.e. './data/nuscenes/samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883530412470.jpg'

            ego_traj_past_tokens = None
            ego_traj_past = None
            if self.cfg.dataset.add_past_traj:
                ego_traj_past = past_data[data_info['token']]['x']
                ego_traj_past = np.array(ego_traj_past)[...,:2]
                # reverse the order of past trajectory
                # ego_traj_past = ego_traj_past[::-1].copy()
                ego_traj_past_tokens = action_tokenizer.tokenize(ego_traj_past)
            
            ego_traj_alternative  = data_info['annos']['sdc_planning'].squeeze()[:num_wps, :2]    # this is the alternative ego trajectory of current frame
            ego_traj = data_info['annos']['gt_sdc_fut_traj'].squeeze()[:num_wps]                # this is the ego trajectory of current frame
            ego_traj_mask = data_info['annos']['gt_sdc_fut_traj_mask'].squeeze()[:num_wps]      # this is the mask of ego trajectory of current frame
            command = data_info['annos']['command']                                         # this is the command of current frame
            if command == 0:
                command = "go right"
            elif command == 1:
                command = "go left"
            elif command == 2:
                command = "go straight"

            # tokenize ego traj
            ego_traj_tokens = action_tokenizer.tokenize(ego_traj)

            # if timestamp in topo_info_all.keys():
            #     topo_info = topo_info_all[timestamp]          # this have traffic elements 2d bb in cam front, centerlines, and the relationship between them
            # else:
            #     # print(f"timestamp not found in topo: {timestamp}")
            #     miss_topo += 1

            # if timestamp in plan_info_all.keys():
            #     plan_info = plan_info_all[timestamp]


            # TODO make this nicer. So many for loops.
            if training_data == 'QA' or training_data =='both' or training_data =='action_QA' or (training_data == 'action' and action_command_input) or (training_data.startswith('action') and test):
                if scene_token not in QA_pairs:
                    continue
                if timestamp in QA_pairs[scene_token]["key_frame"].keys():
                    QA_pair_info = QA_pairs[scene_token]["key_frame"][timestamp]                 # this is the QA pair of current frame.
                    if QA_version != "v0":
                        if QA_type != "all":
                            QA_pair_info_filtered = QA_pair_info[QA_type] # dict: {q:[], a:[]}
                            # get dict: {q: a}
                            QA_pair_info_dict = {QA_pair_info_filtered["q"][i]: QA_pair_info_filtered["a"][i] for i in range(len(QA_pair_info_filtered["q"]))}
                        elif QA_type == "all":
                            if QA_reference == "description_visual" or QA_reference == "description_visual_tags":
                                mapping = QA_pair_info['Perception']['description_visual']
                            else:
                                mapping = QA_pair_info['Perception']['description']
                            QA_pair_info_dict = {}
                            for qa_type_tmp in ['Perception', 'Prediction and Planning']:
                                QA_pair_info_filtered = QA_pair_info[qa_type_tmp]

                                if "description" in QA_reference:
                                    for ix, (QA_pair_single_q, QA_pair_single_a) in enumerate(zip(QA_pair_info_filtered["q"], QA_pair_info_filtered["a"])):
                                        for map_item in mapping:
                                            to_replace_str_1 = f"object {map_item}"
                                            to_replace_str_2 = f"Object {map_item}" # hacky hack
                                            to_replace_str_3 = map_item
                                            replace_with_str_1 = mapping[map_item].replace(f"{map_item} is a", "").replace("<", "").replace(">", "").replace("[", "").replace("]", "").replace(".", "")
                                            replace_with_str_1 = replace_with_str_1.replace("A ", "")
                                            replace_with_str_1 = replace_with_str_1.strip() #removing leading spaces
                                            replace_with_str_2 = f"The {replace_with_str_1}"
                                            replace_with_str_1 = f"the {replace_with_str_1}"
                                            if QA_reference == 'description_visual_tags':
                                                replace_with_str_1 = f"{map_item} ({replace_with_str_1})"
                                                replace_with_str_2 = f"{map_item} ({replace_with_str_2})"
                                            if QA_reference != "description_answer":
                                                if to_replace_str_1 in QA_pair_single_q:
                                                    QA_pair_info_filtered["q"][ix] = QA_pair_info_filtered["q"][ix].replace(to_replace_str_1, replace_with_str_1)
                                                elif to_replace_str_2 in QA_pair_single_q:
                                                    QA_pair_info_filtered["q"][ix] = QA_pair_info_filtered["q"][ix].replace(to_replace_str_2, replace_with_str_2)
                                                elif to_replace_str_3 in QA_pair_single_q:
                                                    QA_pair_info_filtered["q"][ix] = QA_pair_info_filtered["q"][ix].replace(to_replace_str_3, replace_with_str_1)
                                            if to_replace_str_1 in QA_pair_single_a:
                                                QA_pair_info_filtered["a"][ix] = QA_pair_info_filtered["a"][ix].replace(to_replace_str_1, replace_with_str_1)
                                            elif to_replace_str_2 in QA_pair_single_a:
                                                QA_pair_info_filtered["a"][ix] = QA_pair_info_filtered["a"][ix].replace(to_replace_str_2, replace_with_str_2)
                                            elif to_replace_str_3 in QA_pair_single_a:
                                                QA_pair_info_filtered["a"][ix] = QA_pair_info_filtered["a"][ix].replace(to_replace_str_3, replace_with_str_1)


                                QA_pair_info_dict.update({QA_pair_info_filtered["q"][i]: QA_pair_info_filtered["a"][i] for i in range(len(QA_pair_info_filtered["q"]))})

                                    
                    
                    elif QA_version == "v0":
                        QA_pair_info_dict = QA_pair_info

                    if training_data == 'QA' or training_data =='both':
                        for QA_pair_key in QA_pair_info_dict:
                            if QA_pair_key == "Path":
                                continue

                            if filter == "front_view":
                                if 'back' in QA_pair_key or 'back' in QA_pair_info_dict[QA_pair_key]:
                                    continue
                                if 'front left' in QA_pair_key or 'front left' in QA_pair_info_dict[QA_pair_key]:
                                    continue
                                if 'front right' in QA_pair_key or 'front right' in QA_pair_info_dict[QA_pair_key]:
                                    continue
                                if 'CAM_FRONT_LEFT' in QA_pair_key or 'CAM_FRONT_LEFT' in QA_pair_info_dict[QA_pair_key]:
                                    continue
                                if 'CAM_FRONT_RIGHT' in QA_pair_key or 'CAM_FRONT_RIGHT' in QA_pair_info_dict[QA_pair_key]:
                                    continue
                                if 'BACK' in QA_pair_key or 'BACK' in QA_pair_info_dict[QA_pair_key]:
                                    continue

                            
                            QA_pair = {QA_pair_key: QA_pair_info_dict[QA_pair_key]}
                            all_data.append({
                                "timestamp": timestamp,
                                "scene_token": scene_token,
                                "cams_info": cams_info,
                                "ego_traj": ego_traj,
                                "ego_traj_alternative": ego_traj_alternative,
                                "ego_traj_mask": ego_traj_mask,
                                "ego_traj_tokens": ego_traj_tokens,
                                "ego_traj_past": ego_traj_past,
                                "ego_traj_past_tokens": ego_traj_past_tokens,
                                "command": command,
                                # "topo_info": topo_info,
                                # "plan_info": plan_info,
                                "QA_pair_info": QA_pair,
                                "data_idx": data_idx,
                            })

                else:
                    QA_pair_info_dict = {}
                        

            if training_data.startswith('action'):
                if (training_data == 'action' or training_data == 'action_QA' or split == 'val') and scene_token not in QA_pairs:
                    continue
                elif (training_data == 'action' or training_data == 'action_QA' or split == 'val') and timestamp not in QA_pairs[scene_token]["key_frame"].keys():
                    continue
                if self.action_command_input == False and not (training_data == 'action_QA' or (training_data.startswith('action') and test)):
                    QA_pair_info_dict = {f'Predict the action of the ego vehicle for the command: {command}': 'action'}
                all_data.append({
                    "timestamp": timestamp,
                    "scene_token": scene_token,
                    "cams_info": cams_info,
                    "ego_traj_alternative": ego_traj_alternative,
                    "ego_traj": ego_traj,
                    "ego_traj_mask": ego_traj_mask,
                    "ego_traj_tokens": ego_traj_tokens,
                    "ego_traj_past": ego_traj_past,
                    "ego_traj_past_tokens": ego_traj_past_tokens,
                    "command": command,
                    # "topo_info": topo_info,
                    # "plan_info": plan_info,
                    "QA_pair_info": QA_pair_info_dict,
                    "data_idx": data_idx,
                })
                # try:
                #     print(QA_pair_info_dict['Q: What is the goal action of the ego vehicle?'])
                # except:
                #     pass

            num_QAs += len(QA_pair_info_dict)#-1 # -1 because there is one entry for the path

            # else:
            #     # print(f"timestamp not found in QA pair: {timestamp}")
            #     miss_QA += 1


        self.num_QAs = num_QAs
        self.all_data = all_data


        # if test == True:
        #     random.shuffle(self.all_data)
        #     self.all_data = self.all_data[:100]

        if max_samples != -1:
            self.all_data = self.all_data[:max_samples]

        # if split == "val":
        #     #shuffle the data
        #     random.shuffle(self.all_data)
        #     self.all_data = self.all_data[:200]
        
        # print(f"miss topo: {miss_topo}, miss QA: {miss_QA}")
        print(f"num_QAs: {num_QAs}")
        print(f"len(all_data): {len(all_data)}")


    def __len__(self):
        if self.training_data == 'both' or self.training_data == 'action_QA':
            return len(self.all_data) * 2
        
        return len(self.all_data)
    
    
    def get_action_template_11(self, x_dist_mean, y_dist_mean):
        # get template sentences describing the action x is steering and y is speed/ acceleration
        category = []
        if x_dist_mean > 3:
            action_template = "The ego vehicle is making a sharp right turn."
            category.append(0)
        elif x_dist_mean > 2:       
            action_template = "The ego vehicle is steering very strong to the right." 
            category.append(1)
        elif x_dist_mean > 1:
            action_template = "The ego vehicle is steering to the right."
            category.append(2)
        elif x_dist_mean > 0.5:
            action_template = "The ego vehicle is slightly steering to the right."
            category.append(3)
        elif x_dist_mean > 0.2:
            action_template = "The ego vehicle is correcting its course a bit to the right."
            category.append(4)
        elif x_dist_mean > -0.2:
            action_template = "The ego vehicle is going straight."
            category.append(5)
        elif x_dist_mean > -0.5:
            action_template = "The ego vehicle is correcting its course a bit to the left."
            category.append(6)
        elif x_dist_mean > -1:
            action_template = "The ego vehicle is slightly steering to the left."
            category.append(7)
        elif x_dist_mean > -2:
            action_template = "The ego vehicle is steering to the left." 
            category.append(8) 
        elif x_dist_mean > -3:
            action_template = "The ego vehicle is steering very strong to the left." 
            category.append(9)
        else:
            action_template = "The ego vehicle is making a sharp left turn."
            category.append(10)

        if y_dist_mean > 5:
            action_template += " The ego vehicle is driving super fast." 
            category.append(0)
        elif y_dist_mean > 4:         
            action_template += " The ego vehicle is driving very fast." 
            category.append(1)
        elif y_dist_mean > 3:
            action_template += " The ego vehicle is driving fast."
            category.append(2)
        elif y_dist_mean > 2:
            action_template += " The ego vehicle is driving moderately fast."
            category.append(3)
        elif y_dist_mean > 1.5:
            action_template += " The ego vehicle is driving with normal speed."
            category.append(4)
        elif y_dist_mean > 1:
            action_template += " The ego vehicle is driving moderately slow."
            category.append(5)
        elif y_dist_mean > 0.5:
            action_template += " The ego vehicle is driving slowly."
            category.append(6)
        elif y_dist_mean > 0.2:       
            action_template += " The ego vehicle is driving very slowly."  
            category.append(7)
        elif y_dist_mean > -0.2:
            action_template += " The ego vehicle is not moving."
            category.append(8)
        elif y_dist_mean > -0.5:
            action_template += " The ego vehicle is slightly driving backwards."
            category.append(9)
        else:
            action_template += " The ego vehicle is driving backwards."
            category.append(10)

        return action_template, category
    
    def get_action_template_9(self, x_dist_mean, y_dist_mean):
        # action = torch.tensor(self.all_data[index]["ego_traj"])
        # # get average distance in x and y direction
        # x_dist_mean = torch.mean(action[1:, 0] - action[:-1, 0])
        # y_dist_mean = torch.mean(action[1:, 1] - action[:-1, 1])

        # get template sentences describing the action x is steering and y is speed/ acceleration
        category = []
        if x_dist_mean > 2:
            action_template = "The ego vehicle is steering very strong to the right."
            category.append(0)
        elif x_dist_mean > 1:
            action_template = "The ego vehicle is steering to the right."
            category.append(1)
        elif x_dist_mean > 0.5:
            action_template = "The ego vehicle is slightly steering to the right."
            category.append(2)
        elif x_dist_mean > 0.2:
            action_template = "The ego vehicle is correcting its course a bit to the right."
            category.append(3)
        elif x_dist_mean > -0.2:
            action_template = "The ego vehicle is going straight."
            category.append(4)
        elif x_dist_mean > -0.5:
            action_template = "The ego vehicle is correcting its course a bit to the left."
            category.append(5)
        elif x_dist_mean > -1:
            action_template = "The ego vehicle is slightly steering to the left."
            category.append(6)
        elif x_dist_mean > -2:
            action_template = "The ego vehicle is steering to the left."
            category.append(7)
        else:
            action_template = "The ego vehicle is steering very strong to the left."
            category.append(8)

        if y_dist_mean > 4:
            action_template += " The ego vehicle is driving very fast."
            category.append(0)
        elif y_dist_mean > 3:
            action_template += " The ego vehicle is driving fast."
            category.append(1)
        elif y_dist_mean > 2:
            action_template += " The ego vehicle is driving moderately fast."
            category.append(2)
        elif y_dist_mean > 1.5:
            action_template += " The ego vehicle is driving with normal speed."
            category.append(3)
        elif y_dist_mean > 1:
            action_template += " The ego vehicle is driving moderately slow."
            category.append(4)
        elif y_dist_mean > 0.5:
            action_template += " The ego vehicle is driving slowly."
            category.append(5)
        elif y_dist_mean > 0.2:
            action_template += " The ego vehicle is driving very slowly."
            category.append(6)
        elif y_dist_mean > -0.2:
            action_template += " The ego vehicle is not moving."
            category.append(7)
        else:
            action_template += " The ego vehicle is driving backwards."
            category.append(8)

        return action_template, category
    
    def get_action_template_7(self, x_dist_mean, y_dist_mean):

        # get template sentences describing the action x is steering and y is speed/ acceleration
        category = []
        if x_dist_mean > 1:
            action_template = "The ego vehicle is steering to the right."
            category.append(0)
        elif x_dist_mean > 0.5:
            action_template = "The ego vehicle is slightly steering to the right."
            category.append(1)
        elif x_dist_mean > 0.2:
            action_template = "The ego vehicle is correcting its course a bit to the right."
            category.append(2)
        elif x_dist_mean > -0.2:
            action_template = "The ego vehicle is going straight."
            category.append(3)
        elif x_dist_mean > -0.5:
            action_template = "The ego vehicle is correcting its course a bit to the left."
            category.append(4)
        elif x_dist_mean > -1:
            action_template = "The ego vehicle is slightly steering to the left."
            category.append(5)
        else:
            action_template = "The ego vehicle is steering to the left."
            category.append(6)

        if y_dist_mean > 3:
            action_template += " The ego vehicle is driving fast."
            category.append(0)
        elif y_dist_mean > 2:
            action_template += " The ego vehicle is driving moderately fast."
            category.append(1)
        elif y_dist_mean > 1.5:
            action_template += " The ego vehicle is driving with normal speed."
            category.append(2)
        elif y_dist_mean > 1:
            action_template += " The ego vehicle is driving moderately slow."
            category.append(3)
        elif y_dist_mean > 0.5:
            action_template += " The ego vehicle is driving slowly."
            category.append(4)
        elif y_dist_mean > -0.2:
            action_template += " The ego vehicle is not moving."
            category.append(5)
        else:
            action_template += " The ego vehicle is driving backwards."
            category.append(6)

        return action_template, category
    

    def get_action_template_5(self, x_dist_mean, y_dist_mean):

        # get template sentences describing the action x is steering and y is speed/ acceleration
        category = []
        if x_dist_mean > 1:
            action_template = "The ego vehicle is steering to the right."
            category.append(0)
        elif x_dist_mean > 0.2:
            action_template = "The ego vehicle is slightly steering to the right."
            category.append(1)
        elif x_dist_mean > -0.2:
            action_template = "The ego vehicle is going straight."
            category.append(3)
        elif x_dist_mean > -1:
            action_template = "The ego vehicle is slightly steering to the left."
            category.append(5)
        else:
            action_template = "The ego vehicle is steering to the left."
            category.append(6)

        if y_dist_mean > 3:
            action_template += " The ego vehicle is driving fast."
            category.append(0)
        elif y_dist_mean > 1.5:
            action_template += " The ego vehicle is driving with normal speed."
            category.append(2)
        elif y_dist_mean > 0.5:
            action_template += " The ego vehicle is driving slowly."
            category.append(4)
        elif y_dist_mean > -0.2:
            action_template += " The ego vehicle is not moving."
            category.append(5)
        else:
            action_template += " The ego vehicle is driving backwards."
            category.append(6)

        return action_template, category
    
    def get_action_template_3(self, x_dist_mean, y_dist_mean):
        # action = torch.tensor(self.all_data[index]["ego_traj"])
        # # get average distance in x and y direction
        # x_dist_mean = torch.mean(action[1:, 0] - action[:-1, 0])
        # y_dist_mean = torch.mean(action[1:, 1] - action[:-1, 1])

        # get template sentences describing the action x is steering and y is speed/ acceleration
        category = []
        if x_dist_mean > 0.2:
            action_template = "The ego vehicle is steering to the right."
            category.append(0)
        elif x_dist_mean > -0.2:
            action_template = "The ego vehicle is going straight."
            category.append(1)
        else:
            action_template = "The ego vehicle is steering to the left."
            category.append(2)

        if y_dist_mean > 0.5:
            action_template += " The ego vehicle is driving forward."
            category.append(0)
        elif y_dist_mean > -0.1:
            action_template += " The ego vehicle is not moving."
            category.append(1)
        else:
            action_template += " The ego vehicle is driving backwards."
            category.append(2)

        return action_template, category



    # @profile
    def __getitem__(self, index):
                


        if self.training_data == 'both' or self.training_data == 'action_QA':
            # choose randomly between QA and action
            training_type = random.choice(['QA', 'action'])
            index = index // 2
        elif self.training_data == 'action_full':
            training_type = 'action'
        else:
            training_type = self.training_data


        action = torch.tensor(self.all_data[index]["ego_traj"])
        action_mask = torch.tensor(self.all_data[index]["ego_traj_mask"])
        #remove entries from action where action_mask is 0
        new_action = []
        for i in range(len(action_mask)):
            if action_mask[i][0] == 0:
                continue
            new_action.append(action[i])

        new_action = torch.stack(new_action)
    

        # get average distance in x and y direction
        x_dist_mean = torch.mean(new_action[1:, 0] - new_action[:-1, 0])
        y_dist_mean = torch.mean(new_action[1:, 1] - new_action[:-1, 1])

        action_template, category = self.get_action_template(x_dist_mean, y_dist_mean)
        
        ego_traj_past_tokens = None
        baseline_action = None
        if self.baseline == 'random':
            baseline_action = torch.tensor(random.choice(self.all_train_actions))
        elif self.baseline == 'command_random':
            baseline_action = torch.tensor(random.choice(self.all_train_actions[self.all_data[index]['command']]))
        elif self.baseline == 'command_mean':
            baseline_action = torch.tensor(self.all_train_actions[self.all_data[index]['command']])
        elif self.baseline.startswith('nearest_neighbor'):
            val_feature = self.val_features[self.all_data[index]['scene_token']][self.all_data[index]['timestamp']]['features']
            # find nearest neighbor in train with cosine similarity
            if self.baseline == 'nearest_neighbor':
                train_features = self.train_features[self.all_data[index]['command']]['features']
            elif self.baseline == 'nearest_neighbor_action_template':
                if str(category) in self.train_features:
                    train_features = self.train_features[str(category)]['features']
                else:
                    category[0] += 1
                    if str(category) in self.train_features:
                        train_features = self.train_features[str(category)]['features']
                    else:
                        category[0] -= 1
                        category[0] -= 1
                        train_features = self.train_features[str(category)]['features']


            # try: 
            #     length = len(train_features)
            # except:
            #     breakpoint()

            train_features = np.array(train_features)
            val_feature = np.array(val_feature)[0]
            similarities = []
            for i in range(len(train_features)):
                similarity = np.dot(train_features[i], val_feature) / (np.linalg.norm(train_features[i]) * np.linalg.norm(val_feature))
                similarities.append(similarity)
            max_indice = np.argmax(similarities)
            if self.baseline == 'nearest_neighbor':
                baseline_action = torch.tensor(self.train_features[self.all_data[index]['command']]['actions'][max_indice])
            elif self.baseline == 'nearest_neighbor_action_template':
                baseline_action = torch.tensor(self.train_features[str(category)]['actions'][max_indice])

        elif self.baseline == 'mean_action_template':
            category = str(category)
            baseline_action = torch.tensor(self.all_train_actions[category])


        front_image_path = self.all_data[index]["cams_info"]["CAM_FRONT"]["data_path"]
        nuscenes_path = Path(self.data_path).parent / "nuscenes"
        if '/mnt/disk01' in front_image_path:
            front_image_path = front_image_path.replace("/mnt/disk01/nuscenes/Nuscenes", str(nuscenes_path))
        else:
            # path format: './data/nuscenes/samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883530412470.jpg'
            front_image_path = front_image_path.replace("./data/nuscenes", "")
            front_image_path = nuscenes_path / Path(front_image_path)
            # front_image_path = str(nuscenes_path) + front_image_path
        # check if the image path is valid
        if not os.path.exists(front_image_path):
            print(f"image path not found: {front_image_path}")
            return None

        front_image = Image.open(front_image_path)
        if self.test:
            front_image_org = np.asarray(front_image) # * 0.0
        else:
            front_image_org = None
        # resize image
        front_image = front_image.resize((224, 224))
        # front_image2 = resize(np.array(front_image), (224, 224))
        # front_image = cv2.resize(np.array(front_image), (224, 224))
        front_image = np.asarray(front_image) # * 0.0
        # print(front_image.shape)
        # front_image = cv2.imread(front_image_path)
        # front_image = cv2.resize(front_image, (224, 224))
        # dummy image
        # front_image = np.zeros((480, 840, 3), dtype=np.uint8)

        # front_image = cv2.cvtColor(front_image, cv2.COLOR_BGR2RGB)
        # chnage data type to float16
        # front_image = front_image.astype(np.float16)

        # sample one entry from the dict: self.all_data[index]["QA_pair_info"] and get key for question and value for answer
        sample_QA_pair = random.choice(list(self.all_data[index]["QA_pair_info"].items()))
        question = sample_QA_pair[0] + ' A: '
        answer = sample_QA_pair[1][3:]
        question_org = question
        answer_org = answer

        if self.training_data == 'action_QA':
            if self.filter == "front_view":
                while True:
                    sample_QA_pair = random.choice(list(self.all_data[index]["QA_pair_info"].items()))
                    question = sample_QA_pair[0] + ' A: '
                    answer = sample_QA_pair[1][3:]

                    if 'back' in question or 'back' in answer:
                        continue
                    if 'front left' in question or 'front left' in answer:
                        continue
                    if 'front right' in question or 'front right' in answer:
                        continue
                    if 'CAM_FRONT_LEFT' in question or 'CAM_FRONT_LEFT' in answer:
                        continue
                    if 'CAM_FRONT_RIGHT' in question or 'CAM_FRONT_RIGHT' in answer:
                        continue
                    if 'BACK' in question or 'BACK' in answer:
                        continue
                    break
        
        question_filtered = question
        answer_filtered = answer

        answer_ego = 'None'
        import re
        for qs in self.all_data[index]["QA_pair_info"].keys():
            if qs.startswith("Q: What object should the ego vehicle notice first"):
                answer_ego = self.all_data[index]["QA_pair_info"][qs]
                # remove all text inside < >
                answer_ego = re.sub('<.*?>', '', answer_ego)

                answer_ego = answer_ego.split(".")[0]
                answer_ego = answer_ego.split(",")[-1]





        # remove > < [ ] from question
        if training_type == 'QA':
            question = question.replace("<", "").replace(">", "").replace("[", "").replace("]", "")
            if self.training_data == 'action_QA':
                question = "Q: Predict the action command of the ego vehicle. A: "
                answer = action_template
        elif training_type == 'action':
            if self.action_command_input:

                # goal_ego = self.all_data[index]["QA_pair_info"]['Q: What is the goal action of the ego vehicle?']
                if 'Q: In this scenario, what are safe actions to take for the ego vehicle?' in self.all_data[index]["QA_pair_info"]:
                    safe_action = self.all_data[index]["QA_pair_info"]['Q: In this scenario, what are safe actions to take for the ego vehicle?'].replace('A: ', '')
                elif 'Q: In this situation, what are the appropriate actions for the ego vehicle to ensure safety?' in self.all_data[index]["QA_pair_info"]:
                    safe_action = self.all_data[index]["QA_pair_info"]['Q: In this situation, what are the appropriate actions for the ego vehicle to ensure safety?'].replace('A: ', '')
                elif 'Q: In this situation, what are the recommended actions for the ego vehicle to ensure safety?' in self.all_data[index]["QA_pair_info"]:
                    safe_action = self.all_data[index]["QA_pair_info"]['Q: In this situation, what are the recommended actions for the ego vehicle to ensure safety?'].replace('A: ', '')
                elif 'Q: What are the recommended actions for the ego vehicle in this situation?' in self.all_data[index]["QA_pair_info"]:
                    safe_action = self.all_data[index]["QA_pair_info"]['Q: What are the recommended actions for the ego vehicle in this situation?'].replace('A: ', '')
                elif 'Q: In this scenario, what are the safe actions for the ego vehicle to take?' in self.all_data[index]["QA_pair_info"]:
                    safe_action = self.all_data[index]["QA_pair_info"]['Q: In this scenario, what are the safe actions for the ego vehicle to take?'].replace('A: ', '')
                elif 'Q: In this situation, what actions should the ego vehicle take to ensure safety?' in self.all_data[index]["QA_pair_info"]:
                    safe_action = self.all_data[index]["QA_pair_info"]['Q: In this situation, what actions should the ego vehicle take to ensure safety?'].replace('A: ', '')
                elif 'Q: In this scenario, what are the recommended actions for the ego vehicle to ensure safety?' in self.all_data[index]["QA_pair_info"]:
                    safe_action = self.all_data[index]["QA_pair_info"]['Q: In this scenario, what are the recommended actions for the ego vehicle to ensure safety?'].replace('A: ', '')
                elif 'Q: What are the appropriate actions for the ego vehicle to ensure safety in this situation?' in self.all_data[index]["QA_pair_info"]:
                    safe_action = self.all_data[index]["QA_pair_info"]['Q: What are the appropriate actions for the ego vehicle to ensure safety in this situation?'].replace('A: ', '')
                elif 'Q: In this scenario, what are the safe actions to take for the ego vehicle?' in self.all_data[index]["QA_pair_info"]:
                    safe_action = self.all_data[index]["QA_pair_info"]['Q: In this scenario, what are the safe actions to take for the ego vehicle?'].replace('A: ', '')
                elif 'Q: What are the recommended actions for the ego vehicle in this situation to ensure safety?' in self.all_data[index]["QA_pair_info"]:
                    safe_action = self.all_data[index]["QA_pair_info"]['Q: What are the recommended actions for the ego vehicle in this situation to ensure safety?'].replace('A: ', '')
                elif 'Q: In this scenario, what are the safe actions that the ego vehicle should take?' in self.all_data[index]["QA_pair_info"]:
                    safe_action = self.all_data[index]["QA_pair_info"]['Q: In this scenario, what are the safe actions that the ego vehicle should take?'].replace('A: ', '')
                elif 'Q: In this scenario, what are the safe actions the ego vehicle should take?' in self.all_data[index]["QA_pair_info"]:
                    safe_action = self.all_data[index]["QA_pair_info"]['Q: In this scenario, what are the safe actions the ego vehicle should take?'].replace('A: ', '')
                elif 'Q: In this scenario, what are safe actions for the ego vehicle to take?' in self.all_data[index]["QA_pair_info"]:
                    safe_action = self.all_data[index]["QA_pair_info"]['Q: In this scenario, what are safe actions for the ego vehicle to take?'].replace('A: ', '')
                elif 'Q: In this situation, what are the safe actions that the ego vehicle should take?' in self.all_data[index]["QA_pair_info"]:
                    safe_action = self.all_data[index]["QA_pair_info"]['Q: In this situation, what are the safe actions that the ego vehicle should take?'].replace('A: ', '')
                else:
                    # breakpoint()
                    print(self.all_data[index]['scene_token'])
                    print(self.all_data[index]['timestamp'])
                    safe_action = 'None'
                # question = f"Predict the action of the ego vehicle for the command: {goal_ego}. Safe actions are: {safe_action}"
                if self.command_type == 'safe_action':
                    question = f"Predict the action. Safe actions are: {safe_action}"
                elif self.command_type == 'answer_ego':
                    question = f"Predict the action. {answer_ego}"
                elif self.command_type.startswith('action_template'):
                    question = f"{action_template}"
            else:
                question = f"Predict the action of the ego vehicle for the command: {self.all_data[index]['command']}."

        # print(f"question: {question}")
        answer = answer.replace("<", "").replace(">", "").replace("[", "").replace("]", "")

        action_tokens = {
            "input_ids": torch.tensor(self.all_data[index]["ego_traj_tokens"]).unsqueeze(0),
            "attention_mask": torch.ones(len(self.all_data[index]["ego_traj_tokens"])).unsqueeze(0),
            }
        actions = torch.tensor(self.all_data[index]["ego_traj"])
        actions_alternative = torch.tensor(self.all_data[index]["ego_traj_alternative"])
        if self.cfg.dataset.add_past_traj:
            ego_traj_past_tokens = torch.tensor(self.all_data[index]["ego_traj_past_tokens"])
        if training_type == 'action':
            actions_mask = torch.tensor(self.all_data[index]["ego_traj_mask"])
        else:
        #     action_tokens = None
        #     actions = torch.tensor(self.all_data[index]["ego_traj"])
        #     actions_alternative = torch.tensor(self.all_data[index]["ego_traj_alternative"])
        #     # actions = torch.zeros_like(torch.tensor(self.all_data[index]["ego_traj"]))
        #     # actions_alternative = torch.zeros_like(torch.tensor(self.all_data[index]["ego_traj_alternative"]))
            actions_mask = torch.zeros_like(torch.tensor(self.all_data[index]["ego_traj_mask"]))
        
        # print(training_type)
        if SANITY_CHECKS["VIZ_GETITEM"]:
            
            save_path = f"{self.data_path}/viz_split_{self.test_split}"
            Path(save_path).mkdir(parents=True, exist_ok=True)
            # to PIL image
            img = front_image_org
            img = img.astype(np.uint8)
            img2 = img.copy()

            img = Image.fromarray(img)
            # text = "Why should the driver of the own vehicle brake?"

            black_box = np.zeros((550, img2.shape[1], 3), dtype=np.uint8)
            # add text to black box
            cv2.putText(black_box, "Question org: " + question_filtered, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(black_box, "Answer org: " + answer_filtered, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(black_box, "Ego action: " + answer_ego, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(black_box, "Action template: " + action_template, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(black_box, "Scene: " + self.all_data[index]['scene_token'], (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(black_box, "Timestep: " + self.all_data[index]['timestamp'], (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # draw waypoints
            # PIL image
            black_box_pil = Image.fromarray(black_box)
            d = ImageDraw.Draw(black_box_pil)
            pixel_per_meter = 15

            for i in range(actions.shape[0]):
                wp_gt = actions[i]
                wp_gt = wp_gt.cpu().detach().numpy()

                wp_y_pixel = 550 - 20 - int(
                    wp_gt[1] * pixel_per_meter
                )
                wp_x_pixel = 500 // 2 + int(
                    wp_gt[0] * pixel_per_meter
                )

                wp_alt = actions_alternative[i]
                wp_alt = wp_alt.cpu().detach().numpy()

                wp_alt_y_pixel = 550 - 20 - int(
                    wp_alt[1] * pixel_per_meter
                )
                wp_alt_x_pixel = 500 // 2 + int(
                    wp_alt[0] * pixel_per_meter
                )
                
                # draw wps as points
                d.ellipse(
                    (
                        wp_x_pixel - 2,
                        wp_y_pixel - 2,
                        wp_x_pixel + 2,
                        wp_y_pixel + 2,
                    ),
                    fill="white",
                )
                d.ellipse(
                    (
                        wp_alt_x_pixel - 1,
                        wp_alt_y_pixel - 1,
                        wp_alt_x_pixel + 1,
                        wp_alt_y_pixel + 1,
                    ),
                    fill="red",
                )

            black_box = np.array(black_box_pil)
            # save images
            img_all = np.concatenate((img, black_box), axis=0)
            img_all = Image.fromarray(img_all)

            # folder_name = 'img_blip2/content_what_do_driver'
            # Path(folder_name).mkdir(parents=True, exist_ok=True)
            
            img_all.save(f'{save_path}/test_{index}.png')

        if self.test:
        # if self.baseline != 'None':
            return (
                front_image,
                question,
                answer,
                action_tokens,
                front_image_org,
                training_type,
                actions,
                actions_mask,
                baseline_action,
                self.all_data[index]['scene_token'], 
                self.all_data[index]['timestamp'],
                self.all_data[index]['data_idx'],
                ego_traj_past_tokens,
                action_template,
            )


        return (
            front_image,
            question,
            answer,
            action_tokens,
            front_image_org,
            training_type,
            actions,
            actions_mask,
            ego_traj_past_tokens,
        )


if __name__ == "__main__":
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    from transformers import AutoProcessor, Blip2ForConditionalGeneration, BlipForConditionalGeneration, Blip2VisionModel
    import pickle

    get_image_features = False
    classes = 'action_template' # command, action_template
    test_discretize = False
    split = 'train'

    initialize(config_path="../config")
    cfg = compose(config_name="config")

    dataset = DriveLM(
        cfg,
        "data/DriveLM_v2",
        QA_version="v3",
        split=split,
        QA_reference="description_visual_tags", # 'tags', description, description_answer, description_visual, coordinates, 
        training_data='action', # QA, action, both, action_full
        action_command_input=True,
        test=True,
        baseline='None',
        test_split='None',
        # sample="image", # "image" or "question"
    )

    for i in range(len(dataset)):
        dataset[i]
        print(i)
        pass

    if get_image_features:
        model_type = 'Salesforce/blip2-flan-t5-xl'
        model = Blip2ForConditionalGeneration.from_pretrained(model_type).cuda() #, device_map={'':current_device}, load_in_8bit=True)
        processor = AutoProcessor.from_pretrained(
            model_type,
            use_fast=False,
        )
        # if split == 'train':
        #     train_features = {'features': [], "actions": []}
        # elif split == 'val':
        train_features = {}
        # for i in range(len(dataset)):
        #     print(i)
        #     front_image, question, answer, action_tokens, front_image_org, training_type, actions, actions_mask = dataset[i]
        #     img = processor(images=front_image, padding="max_length", return_tensors="pt")["pixel_values"]
        #     features = model(img)["pooler_output"]
        #     train_features['features'].append(features)
        #     train_features['actions'].append(actions)
        batch_size = 1
        if split == 'val':
            batch_size = 1
        for i in tqdm(range(0, len(dataset), batch_size)):
            print(f"Processing batch {i//batch_size}")
            front_images, questions, answers, action_tokens, front_images_org, training_types, actions, actions_masks, ego_traj_past_tokens = zip(*[dataset[j] for j in range(i, min(i+batch_size, len(dataset)))])
            imgs = processor(images=front_images, padding="max_length", return_tensors="pt")["pixel_values"]
            imgs = imgs.cuda()
            features = model.vision_model(imgs)["pooler_output"]
            if split == 'train':
                if classes == 'command':
                    command = dataset.all_data[i]['command']
                elif classes == 'action_template':
                    actions_tmp = actions[0]
                    # get average distance in x and y direction
                    x_dist_mean = torch.mean(actions_tmp[1:, 0] - actions_tmp[:-1, 0])
                    y_dist_mean = torch.mean(actions_tmp[1:, 1] - actions_tmp[:-1, 1])

                    action_template, category = dataset.get_action_template(x_dist_mean, y_dist_mean)
                    command = str(category)
                if command not in train_features.keys():
                    train_features[command] = {'features': [], "actions": []}
                train_features[command]['features'].extend(features.cpu().detach().numpy())
                train_features[command]['actions'].extend(actions)
            elif split == 'val':
                if dataset.all_data[i]['scene_token'] not in train_features.keys():
                    train_features[dataset.all_data[i]['scene_token']] = {}
                if dataset.all_data[i]['timestamp'] not in train_features[dataset.all_data[i]['scene_token']].keys():
                    train_features[dataset.all_data[i]['scene_token']][dataset.all_data[i]['timestamp']] = {}
                train_features[dataset.all_data[i]['scene_token']][dataset.all_data[i]['timestamp']]['features'] = features.cpu().detach().numpy()
                train_features[dataset.all_data[i]['scene_token']][dataset.all_data[i]['timestamp']]['actions'] = actions

        # save
        try:
            path = f'data/DriveLM_v2/img_features/{model_type.split("/")[-1]}/{split}'
            Path(path).mkdir(parents=True, exist_ok=True)
            with open(f'{path}/img_features_command_{classes}.pkl', 'wb') as f:
                pickle.dump(train_features, f)
        except:
            breakpoint()


    if test_discretize:
        for i in range(len(dataset)):
            dataset[i]
            print(i)
            pass
            # if i > 10:
            #     break


        # test discretize
        # from -10 to 100, step 0.1
        x = np.arange(-10, 100, 0.1)
        y = [0] * len(x)

        # y = np.arange(-50, 50, 0.1)
        # x = [0] * len(y)
        # y = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]
        black_image = np.zeros((800, 800, 3), dtype=np.uint8)
        cv2.circle(black_image, (int(y[0]*8+400), int(-x[0]*4+600)), 5, (255, 0, 0), -1)

        x_discretized = []
        y_discretized = []

        x_bin_list = []
        y_bin_list = []
        x_bin1, y_bin1 = dataset._discretize_array(x, y, dataset.x_range, dataset.y_range, dataset.action_bins, dataset.discretize)
        x_tmp1, y_tmp1 = dataset._bin_to_meters_array(x_bin1, y_bin1, dataset.x_range, dataset.y_range, dataset.action_bins, dataset.discretize)

        for i in range(len(x)):
            x_bin, y_bin = dataset._discretize(x[i], y[i], dataset.x_range, dataset.y_range, dataset.action_bins, dataset.discretize)
            x_bin_list.append(x_bin)
            y_bin_list.append(y_bin)
            x_tmp, y_tmp = dataset._bin_to_meters(x_bin, y_bin, dataset.x_range, dataset.y_range, dataset.action_bins, dataset.discretize)

            x_discretized.append(x_tmp)
            y_discretized.append(y_tmp)
            
            # draw original points and discretized points in image and save
            cv2.circle(black_image, (int(y_tmp*8+400), int(-x_tmp*4+600)), 3, (0, 255, 0), -1)
            cv2.circle(black_image, (int(y[i]*8+400), int(-x[i]*4+600)), 1, (0, 0, 255), -1)

        for i in range(len(x_bin_list)):
            # check if the discretized points are the same
            if x_bin_list[i] != x_bin1[i] or y_bin_list[i] != y_bin1[i]:
                print(f"discretize error: {x_bin_list[i]}, {y_bin_list[i]}")
                print(f"discretize error: {x_bin1[i]}, {y_bin1[i]}")
            
            if x_discretized[i] != x_tmp1[i] or y_discretized[i] != y_tmp1[i]:
                print(f"bin to meters error: {i}: {x_discretized[i]}, {y_discretized[i]}")
                print(f"bin to meters error: {i}: {x_tmp1[i]}, {y_tmp1[i]}")

        cv2.imwrite(f"test_discretize_sqrt.png", black_image)

        # get bin sizes
        x_bin_size = []
        y_bin_size = []

        # remove duplicates
        x_discretized = list(dict.fromkeys(x_discretized))
        y_discretized = list(dict.fromkeys(y_discretized))
        for i in range(len(x_discretized)-1):
            x_bin_size.append(x_discretized[i+1] - x_discretized[i])
            # y_bin_size.append(y_discretized[i+1] - y_discretized[i])

        # plot
        black_image = np.zeros((800, 800, 3), dtype=np.uint8)
        for i in range(len(x_bin_size)):
            # bin size on y axis and i on x axis
            cv2.circle(black_image, (int(i*4+400), int(-x_bin_size[i]*10+600)), 2, (0, 255, 0), -1)
            cv2.circle(black_image, (int(i*4+400), int(-x_discretized[i]*10+600)), 2, (0, 0, 255), -1)
        
        cv2.imwrite(f"test_discretize_sqrt_bin_size.png", black_image)




        for i in range(len(dataset)):
            print(dataset[i])
            pass
            if i > 10:
                break
