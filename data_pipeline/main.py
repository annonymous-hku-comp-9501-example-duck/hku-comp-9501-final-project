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

def post_process_DataInfo(data_info):
    gt_boxes = np.array(data_info["gt_boxes"])
    gt_names = data_info["gt_names"]
    gt_attributes = data_info["gt_attrs"]
    gt_dist = []

    front_weight = 0.95

    for i in range(len(gt_boxes)):
        x, y, z = gt_boxes[i][0], gt_boxes[i][1], gt_boxes[i][2]
        dist = math.sqrt(x*x + y*y + z*z)
        if y > 0:
            dist = dist * front_weight**y
        gt_dist.append(dist)

    sorted_gt_dist = sorted(range(len(gt_dist)), key=lambda k: gt_dist[k])
    # print(sorted_gt_dist)

    sorted_gt_boxes = []
    sorted_gt_names = []
    sorted_gt_attrs = []

    max_len = 7

    for i in range(min(max_len, len(sorted_gt_dist))):
        sorted_idx = sorted_gt_dist[i]
        sorted_gt_boxes.append(gt_boxes[sorted_idx])
        sorted_gt_names.append(gt_names[sorted_idx])
        sorted_gt_attrs.append(gt_attributes[sorted_idx])
    
    data_info['sorted_gt_boxes'] = sorted_gt_boxes
    data_info['sorted_gt_names'] = sorted_gt_names
    data_info['sorted_gt_attrs'] = sorted_gt_attrs

    return data_info

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

test_items = 100

def main(VIS_FLAG=False):
    data_path = '/mnt/disk01/nuscenes/'
    language_path = 'language'
    xjw = True
    # data_info_all_w_metadata = pickle.load(open(os.path.join(data_path, "nuscenes_infos_temporal_train.pkl"), "rb"))
    # data_info_all = data_info_all_w_metadata['infos']

    data_info_all = pickle.load(open(os.path.join(data_path, "nuscenes_infos_temporal_val_xjw.pkl"), "rb"))
    topo_info_all = pickle.load(open(os.path.join(data_path, "openlane_v2_nus/data_dict_subset_B_val_new.pkl"), "rb"))

    # data_info_all = pickle.load(open(os.path.join(data_path, "nuscenes_infos_temporal_train_xjw_100.pkl"), "rb"))
    # topo_info_all = pickle.load(open(os.path.join(data_path, "data_dict_subset_B_train_new_100.pkl"), "rb"))

    topo_key_map = dict()
    for tk in topo_info_all.keys():
        tk_short = [tk[0], tk[2]]
        tk_short = tuple(tk_short)
        topo_key_map[tk_short] = tk 

    language_data = {}
    scene_imgs = {}
    for i in tqdm(range(len(data_info_all)), desc ="build success"):
        # if i > 1000:
        #     break
        data_info = data_info_all[i]
        timestamp = data_info["cams"]["CAM_FRONT"]["data_path"].split("__")[-1][:-4]
        scene_token = data_info['scene_token']
        scene_imgs[scene_token] = []

        if data_info["gt_boxes"].shape[0] == 0:
            continue
        if xjw:
            objects_info = {
                "names": data_info["gt_names"],
                "locations": data_info["gt_boxes"][:, :2],
                "attributes": data_info["gt_attrs"],
                "velocities": data_info["gt_velocity"],
            }
        else:
            objects_info = {
                "names": data_info["gt_names"],
                "locations": data_info["gt_boxes"][:, :2],
                # "attributes": data_info["gt_attrs"],
                "velocities": data_info["gt_velocity"],
            }

        if timestamp in topo_info_all:
            traffic_info = topo_info_all[timestamp]["annotation"]
            traffic_info = {
                "lane_centerline": traffic_info["lane_centerline"],  # lane ?
                "traffic_element": traffic_info["traffic_element"],
                "topology_lclc": traffic_info["topology_lclc"],
                "topology_lcte": traffic_info["topology_lcte"],
            }
        else:
            continue
            traffic_info = None

        # below is not used anymore
        # topo_key = ['train', timestamp]
        # topo_key = tuple(topo_key)

        # if topo_key in topo_key_map:
        #     traffic_info = topo_info_all[topo_key_map[topo_key]]["annotation"]
        #     traffic_info = {
        #         "lane_centerline": traffic_info["lane_centerline"],  # lane ?
        #         "traffic_element": traffic_info["traffic_element"],
        #         "topology_lclc": traffic_info["topology_lclc"],
        #         "topology_lcte": traffic_info["topology_lcte"],
        #     }
        # else:
        #     traffic_info = None
        #     continue

        # print("Begin to generate scene graph")

        # skip for genenrating figure only
        # scene_graph = get_scene_graph(timestamp, objects_info, traffic_info)

        # # begin to prepare questions and answer
        # questions_all = []
        # answers_all = []
        # for type_ in ["A", "B", "C", "D"]:
        #     questions, answers = get_existence_text(type_, scene_graph)
        #     questions_all.extend(questions)
        #     answers_all.extend(answers)
        #     questions, answers = get_counting_text(type_, scene_graph)
        #     questions_all.extend(questions)
        #     answers_all.extend(answers)

        # if traffic_info is not None:
        #     for type_ in ["A", "B", "E"]:
        #         questions, answers = get_traffic_topology_text("A", scene_graph)
        #         questions_all.extend(questions)
        #         answers_all.extend(answers)

        # questions_all_new = []
        # answers_all_new = []
        # for j in range(len(questions_all)):
        #     if "Wrong" not in answers_all[j] and "No." not in answers_all[j] and "0" not in answers_all[j]:
        #         questions_all_new.append(questions_all[j])
        #         answers_all_new.append(answers_all[j])

        # language_data[data_info["timestamp"]] = {"q": questions_all_new, "a": answers_all_new}

        # check language_data
        # pdb.set_trace()

        # generate image for annotation
        sorted_data_info = post_process_DataInfo(copy.deepcopy(data_info))
        segment_id = topo_info_all[timestamp]['segment_id']
        save_folder = "anno_fig_numbered_val"
        Path(os.path.join(data_path, language_path, f"{save_folder}/{segment_id}-{scene_token}")).mkdir(parents=True,exist_ok=True)
        if VIS_FLAG and i % 5 == 0:
            generate_question(data_info=sorted_data_info, save_path=os.path.join(data_path, language_path, f"{save_folder}/{segment_id}-{scene_token}/{i}-{timestamp}_qa.xlsx"))
            img_anno = show_image_anno(data_info=sorted_data_info)
            mmcv.imwrite(img_anno, os.path.join(data_path, language_path, f"{save_folder}/{segment_id}-{scene_token}/{i}-{timestamp}_anno.jpg"))
        else:
            img_raw = show_image_raw(data_info=data_info)
            mmcv.imwrite(img_raw, os.path.join(data_path, language_path, f"{save_folder}/{segment_id}-{scene_token}/{i}-{timestamp}.jpg"))
        
        if False and VIS_FLAG and i % 10 == 0 :
            # print("generate image!")
            # print(i % 10 == 0)
            # print(i)
            img = show_image(data_info)
            # mmcv.imwrite(img, f"vis/counting_4/{data_info['timestamp']}.jpg")
            index = np.arange(len(questions_all_new))
            np.random.shuffle(index)
            questions_all = np.array(questions_all_new)[index]
            answers_all = np.array(answers_all_new)[index]
            language_img = np.zeros((2000, img.shape[1], 3)).astype(np.uint8) * 255
            num = min(50, len(questions_all))
            for n in range(num):
                cv2.putText(
                    language_img,
                    questions_all[n] + " " + str(answers_all[n]),
                    (50, 50 * n + 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (72, 101, 241),
                    3,
                )
            img = np.concatenate((img, language_img), 0)
            mmcv.imwrite(img, os.path.join(data_path, language_path, f"vis/testbench/{data_info['timestamp']}.jpg"))

    # mmcv.dump(language_data, os.path.join(data_path, language_path, "language_anno.pkl"))

    # language_data_json = json.dumps(language_data, indent = 4) 

    # with open(os.path.join(data_path, language_path, f"xjw_sample_{test_items}.json"), "w") as outfile:
    #     json.dump(language_data, outfile)


if __name__ == "__main__":
    VIS_FLAG = True
    main(VIS_FLAG)
