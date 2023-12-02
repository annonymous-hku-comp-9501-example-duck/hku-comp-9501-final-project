import os
import json
import re
import cv2
import numpy as np


def click_to_object(jpg_file_path, json_file_path, click_list):
    cam_front_dict = {}
    cam_front_left_dict = {}
    cam_front_right_dict = {}
    cam_back_dict = {}
    cam_back_right_dict = {}
    cam_back_left_dict = {}
    result_dict = {}
    filename = os.path.basename(jpg_file_path)
    match = re.search(r'\d+', filename)
    index = int(match.group())
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    object_list = data['frames'][f"frame{index}"].get('items', [])
    for key in object_list.keys():
        for area in object_list[key]['area_rect'].keys():
            if area == 'CAM_FRONT':
                cam_front_dict[key] = object_list[key]['area_rect'][area]
            elif area == 'CAM_FRONT_LEFT':
                cam_front_left_dict[key] = object_list[key]['area_rect'][area]
            elif area == 'CAM_FRONT_RIGHT':
                cam_front_right_dict[key] = object_list[key]['area_rect'][area]
            elif area == 'CAM_BACK':
                cam_back_dict[key] = object_list[key]['area_rect'][area]
            elif area == 'CAM_BACK_LEFT':
                cam_back_left_dict[key] = object_list[key]['area_rect'][area]
            elif area == 'CAM_BACK_RIGHT':
                cam_back_right_dict[key] = object_list[key]['area_rect'][area]
    for x, y in click_list:
        aim_dic = {}
        temp_dict = {}
        if 0<x<960 and 0<y<540:
            aim_dic = cam_front_left_dict
        elif 960<x<1920 and 0<y<540:
            aim_dic = cam_front_dict
        elif 1920<x<2880 and 0<y<540:
            aim_dic = cam_front_right_dict
        elif 0<x<960 and 540<y<1080:
            aim_dic = cam_back_left_dict
        elif 960<x<1920 and 540<y<1080:
            aim_dic = cam_back_dict
        elif 1920<x<2880 and 540<y<1080:
            aim_dic = cam_back_right_dict
        else:
            print('click out of range')
            continue
        for key in aim_dic.keys():
            if aim_dic[key][0][0] < x < aim_dic[key][1][0] and aim_dic[key][0][1] < y < aim_dic[key][1][1]:
                temp_dict[key] = aim_dic[key]
        key, value = select_object(temp_dict)
        result_dict[key] = value
    draw_rect(jpg_file_path, result_dict)

    return result_dict


def select_object(temp_dict):
    key = next(iter(temp_dict.keys()))
    value = temp_dict[key]
    return key, value


def draw_rect(jpg_file_path, object_dict):
    #filename = os.path.basename(jpg_file_path)
    #save_path = os.path.join('/home/PJLAB/xiechengen/OpenDrive/GUI/rect', filename)
    #img = cv2.imread(jpg_file_path)
    #for key in object_dict.keys():
        #x1 = object_dict[key][0][0]
        #y1 = object_dict[key][0][1]
        #x2 = object_dict[key][1][0]
        #y2 = object_dict[key][1][1]
        #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #cv2.imwrite(save_path, img)
    pass


def process_one_scene(jpg_folder_path, object_json_file_path, click_list: list, frame_list: list):
    object_dict = {}
    for root, _, files in os.walk(jpg_folder_path):
        for file in files:
            if not file in frame_list:
                continue
            jpg_file_path = os.path.join(root, file)
            part_result_dict = click_to_object(jpg_file_path, object_json_file_path, click_list)
            object_dict[f"{file}"] = part_result_dict
    return object_dict


def process_all_scene(jpg_folder_all_path, object_json_folder_path, click_list_dict: dict, frame_list_dict: dict, annotation_id: str):
    scene_object_dict = {}
    for scene_id in frame_list_dict.keys():
        jpg_folder_path = os.path.join(jpg_folder_all_path, scene_id)
        object_json_file_path = os.path.join(object_json_folder_path, f"{scene_id}.json")
        click_list = click_list_dict[scene_id][annotation_id]
        frame_list = frame_list_dict[scene_id][annotation_id]
        object_dict = process_one_scene(jpg_folder_path, object_json_file_path, click_list, frame_list)
        scene_object_dict[scene_id] = object_dict
    return scene_object_dict


def read_click_json(click_json_path):
    click_list_dict = {}
    with open(click_json_path, 'r') as file:
        data = json.load(file)
    for picture in data:
        annotation_list = picture['annotations']
        click_list_anno_dict = {}
        for annotation in annotation_list:
            result_list = annotation['result']
            click_list = []
            for result in result_list:
                click_x = result['original_width'] * result['value']['x'] / result['original_width']
                click_y = result['original_height'] * result['value']['y'] / result['original_height']
                click_list.append((click_x, click_y))
            click_list_anno_dict[annotation['id']] = click_list
        click_list_dict[picture['id']] = click_list_anno_dict

    return click_list_dict


def read_frame_json(frame_json_path):
    frame_list_dict = {}
    with open(frame_json_path, 'r') as file:
        data = json.load(file)
    for scene in data:
        annotation_list = scene['annotations']  
        frame_list_anno_dict = {}
        for annotation in annotation_list:
            frame_sequence = annotation['result'][0]['value']['sequence']
            frame_list = []
            for frame in frame_sequence:
                frame_list.append(frame['frame'])
            frame_list_anno_dict[annotation['id']] = frame_list
        frame_list_dict[scene['id']] = frame_list_anno_dict
    return frame_list_dict



#json_file_path = '11000-e7ef871f77f44331aefdebc24ec034b7.json'
#jpg_file = '1-1533201470412460.jpg'
#click_list = [(1000, 270)]
#result = click_to_object(jpg_file, json_file_path, click_list)
#print(result)

click_json_file_path = '/home/PJLAB/xiechengen/OpenDrive/GUI/Keypoints_export.json'
frame_json_file_path = '/home/PJLAB/xiechengen/OpenDrive/GUI/Keyframes_export.json'

click_list_dict = read_click_json(click_json_file_path)
frame_list_dict = read_frame_json(frame_json_file_path)


key_frames = frame_list_dict['11000-e7ef871f77f44331aefdebc24ec034b7'][20374852]
click_list = click_list_dict[key_frames[0]][20375256]
jpg_file_path = f"/home/PJLAB/xiechengen/OpenDrive/GUI/11000-e7ef871f77f44331aefdebc24ec034b7/{key_frames[0]}.jpg"
json_file_path = '/home/PJLAB/xiechengen/OpenDrive/GUI/11000-e7ef871f77f44331aefdebc24ec034b7.json'
result = click_to_object(jpg_file_path, json_file_path, click_list)
print(result)

