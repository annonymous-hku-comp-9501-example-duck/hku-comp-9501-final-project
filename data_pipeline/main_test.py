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
import json
from json import JSONEncoder
import math
import copy
import pandas as pd
import tkinter
import openpyxl
import xlsxwriter
import random
from random import sample
from itertools import islice
from datetime import datetime


def chinese_english_dict(chinese_qa, english_qa):
    translate_dict = dict()
    pass


import excel_processor

# call once for one frame
def generate_question(data_info: dict, save_path=None):

    random.seed(datetime.now().timestamp())
    '''
        General comment:
        1. 关于action类的回答, 尽量集中采样到同一个主体.
            related q: 1.6, the take-action entity should be consistent in one frame. DONE
        2. 在提问A是否会基于B改变运动状态时, 是否需要把 自车 也列入B的采样范围
            related q: 1.4, adding ego vehicle in sampling of observation. DONE
        3. 提问主体(如基于observation是否会改变运动状态, 是否在行进方向上), 最好限定在移动中的车辆, 考虑相邻物体
            related q: 1.2AB(DONE), 1.3, 1.4A, 1.5, 1.6A, 
        4. 我的想法是2.7是否有可能将动作和提问的物体结合起来, 就比如采取右转加速的动作, 可能相撞的物体最好也是在右侧的物体
            related q: 2.7, 1.6
        5. 提问两者关系的问题, 最好限定两个物体的距离较近
            related q: 1.2, 1.4, 1.6, 
    '''

    gt_boxes = np.array(data_info["sorted_gt_boxes"])
    gt_boxes[:, 2] -= 0.5 * gt_boxes[:, 5]
    gt_names = data_info["sorted_gt_names"]
    gt_attributes = data_info["sorted_gt_attrs"]
    gt_ids = []
    gt_cat_num = []     # this is the <cXX> list, [c0, c1, c2, xxx], i.e., <c0> = gt_cat_num[0], <c0>bbox = gt_boxes[0] (x,y,z,w,l,h,yaw)
    category_count = {
        "car": 0,
        "truck": 0,
        "construction_vehicle": 0,
        "bus": 0,
        "trailer": 0,
        "barrier": 0,
        "motorcycle": 0,
        "bicycle": 0,
        "pedestrian": 0,
        "traffic_cone": 0,
    }

    for i in range(len(gt_names)):
        # gt_ids.append(gt_names[i] + "_" + gt_attributes[i] + "_" + str(category_count[gt_names[i]]))
        if gt_names[i] in category_count:
            gt_ids.append(gt_names[i] + "_" + gt_attributes[i] + "_" + str(category_count[gt_names[i]]))
            gt_cat_num.append(f"<c{i}>")
            category_count[gt_names[i]] += 1
        else:
            category_count[gt_names[i]] = 0
            gt_ids.append(gt_names[i] + "_" + gt_attributes[i] + "_" + str(category_count[gt_names[i]]))

    if len(gt_cat_num) <= 1:
        return
    
    # write excel
    if save_path is None:
        workbook = xlsxwriter.Workbook(os.path.join(data_path, language_path, f"vis/testbench/qa_sample_wb.xlsx"))
    else:
        workbook = xlsxwriter.Workbook(save_path)  

    worksheet = workbook.add_worksheet()

    header_format = workbook.add_format(
        {
            "border": 1,
            "bg_color": "#C6EFCE",
            "bold": True,
            "text_wrap": True,
            "valign": "vcenter",
            "indent": 1,
        }
    )

    fillblank_format = workbook.add_format(
        {
            "border": 1,
            "bg_color": "#b0cc96",
            "bold": False,
            "text_wrap": True,
            "valign": "vcenter",
            "indent": 1,
            'font_color': 'red',
        }
    )

    if True:
        worksheet.set_column("A:A", 10)
        worksheet.set_column("B:B", 50)
        worksheet.set_column("C:C", 30)
        worksheet.set_column("D:K", 30)
        worksheet.set_column("F:F", 50)

    if True:
        worksheet.write("A1", "问题分类", header_format)
        worksheet.write("B1", "问题", header_format)
        worksheet.write("C1", "子问题字段1", header_format)
        worksheet.write("D1", "子问题字段2", header_format)
        worksheet.write("E1", "子问题字段3", header_format)
        worksheet.write("F1", "子问题字段4", header_format)
        worksheet.write("G1", "子问题字段5", header_format)
        worksheet.write("H1", "子问题字段6", header_format)
        worksheet.write("I1", "子问题字段7", header_format)
        worksheet.write("J1", "子问题字段8", header_format)
        worksheet.write("K1", "子问题字段9", header_format)

    row_num = 2 # use it then +=1

    # q1.1
    # need to focus more on nearest obj
    sample_list = sample(range(0, len(gt_cat_num)), min(5, len(gt_cat_num)))

    for i_q_pred_11 in sample_list:
        # this is a question template
        # how to write pure text, see below
        worksheet.write(f"A{row_num}", "预测")

        worksheet.write(f"B{row_num}", "1.1 目标A的当前观测移动状态 (A非交通标志)")
        worksheet.write(f"C{row_num}", f"{gt_cat_num[i_q_pred_11]}的当前观测移动状态为")

        # how to write drop down list with options, see below
        worksheet.data_validation(f'D{row_num}', {'validate': 'list', 'source': ["请选择", "静止", "前进", "左转", "右转"]})
        worksheet.write(f"D{row_num}", "请选择", fillblank_format)

        row_num += 1

    # q1.2
    sample_list = sample(range(0, len(gt_cat_num)), min(5, len(gt_cat_num)))
    # select B, and B should be moving
    B_list_12 = [i_12 for i_12, x_12 in enumerate(gt_attributes) if x_12 == 'moving']

    for i_q_pred_12 in sample_list:
        # select A 
        # A_12 = i_q_pred_12 if i_q_pred_12 not in B_list_12
        B_12 = random.choice(B_list_12)
        if i_q_pred_12 == B_12:
            sample_list_exp = copy.deepcopy(sample_list)
            sample_list_exp.remove(B_12)
            A_12 = random.choice(sample_list_exp)
        else:
            A_12 = i_q_pred_12
        
        worksheet.write(f"A{row_num}", "预测")
        worksheet.write(f"B{row_num}", "1.2 A是否会在B的行进方向上 (A,B非交通标志)")
        worksheet.write(f"C{row_num}", f"{gt_cat_num[A_12]}")

        worksheet.data_validation(f"D{row_num}", {'validate': 'list', 'source': ["请选择", "在", "不在"]})
        worksheet.write(f"D{row_num}", "请选择", fillblank_format)

        # select a cXX that is not the same as C column
        # sample_list_exp = copy.deepcopy(sample_list)
        # sample_list_exp.remove(i_q_pred_12)
        # chosen_i = random.choice(sample_list_exp)

        worksheet.write(f"E{row_num}", f"{gt_cat_num[B_12]}的行进方向上")

        row_num += 1
    
    # q1.3
    sample_list = sample(range(0, len(gt_cat_num)), min(5, len(gt_cat_num)))

    for i_q_pred_13 in sample_list:    
        worksheet.write(f"A{row_num}", "预测")
        worksheet.write(f"B{row_num}", "1.3 A是否会在自车的行进方向上 (A非交通标志)")
        worksheet.write(f"C{row_num}", f"{gt_cat_num[i_q_pred_13]}")

        worksheet.data_validation(f"D{row_num}", {'validate': 'list', 'source': ["请选择", "在", "不在"]})
        worksheet.write(f"D{row_num}", "请选择", fillblank_format)

        worksheet.write(f"E{row_num}", "自车的行进方向上")

        row_num += 1
        
    # q1.4
    sample_list = sample(range(0, len(gt_cat_num)), min(5, len(gt_cat_num)))
    sample_list_14 = copy.deepcopy(sample_list)
    sample_list_14.append(-1)   # -1 stand for ego vehicle

    for i_q_pred_14 in sample_list_14: 
        worksheet.write(f"A{row_num}", "预测")
        worksheet.write(f"B{row_num}", "1.4 A是否会基于B改变其运动状态 (A非交通标志, B含交通标志)")
        worksheet.write(f"C{row_num}", f"基于对{gt_cat_num[i_q_pred_14]}的观察")

        # select a cXX that is not the same as C column
        sample_list_exp = copy.deepcopy(sample_list_14)
        sample_list_exp.remove(i_q_pred_14)
        chosen_i = random.choice(sample_list_exp)
        if chosen_i >= 0:
            worksheet.write(f"D{row_num}", f"{gt_cat_num[chosen_i]}")
        else:
            worksheet.write(f"D{row_num}", "自车")

        worksheet.data_validation(f"E{row_num}", {'validate': 'list', 'source': ["请选择", "会", "不会"]})
        worksheet.write(f"E{row_num}", "请选择", fillblank_format)
        worksheet.write(f"F{row_num}", "改变其运动状态")

        row_num += 1

    # q1.5
    sample_list = sample(range(0, len(gt_cat_num)), min(5, len(gt_cat_num)))

    for i_q_pred_15 in sample_list: 
        worksheet.write(f"A{row_num}", "预测")
        worksheet.write(f"B{row_num}", "1.5 A的未来预计目标状态 (A非交通标志)")
        worksheet.write(f"C{row_num}", f"{gt_cat_num[i_q_pred_15]}的未来预计目标状态为")

        # worksheet.data_validation('D13', {'validate': 'list', 'source': ["静止", "直行", "左转", "右转"]})
        worksheet.data_validation(f'D{row_num}', {'validate': 'list', 'source': ["请选择", "静止", "直行", "左转", "右转"]})
        worksheet.write(f"D{row_num}", "请选择", fillblank_format)

        row_num += 1

    # q1.6 
    sample_list = sample(range(0, len(gt_cat_num)), min(5, len(gt_cat_num)))
    action_list_16 = ["请选择", "无", "加速", "减速", "保持速度", "左转", "右转", "变至左车道", "变至右车道"]
    cause_list_16 = ["请选择", "不存在安全问题", "保持安全距离", "避免碰撞", "遵守交通规则"]
    prob_list_16 = ["请选择", "高", "中", "低"]

    A_16 = random.choice(sample_list)
    sample_list_16 = copy.deepcopy(sample_list)
    sample_list_16.remove(A_16)

    for i_q_pred_16 in sample_list_16: 
        worksheet.write(f"A{row_num}", "预测")
        worksheet.write(f"B{row_num}", "1.6 A基于B可能采取的行动及可能性 (A非交通标志, B含交通标志)")
        worksheet.write(f"C{row_num}", f"基于对{gt_cat_num[i_q_pred_16]}的观察")

        # sample_list_exp = copy.deepcopy(sample_list)
        # sample_list_exp.remove(i_q_pred_16)
        # chosen_i = random.choice(sample_list_exp)
        worksheet.write(f"D{row_num}", f"{gt_cat_num[A_16]}可能采取的行动为")

        worksheet.data_validation(f"E{row_num}", {'validate': 'list', 'source': action_list_16})
        worksheet.write(f"E{row_num}", "请选择", fillblank_format)

        worksheet.write(f"F{row_num}", "原因是")

        worksheet.data_validation(f"G{row_num}", {'validate': 'list', 'source': cause_list_16})
        worksheet.write(f"G{row_num}", "请选择", fillblank_format)

        worksheet.write(f"H{row_num}", "采取该行动的概率为")

        worksheet.data_validation(f"I{row_num}", {'validate': 'list', 'source': prob_list_16})
        worksheet.write(f"I{row_num}", "请选择", fillblank_format)

        row_num += 1

    # q2.1
    worksheet.write(f"A{row_num}", "规划")
    worksheet.write(f"B{row_num}", "2.1 自车应该注意什么交通信号")
    worksheet.write(f"C{row_num}", "自车应当注意的交通信号为")
    worksheet.data_validation(f'D{row_num}', {'validate': 'list', 'source': ["请选择", "无", "红灯", "黄灯", "绿灯", "左转", "右转", "直行",
                                                                             "禁止左转", "禁止右转", "掉头", "禁止掉头", "略左直行", "略右直行"]})
    worksheet.write(f"D{row_num}", "请选择", fillblank_format)
    row_num += 1

    # q2.2
    worksheet.write(f"A{row_num}", "规划")
    worksheet.write(f"B{row_num}", "2.2 哪些车道线需要被注意到")
    worksheet.write(f"C{row_num}", "暂时留空", fillblank_format)
    worksheet.write(f"D{row_num}", "需要被注意到")
    row_num += 1

    # q2.3
    worksheet.write(f"A{row_num}", "规划")
    worksheet.write(f"B{row_num}", "2.3 自车的目标动作")
    worksheet.write(f"C{row_num}", "自车的目标动作为")
    worksheet.data_validation(f'D{row_num}', {'validate': 'list', 'source': ["请选择", "静止", "直行", "左转", "右转"]})
    worksheet.write(f"D{row_num}", "请选择", fillblank_format)
    row_num += 1

    # q2.4
    notice_list = ["首先注意到", "其次注意到", "最后注意到", ]
    for i_q_pred_24 in range(len(notice_list)):
        worksheet.write(f"A{row_num}", "规划")
        worksheet.write(f"B{row_num}", "2.4 自车按照什么逻辑去到达下一个可能的位置")

        worksheet.write(f"C{row_num}", f"{notice_list[i_q_pred_24]}")
        worksheet.data_validation(f'D{row_num}', {'validate': 'list', 'source': gt_cat_num + ["请选择", ]})
        worksheet.write(f"D{row_num}", "请选择", fillblank_format)

        worksheet.write(f"E{row_num}", "他的运动状态是")
        worksheet.data_validation(f'F{row_num}', {'validate': 'list', 'source': ["请选择", "静止", "直行", "左转", "右转", "交通标志"]})
        worksheet.write(f"F{row_num}", "请选择", fillblank_format)

        worksheet.write(f"G{row_num}", "因而自车应当采取的行动为")
        worksheet.data_validation(f'H{row_num}', {'validate': 'list', 'source': ["请选择", "无", "加速", "减速", "保持速度", "跟随前车",
                                                                                "变至左车道", "变至右车道", "左转", "右转"]})
        worksheet.write(f"H{row_num}", "请选择", fillblank_format)
        row_num += 1

    # q2.5
    worksheet.write(f"A{row_num}", "规划")
    worksheet.write(f"B{row_num}", "2.5 自车应考虑目标的逻辑顺序")
    worksheet.write(f"C{row_num}", "自车应考虑目标的优先级为(按照降序)")
    worksheet.data_validation(f'D{row_num}', {'validate': 'list', 'source': gt_cat_num + ["请选择", "无", ]})
    worksheet.write(f"D{row_num}", "请选择", fillblank_format)
    worksheet.data_validation(f'E{row_num}', {'validate': 'list', 'source': gt_cat_num + ["请选择", "无", ]})
    worksheet.write(f"E{row_num}", "请选择", fillblank_format)
    worksheet.data_validation(f'F{row_num}', {'validate': 'list', 'source': gt_cat_num + ["请选择", "无", ]})
    worksheet.write(f"F{row_num}", "请选择", fillblank_format)
    worksheet.data_validation(f'G{row_num}', {'validate': 'list', 'source': gt_cat_num + ["请选择", "无", ]})
    worksheet.write(f"G{row_num}", "请选择", fillblank_format)
    worksheet.data_validation(f'H{row_num}', {'validate': 'list', 'source': gt_cat_num + ["请选择", "无", ]})
    worksheet.write(f"H{row_num}", "请选择", fillblank_format)
    worksheet.data_validation(f'I{row_num}', {'validate': 'list', 'source': gt_cat_num + ["请选择", "无", ]})
    worksheet.write(f"I{row_num}", "请选择", fillblank_format)
    row_num += 1

    # q2.6
    # action_list_26 = ["直行加速", "直行减速", "直行匀速", "右转加速", "右转减速", "右转匀速", "左转加速", "左转减速", "左转匀速", "踩死刹车"]
    action_list_26 = ["无", "加速", "减速", "保持速度", "左转", "右转", "变至左车道", "变至右车道", "跟随前车"]
    sample_list_26 = copy.deepcopy(sample_list)
    for i_q_pred_26 in range(min(3, len(sample_list))):
        chosen_i = random.choice(sample_list_26)
        sample_list_26.remove(chosen_i)
        worksheet.write(f"A{row_num}", "规划")
        worksheet.write(f"B{row_num}", "2.6 基于观测，自车可能采取的行动")

        worksheet.write(f"C{row_num}", f"基于对{gt_cat_num[chosen_i]}的观察")

        worksheet.write(f"D{row_num}", "自车可能采取的行动为")
        worksheet.data_validation(f'E{row_num}', {'validate': 'list', 'source': action_list_26 + ["请选择", ]})
        worksheet.write(f"E{row_num}", "请选择", fillblank_format)

        worksheet.write(f"F{row_num}", "原因是")
        worksheet.data_validation(f'G{row_num}', {'validate': 'list', 'source': ["请选择", "不存在安全问题", "保持安全距离", "避免碰撞",
                                                                                 "遵守交通规则"]})
        worksheet.write(f"G{row_num}", "请选择", fillblank_format)

        worksheet.write(f"H{row_num}", "采取该行动的概率为")
        worksheet.data_validation(f'I{row_num}', {'validate': 'list', 'source': ["请选择", "高", "中", "低"]})
        worksheet.write(f"I{row_num}", "请选择", fillblank_format)
        row_num += 1

    # q2.7
    '''
        @Chonghao Sima 虽然为我本意是提问里直接用”优先级为1的物体“这样的表述,
        不过这样就需要语言模型自己去理解替换了
    '''
    action_list_27 = ["直行加速", "直行减速", "直行匀速", "右转加速", "右转减速", "右转匀速", "左转加速", "左转减速", "左转匀速", "踩死刹车"]
    sample_list_27 = copy.deepcopy(sample_list)
    for i_q_pred_27 in range(min(5, len(sample_list))):     
        worksheet.write(f"A{row_num}", "规划")
        worksheet.write(f"B{row_num}", "2.7 自车采取动作后与B物体相撞的可能性")

        chosen_action = random.choice(action_list_27)
        chosen_cat = random.choice(sample_list_27)
        action_list_27.remove(chosen_action)
        sample_list_27.remove(chosen_cat)

        worksheet.write(f"C{row_num}", f"采取**{chosen_action}**的行动后")
        worksheet.write(f"D{row_num}", f"自车与{gt_cat_num[chosen_cat]}碰撞概率为")

        worksheet.data_validation(f'E{row_num}', {'validate': 'list',
                                                'source': ["请选择", "高", "中", "低"]})
        worksheet.write(f"E{row_num}", "请选择", fillblank_format)
        
        worksheet.write(f"F{row_num}", "**若碰撞概率为高**，责任归属是否有更大倾向为自车?")

        worksheet.data_validation(f'G{row_num}', {'validate': 'list', 'source': ["请选择", "是", "否"]})
        worksheet.write(f"G{row_num}", "请选择", fillblank_format)

        row_num += 1

    workbook.close()

# call once for one frame
def post_process_DataInfo(data_info: dict, ):
    """Compute the most near object and focus on front view (front, front_left, front_right)

    Important! This logic should not be change, unless annotation will mess up
        front_weight = 0.95, 
        dist = math.sqrt(x*x + y*y + z*z) # using L2 distance of xyz here
        for front_view obj (y>0), 
            dist = dist * front_weight**y, here y is the front distance
    having 7 as max_len, this is the max len of all sorted_* field

    Args:
        data_info (dict): info of frame, don't have sorted_* filed bur org ones
            gt_boxes
            gt_names
            gt_attrs

    Returns:
        dict: updated info of frame with sorted_* fields
            sorted_gt_boxes
            sorted_gt_names
            sorted_gt_attrs
            
    """
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
    data_info['sorted_gt_dist'] = sorted_gt_dist[:max_len]

    return data_info

# call once for one frame
def n_n_dist_mat(data_info: dict,):
    """Compute the n * n distant matrix on objects in sorted_* field,
        should have distance between every two object, and related direction

    Args:
        data_info (dict): info of frame, have sorted_* filed
            sorted_gt_boxes
            sorted_gt_names
            sorted_gt_attrs
            sorted_gt_dist
    
    Returns:
        dict: updated info of frame with 
            n_n_dist_mat: (distance, is_crossing)_{n * n}

    is_crossing: If the extension lines of the two vehicles' heading 
    directions intersect, set is_crossing = 1, else set is_crossing = 0.
    """
    # compute distance
    objects = np.array(data_info['sorted_gt_boxes'])
    n = len(objects)
    coordinate = np.array(objects[:,:3])
    coor_mat = np.stack([coordinate for _ in range(n)],axis = 0)
    dist_mat = coor_mat - coor_mat.transpose((1,0,2))
    dist_mat = np.sqrt((dist_mat * dist_mat).sum(axis=2))
    
    #compute crossing
    A1 = np.sin(objects[:,6])
    A2 = np.cos(objects[:,6])
    M1 = np.stack([A1 for _ in range(n)],axis = 0) # n*n
    M2 = np.stack([A2 for _ in range(n)],axis = 0)
    A = np.stack([np.stack((M1,M2),axis=-1),np.stack((-M1.transpose(),-M2.transpose()),axis=-1)],axis = -1)
    A = A.reshape(n*n,2,2) + np.identity(2)*1e-8 # avoid the sigular matrix error

    x_mat = coor_mat[:,:,0]
    y_mat = coor_mat[:,:,1]
    B = np.stack((y_mat.transpose()-y_mat,x_mat.transpose()),axis=-1)

    T = np.linalg.solve(A.reshape(n*n,2,2),B.reshape(n*n,2)).reshape(n,n,2)
    cross_mat = (((T>=0).sum(axis=-1))==2)

    data_info['n_n_dist_mat'] = np.stack((dist_mat,cross_mat),axis=-1)
    return data_info

# call when needed (maybe multiple times)
def find_nearest_obj(data_info: dict, base_obj_idx: int):
    """find the nearest obj idx of base_obj_idx, based on n_n_dist_mat

    Args:
        data_info (dict): info of frame, have sorted_* filed, n_n_dist_mat
            n_n_dist_mat

    Returns:
        int: nearest object idx of base_obj_idx
    """
    # 读取n_n_dist_mat, 大小为n*n*2
    n_n_dist_mat = np.array(data_info["n_n_dist_mat"])

    # 只保留距离
    n_n_dist = n_n_dist_mat[:, :, 0]

    # 找出base object相对其他object的距离
    try:
        row_base = n_n_dist[base_obj_idx, :]
    except IndexError:
        print(f'Index {base_obj_idx} out of range')
        return None
    # 找出最小的两个值的索引
    min_two_indices = np.argsort(row_base)[:2]
    # 返回第二小的值的索引
    return min_two_indices[1]

# WIP
def sorted_moving_obj(data_info: dict):
    """ Add a field of moving status, based on sorted_gt_attrs

    Args:
        data_info (dict): info of frame, have sorted_* filed
            sorted_gt_attrs

    Returns:
        dict: updated info of frame with 
            moving_status
    """
    pass

# call when needed (maybe multiple times)
def find_direction_obj(data_info: dict, base_obj_idx: int, ):
    """find the nearest obj idx in the moving direction of base_obj_idx, 
        based on n_n_dist_mat

    Args:
        data_info (dict): info of frame, have sorted_* filed, n_n_dist_mat
            sorted_gt_boxes
            sorted_gt_names
            sorted_gt_attrs
            sorted_gt_dist
            n_n_dist_mat

    Returns:
        int: the nearest obj idx in the moving direction of base_obj_idx.
             -1 if there is no object in interest is in the moving direction.
    """

    gt_indices = data_info['sorted_gt_dist']
    assert base_obj_idx in gt_indices, "Box index out of the list."
    sorted_base_idx = gt_indices.index(base_obj_idx)
    
    # we only consider the objects in the sorted_gt_dist
    print(sorted_base_idx, "nearest: ")
    gt_boxes = np.stack(data_info['sorted_gt_boxes'])
    moving_direction = gt_boxes[sorted_base_idx][-1]

    # project all boxes to the moving direction of the base object
    proj_vec = np.array([[np.cos(moving_direction)],[np.sin(moving_direction)]])
    proj_dist = gt_boxes[:, :2] @ proj_vec
    proj_dist -= proj_dist[sorted_base_idx]     # get the true distance
    proj_dist[proj_dist < 0] = np.inf
    proj_dist[sorted_base_idx] = np.inf         # ignore the objects in the back of the object

    nearest_forward = proj_dist.argmin()
    if proj_dist[nearest_forward] < np.inf:
        return gt_indices[nearest_forward]
    else:
        return -1


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
    

def take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))


def main(VIS_FLAG=False):
    data_path = '/mnt/disk01/nuscenes/'
    data_info_all_w_metadata = pickle.load(open(os.path.join(data_path, "nuscenes_infos_temporal_train.pkl"), "rb"))
    data_info_all = data_info_all_w_metadata['infos']
    topo_info_all = pickle.load(open(os.path.join(data_path, "openlane_v2_nus/data_dict_subset_B_train.pkl"), "rb"))

    topo_count = 0
    topo_miss = 0

    topo_key_map = dict()
    for tk in topo_info_all.keys():
        tk_short = [tk[0], tk[2]]
        tk_short = tuple(tk_short)
        topo_key_map[tk_short] = tk 

    for i in range(len(data_info_all)):
        data_info = data_info_all[i]
        timestamp = data_info["cams"]["CAM_FRONT"]["data_path"].split("__")[-1][:-4]

        topo_key = ['train', timestamp]
        topo_key = tuple(topo_key)

        if topo_key in topo_key_map:
            topo_count += 1
        else:
            topo_miss += 1

        # pdb.set_trace()

    print(topo_count)
    print(topo_miss)


xjw = True
data_path = '/mnt/disk01/nuscenes/'
language_path = 'language'

# set up read data path
# data_info mainly 3d bounding boxes of car ped truck
# data_info_all = pickle.load(open(os.path.join(data_path, "nuscenes_infos_temporal_train_xjw_100.pkl"), "rb"))
data_info_all = pickle.load(open(os.path.join(data_path, "nuscenes_infos_temporal_train_xjw_0.pkl"), "rb"))
# topo_info mainly traffic light, lanelines
# topo_info_all = pickle.load(open(os.path.join(data_path, "data_dict_subset_B_train_new_100.pkl"), "rb"))
topo_info_all = pickle.load(open(os.path.join(data_path, "data_dict_subset_B_train_new_0.pkl"), "rb"))

# topo_key_map = dict()
# for tk in topo_info_all.keys():
#     tk_short = [tk[0], tk[2]]
#     tk_short = tuple(tk_short)
#     topo_key_map[tk_short] = tk 

# language_data = {}
# random select a sample
i = 0

# data_info: dict
data_info = data_info_all[i]
timestamp = data_info["cams"]["CAM_FRONT"]["data_path"].split("__")[-1][:-4]
if data_info["gt_boxes"].shape[0] == 0:
    pass
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
    traffic_info = None

# this is used to generate perception q&a
scene_graph = get_scene_graph(timestamp, objects_info, traffic_info)

# begin to prepare questions and answer
questions_all = []
answers_all = []
for type_ in ["A", "B", "C", "D"]:
    questions, answers = get_existence_text(type_, scene_graph)
    questions_all.extend(questions)
    answers_all.extend(answers)
    questions, answers = get_counting_text(type_, scene_graph)
    questions_all.extend(questions)
    answers_all.extend(answers)

if traffic_info is not None:
    for type_ in ["A", "B", "E"]:
        questions, answers = get_traffic_topology_text("A", scene_graph)
        questions_all.extend(questions)
        answers_all.extend(answers)

questions_all_new = []
answers_all_new = []
for j in range(len(questions_all)):
    if "Wrong" not in answers_all[j] and "No." not in answers_all[j] and "0" not in answers_all[j]:
        questions_all_new.append(questions_all[j])
        answers_all_new.append(answers_all[j])

# language_data[data_info["timestamp"]] = {"q": questions_all_new, "a": answers_all_new}

sorted_data_info = post_process_DataInfo(copy.deepcopy(data_info))

sorted_data_info = n_n_dist_mat(sorted_data_info)

# img_anno = show_image_anno(data_info=sorted_data_info)
# mmcv.imwrite(img_anno, "./test_6.jpg")

# img_raw = show_image_raw(data_info=data_info)
# mmcv.imwrite(img_raw, os.path.join(data_path, language_path, f"vis/testbench/raw_6_{data_info['timestamp']}.jpg"))

# with open(os.path.join(data_path, language_path, f"vis/testbench/test_qa_sample.json"), "w") as outfile:
#     json.dump(language_data, outfile)

# with open(os.path.join(data_path, language_path, f"vis/testbench/DataInfo_sample.json"), "w") as outfile:
#     json.dump(data_info, outfile, cls=NumpyArrayEncoder)

generate_question(sorted_data_info)
# if __name__ == "__main__":
#     VIS_FLAG = True
#     main(VIS_FLAG)


'''
    questions_columns = {
        "问题分类": [],
        "问题": [], 
        "子问题字段1": [], 
        "子问题字段2": [], 
        "子问题字段3": [],
        "子问题字段4": [],
        "子问题字段5": [],
        "子问题字段6": [],
        "子问题字段7": [],
        "子问题字段8": [],
        "子问题字段9": [],
    }
       questions_table = pd.DataFrame(questions_columns)

    questions_prediction_template = {
        "问题分类": '预测',                         # B
        "问题": '1.1 目标A移动状态(A非交通标志)',     # C 
        "子问题字段1": '<c0>的运动状态为',          # D
        "子问题字段2": '',                          # E
        "子问题字段3": 'xxx',                          # F
        "子问题字段4": '',                          # G
        "子问题字段5": '',                          # H
        "子问题字段6": '',                          # I
        "子问题字段7": '',                          # J
        "子问题字段8": '',                          # K
        "子问题字段9": '',                          # L
    }

    questions_table = questions_table.append(questions_prediction_template, ignore_index=True)

    writer = pd.ExcelWriter('test.xlsx', engine='xlsxwriter')
    questions_table.to_excel(writer, sheet_name='Sheet1')
    # Assign workbook and worksheet
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']

    # Creation of unlocked format
    unlocked = workbook.add_format({'locked': False})
    worksheet.set_column('B:L', None, unlocked)
    worksheet.write('B3', '预测')
    worksheet.write('C3', '1.1 目标A移动状态(A非交通标志)')
    worksheet.write('D3', '<c1>的运动状态为')
    # worksheet.write('E3', '<c1>的运动状态为')

    dv = openpyxl.worksheet.datavalidation.DataValidation(type='list', formula1="静止,前进,左转,右转", allow_blank=True)

    worksheet.data_validation('E3', {'validate': 'list', 'source': ["静止", "前进", "左转", "右转"]})
    worksheet.data_validation('E2', {'validate': 'list', 'source': ["静止", "前进", "左转", "右转"]})


    # worksheet.add_data_validation(dv)
    # worksheet['E3'].data_validation = dv
    # worksheet['E2'].data_validation = dv

    # workbook.save(os.path.join(data_path, language_path, f"vis/testbench/qa_sample_wb.xlsx"))

    questions_table.to_excel(os.path.join(data_path, language_path, f"vis/testbench/qa_sample.xlsx"))
'''