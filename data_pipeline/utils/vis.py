import copy
import cv2
import numpy as np
import torch
import mmcv
from mmdet3d.core.bbox import LiDARInstance3DBoxes

# for opencv, it is BGR
color_map = {
    "<c0>": (0, 0, 255),
    "<c1>": (0, 255, 255),
    "<c2>": (255, 234, 0),
    "<c3>": (255, 0, 170),
    "<c4>": (0, 127, 255),
    "<c5>": (0, 255, 191),
    "<c6>": (255, 149, 0),
    "<c7>": (170, 0, 255),
    "<c8>": (0, 212, 255),
    "<c8>": (0, 255, 106),
    "<c9>": (255, 64, 0),
    "<c10>": (185, 185, 237),
    "<c11>": (237, 215, 185),
    "<c12>": (185, 233, 231),
    "<c13>": (237, 185, 220),
    "<c14>": (224, 237, 185),
    "<c15>": (35, 35, 143),
    "<c16>": (143, 98, 35),
    "<c17>": (35, 106, 143),
    "<c18>": (143, 35, 107),
    "<c19>": (35, 143, 79),
    "<c20>": (0, 0, 0),
    "<c21>": (115, 115, 115),
    "<c22>": (204, 204, 204),
}

def bev_to_corners(bev):
    n = bev.shape[0]

    corners = torch.stack(
        (
            0.5 * bev[:, 2] * torch.cos(bev[:, -1]) - 0.5 * bev[:, 3] * torch.sin(bev[:, -1]) + bev[:, 0],
            0.5 * bev[:, 2] * torch.sin(bev[:, -1]) + 0.5 * bev[:, 3] * torch.cos(bev[:, -1]) + bev[:, 1],
            0.5 * bev[:, 2] * torch.cos(bev[:, -1]) + 0.5 * bev[:, 3] * torch.sin(bev[:, -1]) + bev[:, 0],
            0.5 * bev[:, 2] * torch.sin(bev[:, -1]) - 0.5 * bev[:, 3] * torch.cos(bev[:, -1]) + bev[:, 1],
            -0.5 * bev[:, 2] * torch.cos(bev[:, -1]) + 0.5 * bev[:, 3] * torch.sin(bev[:, -1]) + bev[:, 0],
            -0.5 * bev[:, 2] * torch.sin(bev[:, -1]) - 0.5 * bev[:, 3] * torch.cos(bev[:, -1]) + bev[:, 1],
            -0.5 * bev[:, 2] * torch.cos(bev[:, -1]) - 0.5 * bev[:, 3] * torch.sin(bev[:, -1]) + bev[:, 0],
            -0.5 * bev[:, 2] * torch.sin(bev[:, -1]) + 0.5 * bev[:, 3] * torch.cos(bev[:, -1]) + bev[:, 1],
        )
    )
    corners = corners.reshape(4, 2, n).permute(2, 0, 1)
    return corners


def draw_bev_result(img, gt_bev_boxes, gt_ids):
    temp = copy.deepcopy(gt_bev_boxes)
    gt_bev_corners = bev_to_corners(temp)

    bev_size = 2200
    scale = 10
    # bev_size // scale : bev range in lidar/ego coord

    if img is None:
        img = np.ones((bev_size, bev_size, 3)) * 255

    # draw circle
    for i in range(bev_size // (10 * scale)):
        cv2.circle(img, (bev_size // 2, bev_size // 2), (i + 1) * 10 * scale, (125, 217, 233), 2)
        if i == 4:
            cv2.circle(img, (bev_size // 2, bev_size // 2), (i + 1) * 10 * scale, (255, 255, 255), 2)

    if gt_bev_corners is not None:
        gt_bev_buffer = copy.deepcopy(gt_bev_corners)
        # gt_bev_corners[:, :, 0] = -gt_bev_buffer[:, :, 1] * scale + bev_size // 2
        # gt_bev_corners[:, :, 1] = -gt_bev_buffer[:, :, 0] * scale + bev_size // 2
        gt_bev_corners[:, :, 0] = gt_bev_buffer[:, :, 0] * scale + bev_size // 2
        gt_bev_corners[:, :, 1] = -gt_bev_buffer[:, :, 1] * scale + bev_size // 2

        gt_color = (255, 255, 255)
        for i, corners in enumerate(gt_bev_corners):
            gt_color = color_map[gt_ids[i]]
            xmin = int(corners[:, 0].min())
            ymax = int(corners[:, 1].max())
            cv2.putText(
                img,
                gt_ids[i],
                (xmin - 70, ymax + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                gt_color,
                4,
            )

            cv2.line(
                img, (int(corners[0, 0]), int(corners[0, 1])), (int(corners[1, 0]), int(corners[1, 1])), gt_color, 4
            )
            cv2.line(
                img, (int(corners[1, 0]), int(corners[1, 1])), (int(corners[2, 0]), int(corners[2, 1])), gt_color, 4
            )
            cv2.line(
                img, (int(corners[2, 0]), int(corners[2, 1])), (int(corners[3, 0]), int(corners[3, 1])), gt_color, 4
            )
            cv2.line(
                img, (int(corners[3, 0]), int(corners[3, 1])), (int(corners[0, 0]), int(corners[0, 1])), gt_color, 4
            )

    return img


def plot_rect3d_on_img(img, num_rects, rect_corners, names, color=(0, 255, 0), thickness=1):
    """Plot the boundary lines of 3D rectangular on 2D images.

    Args:
        img (numpy.array): The numpy array of image.
        num_rects (int): Number of 3D rectangulars.
        rect_corners (numpy.array): Coordinates of the corners of 3D
            rectangulars. Should be in the shape of [num_rect, 8, 2].
        color (tuple[int]): The color to draw bboxes. Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    
    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7), (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
    for i in range(num_rects):
        if type(color) is tuple:
            color_draw = color
        elif type(color) is dict:
            color_draw = color[names[i]]
        corners = rect_corners[i].astype(np.int)
        xmin = int(corners[:, 0].min())
        ymax = int(corners[:, 1].max())
        cv2.putText(
            img,
            names[i],
            (xmin, ymax + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            color_draw,
            8,
        )
        for start, end in line_indices:
            cv2.line(
                img,
                (corners[start, 0], corners[start, 1]),
                (corners[end, 0], corners[end, 1]),
                color_draw,
                thickness,
                cv2.LINE_AA,
            )

    return img.astype(np.uint8)


def draw_lidar_bbox3d_on_img(bboxes3d, names, raw_img, lidar2img_rt, img_metas, color=(0, 255, 0), thickness=1):
    img = raw_img.copy()
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate([corners_3d.reshape(-1, 3), np.ones((num_bbox * 8, 1))], axis=-1)
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    pts_2d = pts_4d @ lidar2img_rt.T

    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    pts_2d[:, 0] = np.clip(pts_2d[:, 0], a_min=-1e5, a_max=1e5)
    pts_2d[:, 1] = np.clip(pts_2d[:, 1], a_min=-1e5, a_max=1e5)
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, names, color, thickness)


def show_image(data_info):
    gt_boxes = np.array(data_info["gt_boxes"])
    gt_boxes[:, 2] -= 0.5 * gt_boxes[:, 5]
    gt_boxes = LiDARInstance3DBoxes(gt_boxes)
    gt_names = data_info["gt_names"]
    gt_attributes = data_info["gt_attrs"]
    gt_ids = []
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
            category_count[gt_names[i]] += 1
        else:
            category_count[gt_names[i]] = 0
            gt_ids.append(gt_names[i] + "_" + gt_attributes[i] + "_" + str(category_count[gt_names[i]]))

    file_client_args = dict(
        backend="petrel",
        path_mapping=dict(
            {
                "./data/nuscenes/": "tianhao2_1424:s3://pubdata/nuScenes/",
                "data/nuscenes/": "tianhao2_1424:s3://pubdata/nuScenes/",
            }
        ),
    )

    bev_img = None
    bev_img = draw_bev_result(bev_img, copy.deepcopy(gt_boxes.bev), gt_ids)
    # bev_img = cv2.rotate(bev_img, cv2.ROTATE_90_CLOCKWISE)  # cv2.ROTATE_90_CLOCKWISE)

    big_img = []
    for cam_type in ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]:
        data_path = data_info["cams"][cam_type]["data_path"]

        if "Nuscenes" in data_path:
            data_path = data_path.replace("Nuscenes/", "")
        
        if "./data" in data_path:
            data_path = data_path.replace("./data", "/mnt/disk01")

        # img = mmcv.imread(data_path, file_client_args=file_client_args)
        img = mmcv.imread(data_path)
        resize = (1920, 1080)

        intrinsic = np.matrix(data_info["cams"][cam_type]["cam_intrinsic"])
        intrinsic = intrinsic[:3, :3]
        intrinsic[1] *= resize[1] / img.shape[0]
        intrinsic[0] *= resize[0] / img.shape[1]
        viewpad = np.eye(4)
        viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
        R = np.array(data_info["cams"][cam_type]["sensor2lidar_rotation"])
        T = np.array(data_info["cams"][cam_type]["sensor2lidar_translation"]).reshape(3, 1)
        R = np.vstack((R, np.array([[0, 0, 0]], dtype=np.float32)))
        T = np.vstack((T, np.array([1], dtype=np.float32)))
        extrinsic = np.linalg.inv(np.hstack((R, T)))
        lidar2img = np.array((viewpad @ extrinsic))

        img = mmcv.imresize(img, resize)
        gt_bbox_color = (72, 101, 241)
        img = draw_lidar_bbox3d_on_img(
            copy.deepcopy(gt_boxes),
            copy.deepcopy(gt_ids),
            img,
            lidar2img,
            None,
            color=gt_bbox_color,
            thickness=4,
        )
        big_img.append(img)

    img_part1 = cv2.resize(bev_img, (1080, 1080))
    img_part2 = np.concatenate((big_img[:3]), axis=1)
    img_part3 = np.concatenate((big_img[3:]), axis=1)
    big_img = np.concatenate((img_part2, img_part3), axis=0)
    big_img = cv2.resize(big_img, (2880, 1080))
    big_img = np.concatenate((big_img, img_part1), axis=1).astype(np.uint8)

    # mmcv.imwrite(big_img, "test.jpg")
    return big_img


def show_image_anno(data_info):
    # import pdb
    # pdb.set_trace()
    gt_boxes = np.array(data_info["sorted_gt_boxes"])
    gt_boxes[:, 2] -= 0.5 * gt_boxes[:, 5]
    gt_boxes = LiDARInstance3DBoxes(gt_boxes)
    gt_names = data_info["sorted_gt_names"]
    gt_attributes = data_info["sorted_gt_attrs"]
    gt_ids = []

    gt_cat_num = []
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

    file_client_args = dict(
        backend="petrel",
        path_mapping=dict(
            {
                "./data/nuscenes/": "tianhao2_1424:s3://pubdata/nuScenes/",
                "data/nuscenes/": "tianhao2_1424:s3://pubdata/nuScenes/",
            }
        ),
    )
    # pdb.set_trace()

    bev_img = None
    # bev_img = draw_bev_result(bev_img, copy.deepcopy(gt_boxes.bev), gt_ids)
    bev_img = draw_bev_result(bev_img, copy.deepcopy(gt_boxes.bev), copy.deepcopy(gt_cat_num))
    # bev_img = cv2.rotate(bev_img, cv2.ROTATE_90_CLOCKWISE)  # cv2.ROTATE_90_CLOCKWISE)

    big_img = []
    for cam_type in ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]:
        data_path = data_info["cams"][cam_type]["data_path"]

        if "Nuscenes" in data_path:
            data_path = data_path.replace("Nuscenes/", "")
        
        if "./data" in data_path:
            data_path = data_path.replace("./data", "/mnt/disk01")

        # img = mmcv.imread(data_path, file_client_args=file_client_args)
        img = mmcv.imread(data_path)
        resize = (1920, 1080)

        intrinsic = np.matrix(data_info["cams"][cam_type]["cam_intrinsic"])
        intrinsic = intrinsic[:3, :3]
        intrinsic[1] *= resize[1] / img.shape[0]
        intrinsic[0] *= resize[0] / img.shape[1]
        viewpad = np.eye(4)
        viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
        R = np.array(data_info["cams"][cam_type]["sensor2lidar_rotation"])
        T = np.array(data_info["cams"][cam_type]["sensor2lidar_translation"]).reshape(3, 1)
        R = np.vstack((R, np.array([[0, 0, 0]], dtype=np.float32)))
        T = np.vstack((T, np.array([1], dtype=np.float32)))
        extrinsic = np.linalg.inv(np.hstack((R, T)))
        lidar2img = np.array((viewpad @ extrinsic))

        # pdb.set_trace()

        img = mmcv.imresize(img, resize)
        gt_bbox_color = (72, 101, 241)
        img = draw_lidar_bbox3d_on_img(
            copy.deepcopy(gt_boxes),
            copy.deepcopy(gt_cat_num),
            img,
            lidar2img,
            None,
            color=color_map,
            thickness=4,
        )
        big_img.append(img)

    img_part1 = cv2.resize(bev_img, (1080, 1080))
    img_part2 = np.concatenate((big_img[:3]), axis=1)
    img_part3 = np.concatenate((big_img[3:]), axis=1)
    big_img = np.concatenate((img_part2, img_part3), axis=0)
    big_img = cv2.resize(big_img, (2880, 1080))
    big_img = np.concatenate((big_img, img_part1), axis=1).astype(np.uint8)

    # mmcv.imwrite(big_img, "test.jpg")
    return big_img



def show_image_raw(data_info):
    import pdb
    # pdb.set_trace()

    # pdb.set_trace()

    big_img = []
    for cam_type in ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]:
        data_path = data_info["cams"][cam_type]["data_path"]

        if "Nuscenes" in data_path:
            data_path = data_path.replace("Nuscenes/", "")
        
        if "./data" in data_path:
            data_path = data_path.replace("./data", "/mnt/disk01")

        # img = mmcv.imread(data_path, file_client_args=file_client_args)
        img = mmcv.imread(data_path)
        resize = (1920, 1080)

        big_img.append(img)

    # img_part1 = cv2.resize(bev_img, (1080, 1080))
    img_part2 = np.concatenate((big_img[:3]), axis=1)
    img_part3 = np.concatenate((big_img[3:]), axis=1)
    big_img = np.concatenate((img_part2, img_part3), axis=0)
    big_img = cv2.resize(big_img, (2880, 1080))
    # big_img = np.concatenate((big_img, img_part1), axis=1).astype(np.uint8)

    # mmcv.imwrite(big_img, "test.jpg")
    return big_img
