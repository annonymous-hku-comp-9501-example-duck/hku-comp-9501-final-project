import pickle
import numpy as np
from tqdm import tqdm


NameMapping = {
    'movable_object.barrier': 'barrier',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck'
}
DefaultAttribute = {
    'car': 'vehicle.parked',
    'pedestrian': 'pedestrian.moving',
    'trailer': 'vehicle.parked',
    'truck': 'vehicle.parked',
    'bus': 'vehicle.moving',
    'motorcycle': 'cycle.without_rider',
    'construction_vehicle': 'vehicle.parked',
    'bicycle': 'cycle.without_rider',
    'barrier': '',
    'traffic_cone': '',
}
AttrMapping = {
    'cycle.with_rider': 0,
    'cycle.without_rider': 1,
    'pedestrian.moving': 2,
    'pedestrian.standing': 3,
    'pedestrian.sitting_lying_down': 4,
    'vehicle.moving': 5,
    'vehicle.parked': 6,
    'vehicle.stopped': 7,
}
AttrMapping_rev = [
    'cycle.with_rider',
    'cycle.without_rider',
    'pedestrian.moving',
    'pedestrian.standing',
    'pedestrian.sitting_lying_down',
    'vehicle.moving',
    'vehicle.parked',
    'vehicle.stopped',
]
CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
            'barrier')


def convert_names_to_labels(names):
    labels = []
    for name in names:
        try:
            label = CLASSES.index(name)
        except:
            label = -1
        labels.append(label)
    return np.array(labels)


def main(results):
    """Convert the results to the standard format.

    Args:
        results (list[dict]): Testing results of the dataset.
        jsonfile_prefix (str): The prefix of the output jsonfile.
            You can specify the output directory/filename by
            modifying the jsonfile_prefix. Default: None.

    Returns:
        str: Path of the output json file.
    """
    mapped_class_names = CLASSES

    print('Start to convert detection format...')
    for data_info in tqdm(results):
        boxes = data_info['gt_boxes']
        velocity = data_info['gt_velocity']
        names = data_info['gt_names']
        labels = convert_names_to_labels(names)

        boxes_new = []
        velocity_new = []
        names_new = []
        attrs = []
        for i in range(len(labels)):
            if labels[i] == -1:
                continue
            name = mapped_class_names[labels[i]]
            if np.sqrt(velocity[i, 0]**2 + velocity[i, 1]**2) > 0.2:
                if name in [
                        'car',
                        'construction_vehicle',
                        'bus',
                        'truck',
                        'trailer',
                ]:
                    attr = 'vehicle.moving'
                elif name in ['bicycle', 'motorcycle']:
                    attr = 'cycle.with_rider'
                else:
                    attr = DefaultAttribute[name]
            else:
                if name in ['pedestrian']:
                    attr = 'pedestrian.standing'
                elif name in ['bus']:
                    attr = 'vehicle.stopped'
                else:
                    attr = DefaultAttribute[name]
            boxes_new.append(boxes[i])
            velocity_new.append(velocity[i])
            names_new.append(names[i])
            attrs.append(attr.split('.')[-1])
        data_info['gt_attrs'] = attrs
        data_info['gt_boxes'] = np.array(boxes_new)
        data_info['gt_velocity'] = np.array(velocity_new)
        data_info['gt_names'] = np.array(names_new)
    
    with open('/mnt/disk01/nuscenes/nuscenes_infos_temporal_val_xjw.pkl', 'wb') as pickle_file:
        pickle.dump(results, pickle_file)
    return


if __name__ == "__main__":
    results = pickle.load(open('/mnt/disk01/nuscenes/nuscenes_infos_temporal_val.pkl', 'rb'))['infos']
    main(results)
