import numpy as np


NameMapping = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}
DefaultAttribute = {
    "car": "vehicle.parked",
    "pedestrian": "pedestrian.moving",
    "trailer": "vehicle.parked",
    "truck": "vehicle.parked",
    "bus": "vehicle.moving",
    "motorcycle": "cycle.without_rider",
    "construction_vehicle": "vehicle.parked",
    "bicycle": "cycle.without_rider",
    "barrier": "",
    "traffic_cone": "",
}
AttrMapping = {
    "cycle.with_rider": 0,
    "cycle.without_rider": 1,
    "pedestrian.moving": 2,
    "pedestrian.standing": 3,
    "pedestrian.sitting_lying_down": 4,
    "vehicle.moving": 5,
    "vehicle.parked": 6,
    "vehicle.stopped": 7,
}
AttrMapping_rev = [
    "cycle.with_rider",
    "cycle.without_rider",
    "pedestrian.moving",
    "pedestrian.standing",
    "pedestrian.sitting_lying_down",
    "vehicle.moving",
    "vehicle.parked",
    "vehicle.stopped",
]


class Object_Info(object):
    def __init__(self, category_id, name, location, velocity, attribute=None):
        self.category_id = category_id
        self.name = name
        self.location = location
        self.velocity = velocity
        self.attr = attribute

        self.relationships = {
            "front": [],
            "front left": [],
            "front right": [],
            "back": [],
            "back left": [],
            "back right": [],
        }

    def show_relative_position(self):
        for loc in self.relationships:
            for i in range(len(self.relationships[loc])):
                print(loc, self.relationships[loc][i].name, self.relationships[loc][i].category_id)


class Scene_Graph(object):
    def __init__(self, timestamp, objects_info, traffic_info):
        self.timestamp = timestamp
        # self.velocity_ego = velocity_ego
        self.names = objects_info["names"]
        self.locations = objects_info["locations"]
        self.velocities = objects_info["velocities"]
        self.attributes = objects_info["attributes"]

        if traffic_info is None:
            self.lane_centerline = None
            self.traffic_element = None
            self.topology_lclc = None
            self.topology_lcte = None
        else:
            self.lane_centerline = traffic_info["lane_centerline"]
            self.traffic_element = traffic_info["traffic_element"]
            self.topology_lclc = traffic_info["topology_lclc"]
            self.topology_lcte = traffic_info["topology_lcte"]

        self.category_count = {
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

        self.construct_object_infos()
        self.construct_traffic_topology_infos()
        self.compute_relative_position()

    def construct_object_infos(self):
        self.objects = []
        for i in range(len(self.names)):
            if self.names[i] in self.category_count.keys():
                object_info = Object_Info(
                    self.category_count[self.names[i]],
                    self.names[i],
                    self.locations[i, :2],
                    self.velocities[i, :2],
                    self.attributes[i],
                )
                self.category_count[self.names[i]] += 1
                self.objects.append(object_info)
            else:
                object_info = Object_Info(
                    0,
                    self.names[i],
                    self.locations[i, :2],
                    self.velocities[i, :2],
                    self.attributes[i],
                )
                self.category_count[self.names[i]] = 0
                # self.category_count[self.names[i]] += 1
                self.objects.append(object_info)

    def construct_traffic_topology_infos(self):
        pass

    def compute_relative_position(self):
        locations1 = self.locations.reshape(-1, 1, 2)
        locations2 = self.locations.reshape(1, -1, 2)
        angle = (
            np.arccos(-(locations1 - locations2)[:, :, 1] / (np.linalg.norm((locations1 - locations2), axis=-1)))
            * ((locations1 - locations2) / abs(locations1 - locations2))[:, :, 0]
        )
        angle = angle / np.pi * 180
        for i in range(len(self.names)):
            for j in range(len(self.names)):
                if i != j:
                    if angle[i, j] <= 30 and angle[i, j] > -30:
                        self.objects[i].relationships["front"].append(self.objects[j])
                    elif angle[i, j] <= 90 and angle[i, j] > 30:
                        self.objects[i].relationships["front left"].append(self.objects[j])
                    elif angle[i, j] <= -30 and angle[i, j] > -90:
                        self.objects[i].relationships["front right"].append(self.objects[j])
                    elif angle[i, j] <= 150 and angle[i, j] > 90:
                        self.objects[i].relationships["back left"].append(self.objects[j])
                    elif angle[i, j] <= -90 and angle[i, j] > -150:
                        self.objects[i].relationships["back right"].append(self.objects[j])
                    elif angle[i, j] <= -150 or angle[i, j] > 150:
                        self.objects[i].relationships["back"].append(self.objects[j])
                    else:
                        raise NotImplementedError

    def compute_position(self):
        angle = np.arccos((-self.locations[:, 1]) / (np.linalg.norm((self.locations), axis=-1)))
        angle = angle / np.pi * 180
        print(angle)
        for i in range(len(self.names)):
            if angle[i] <= 30 and angle[i] > -30:
                print("front", self.objects[i].name, self.objects[i].category_id)
            elif angle[i] <= 90 and angle[i] > 30:
                print("front left", self.objects[i].name, self.objects[i].category_id)
            elif angle[i] <= -30 and angle[i] > -90:
                print("front right", self.objects[i].name, self.objects[i].category_id)
            elif angle[i] <= 150 and angle[i] > 90:
                print("back left", self.objects[i].name, self.objects[i].category_id)
            elif angle[i] <= -90 and angle[i] > -150:
                print("back right", self.objects[i].name, self.objects[i].category_id)
            elif angle[i] <= -150 or angle[i] > 150:
                print("back", self.objects[i].name, self.objects[i].category_id)


def get_scene_graph(timestamp, objects_info, traffic_info):
    return Scene_Graph(timestamp, objects_info, traffic_info)
