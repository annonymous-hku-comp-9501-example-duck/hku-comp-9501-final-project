import pickle

category_attr_mapping = {
    "traffic light": ["unknown", "red", "green", "yellow"],
    "road sign": [
        "go_straight",
        "turn left",
        "turn right",
        "no left turn",
        "no right turn",
        "u turn",
        "no u turn",
        "slight left",
        "slight right",
    ],
}

category_label_mapping = {"traffic light": 1, "road sign": 2}

TRAFFIC_ELEMENT_ATTRIBUTE = {
    0: "unknown",
    1: "red",
    2: "green",
    3: "yellow",
    4: "go straight",
    5: "turn left",
    6: "turn right",
    7: "no left turn",
    8: "no right turn",
    9: "u turn",
    10: "no u turn",
    11: "slight left",
    12: "slight right",
}

template_list = [
    "Is there any lanes on my left ?",
    "Is there any lanes on my right ?",
    "Is there a <A> path ?",
    "Which <> show I pay attention",
]


def get_traffic_topology_text(type, scene_graph):
    # A/B/C: traffic elements; xx/xx/xx: lane; xx/xx/xx: topology
    if type.upper() == "A":  # counting of traffic elements.
        return topology_type_a_text_generation(scene_graph)
    elif type.upper() == "B":  # content of traffic elements.
        return topology_type_b_text_generation(scene_graph)  # what are the traffic lights / road sign ?
    elif type.upper() == "C":  # relative position of traffic elements.
        return topology_type_c_text_generation(scene_graph)
    elif type.upper() == "D":  # lane / centerline
        return topology_type_d_text_generation(scene_graph)
    elif type.upper() == "E":  # topology
        return topology_type_e_text_generation(scene_graph)
    elif type.upper() == "F":  # logicical reasoning
        return topology_type_f_text_generation(scene_graph)


def topology_type_a_text_generation(scene_graph):
    template_list = [
        "How many <O>s are there?",
        "What number of <O>s are there?",
    ]

    questions_all = []
    answers_all = []
    for category in category_attr_mapping:
        answer = get_answer_type_a(
            scene_graph,
            category,
        )
        questions = [template.replace("<O>", f"{category}") for template in template_list]
        answers = [answer for j in range(len(questions))]
        questions_all.extend(questions)
        answers_all.extend(answers)
    return questions_all, answers_all


def topology_type_b_text_generation(scene_graph):
    template_list = ["What are the status of the <O>s ?"]

    questions_all = []
    answers_all = []
    for category in category_attr_mapping:
        answer = get_answer_type_b(scene_graph, category)
        questions = [template.replace("<O>", f"{category}") for template in template_list]
        answers = [answer for j in range(len(questions))]
        questions_all.extend(questions)
        answers_all.extend(answers)
    return questions_all, answers_all


def topology_type_c_text_generation(scene_graph):
    pass


def topology_type_d_text_generation(scene_graph):
    template_list = [
        "How many lane centerlines are there?",
        "What number of lane centerlines are there?",
    ]
    answer = get_answer_type_b(scene_graph)
    questions = [template for template in template_list]
    answers = [answer for j in range(len(questions))]
    return questions, answers


def topology_type_e_text_generation(scene_graph):
    template_list = [
        "How many intersected or connected lane centerlines",
    ]
    answer = get_answer_type_b(scene_graph)
    questions = [template for template in template_list]
    answers = [answer for j in range(len(questions))]
    return questions, answers


def topology_type_f_text_generation(scene_graph):
    pass


def get_answer_type_a(scene_graph, category):
    count = 0
    for i in range(len(scene_graph.traffic_element)):
        if scene_graph.traffic_element[i]["category"] == category_label_mapping[category]:
            count += 1
    return str(count)


def get_answer_type_b(scene_graph, category):
    attrs = {}
    for i in range(len(scene_graph.traffic_element)):
        if scene_graph.traffic_element[i]["category"] == category_label_mapping[category]:
            attr = TRAFFIC_ELEMENT_ATTRIBUTE[scene_graph.traffic_element[i]["attribute"]]
            if attr not in attrs:
                attrs[attr] = 1
            else:
                attrs[attr] += 1
    return "There are " + [str(attrs[attr]) + f"{attr}" for attr in attrs]


def get_answer_type_c(scene_graph):
    pass


def get_answer_type_d(scene_graph):
    return str(len(scene_graph.lane_centerline))


def get_answer_type_e(scene_graph):
    count = 0
    for i in range(len(scene_graph.lane_centerline)):
        if scene_graph.lane_centerline[i]["is_intersection_or_connector"]:
            count += 1
    return count


def get_answer_type_f(scene_graph):
    count = 0
    for i in range(len(scene_graph.lane_centerline)):
        if scene_graph.lane_centerline[i]["is_intersection_or_connector"]:
            count += 1
    return count
