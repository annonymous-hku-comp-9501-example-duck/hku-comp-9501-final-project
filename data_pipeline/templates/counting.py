import numpy as np


category_attr_mapping = {
    "car": ["moving", "parked"],
    "truck": ["parked", "moving"],
    "construction_vehicle": ["parked", "moving"],
    "bus": ["stopped", "moving"],
    "trailer": ["moving", "parked"],
    "barrier": [""],
    "motorcycle": ["with_rider", "without_rider"],
    "bicycle": ["without_rider", "with_rider"],
    "pedestrian": ["standing", "moving"],
    "traffic_cone": [""],
}

relationship_list = ["front", "front left", "front right", "back left", "back right", "back"]


def get_counting_text(type, scene_graph):
    if type.upper() == "A":
        return counting_type_a_text_generation(scene_graph)
    elif type.upper() == "B":
        return counting_type_b_text_generation(scene_graph)
    elif type.upper() == "C":
        return counting_type_c_text_generation(scene_graph)
    elif type.upper() == "D":
        return counting_type_d_text_generation(scene_graph)
    else:
        raise NotImplementedError


def counting_type_a_text_generation(scene_graph):
    template_list = [
        "How many <A><O>s are there?",
        # "What number of <A><O>s are there?",
    ]

    questions_all = []
    answers_all = []
    for category in category_attr_mapping:
        for attribute in category_attr_mapping[category]:
            answer = get_answer_type_a(
                scene_graph,
                category,
                attribute,
            )
            questions = [template.replace("<A><O>", f"{attribute} {category}") for template in template_list]
            answers = [answer for j in range(len(questions))]
            questions_all.extend(questions)
            answers_all.extend(answers)
    return questions_all, answers_all


def counting_type_b_text_generation(scene_graph):
    template_list = [
        "How many other things are in the same status as the <A><O>?",
        # "What number of other things are in the same status as the <A><O>?",
        # "How many other things are there of the same status as the <A><O>?",
        # "What number of other things are there of the same status as the <A><O>?",
    ]
    questions_all = []
    answers_all = []
    for category in category_attr_mapping:
        for attribute in category_attr_mapping[category]:
            answer = get_answer_type_b(
                scene_graph,
                category,
                attribute,
            )
            questions = [template.replace("<A><O>", f"{attribute} {category}") for template in template_list]
            answers = [answer for j in range(len(questions))]
            questions_all.extend(questions)
            answers_all.extend(answers)
    return questions_all, answers_all


def counting_type_c_text_generation(scene_graph):
    template_list = [
        "What number of <A2><O2>s are to the <R> of the <A><O>?",
        # "How many <A2><O2>s are to the <R> of the <A><O>?",
        # "There is a <A><O>; how many <A2><O2>s are to the <R> of it?",
        # "There is a <A><O>; what number of <A2><O2>s are to the <R> of it?",
    ]
    questions_all = []
    answers_all = []
    for category in category_attr_mapping:
        for attribute in category_attr_mapping[category]:
            for relationship in relationship_list:
                for category2 in category_attr_mapping:
                    for attribute2 in category_attr_mapping[category]:
                        answer = get_answer_type_c(
                            scene_graph,
                            category,
                            relationship,
                            attribute,
                            category2,
                            attribute2,
                        )
                        questions = [
                            template.replace("<A><O>", f"{attribute} {category}")
                            .replace("<A2><O2>", f"{attribute2} {category2}")
                            .replace("<R>", relationship)
                            for template in template_list
                        ]
                        answers = [answer for j in range(len(questions))]
                        questions_all.extend(questions)
                        answers_all.extend(answers)
    return questions_all, answers_all


def counting_type_d_text_generation(scene_graph):
    template_list = [
        "How many other <O3>s in the same status as the <A2><O2> [that is] to the <R> of the <A><O>?",
        # "How many other <O3>s are in the same status as the <A2><O2> [that is] to the <R> of the <A><O>?",
        # "What number of other <O3>s in the same status as the <A2><O2> [that is] to the <R> of the <A><O>?",
    ]

    questions_all = []
    answers_all = []
    for category in category_attr_mapping:
        for attribute in category_attr_mapping[category]:
            for relationship in relationship_list:
                for category2 in category_attr_mapping:
                    for attribute2 in category_attr_mapping[category2]:
                        answer = get_answer_type_d(
                            scene_graph,
                            category,
                            relationship,
                            attribute,
                            category2,
                            attribute2,
                        )
                        questions = [
                            template.replace("<A><O>", f"{attribute} {category}")
                            .replace("<A2><O2>", f"{attribute2} {category2}")
                            .replace("<O3>", f"{category2}")
                            .replace("<R>", relationship)
                            for template in template_list
                        ]
                        answers = [answer for j in range(len(questions))]
                        questions_all.extend(questions)
                        answers_all.extend(answers)
    return questions_all, answers_all


def get_answer_type_a(scene_graph, category, attribute):
    count = 0
    for i in range(len(scene_graph.objects)):
        if scene_graph.objects[i].attr == attribute and scene_graph.objects[i].name == category:
            count += 1
    return str(count)


def get_answer_type_b(scene_graph, category, attribute):
    count = 0
    for i in range(len(scene_graph.objects)):
        if scene_graph.objects[i].attr == attribute and scene_graph.objects[i].name != category:
            count += 1
    return str(count)


def get_answer_type_c(
    scene_graph,
    category,
    relationship,
    attribute,
    category2,
    attribute2,
):
    sorted_objects = sorted(scene_graph.objects, key=lambda x: np.linalg.norm(x.location))
    sorted_objects = list(filter(lambda x: (x.name == category and x.attr == attribute), sorted_objects))
    if len(sorted_objects) == 0:
        return "Wrong question !"
    closest_object = sorted_objects[0]
    if closest_object is None or len(closest_object.relationships[relationship]) <= 0:
        return "Wrong question !"
    else:
        count = 0
        for i in range(len(closest_object.relationships[relationship])):
            target_object = closest_object.relationships[relationship][i]
            if target_object.name == category2 and target_object.attr == attribute2:
                count += 1
    return str(count)


def get_answer_type_d(
    scene_graph,
    category,
    relationship,
    attribute,
    category2,
    attribute2,
):
    sorted_objects = sorted(scene_graph.objects, key=lambda x: np.linalg.norm(x.location))
    sorted_objects = list(filter(lambda x: (x.name == category and x.attr == attribute), sorted_objects))
    if len(sorted_objects) == 0 or len(sorted_objects[0].relationships[relationship]) <= 0:
        return "Wrong question !"
    else:
        closest_object = sorted_objects[0]
        sorted_objects2 = sorted(closest_object.relationships[relationship], key=lambda x: np.linalg.norm(x.location))
        sorted_objects2 = list(filter(lambda x: (x.name == category2 and x.attr == attribute2), sorted_objects2))
        if len(sorted_objects2) > 0:
            id = sorted_objects2[0].category_id
        else:
            return "Wrong question !"

        count = 0
        for i in range(len(scene_graph.objects)):
            if scene_graph.objects[i].name == category2 and scene_graph.objects[i].attr == attribute2:
                if scene_graph.objects[i].category_id == id:
                    continue
                count += 1
    return str(count)
