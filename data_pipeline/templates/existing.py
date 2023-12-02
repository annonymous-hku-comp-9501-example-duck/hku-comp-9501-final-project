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


def get_existence_text(type, scene_graph):
    if type.upper() == "A":
        return existing_type_a_text_generation(scene_graph)
    elif type.upper() == "B":
        return existing_type_b_text_generation(scene_graph)
    elif type.upper() == "C":
        return existing_type_c_text_generation(scene_graph)
    elif type.upper() == "D":
        return existing_type_d_text_generation(scene_graph)
    else:
        raise NotImplementedError


def existing_type_a_text_generation(scene_graph):
    template_list = [
        "Are there any <A><O>s?",
        # "Are any <A><O>s visible?",
    ]

    questions_all = []
    answers_all = []
    for category in category_attr_mapping:
        for attribute in category_attr_mapping[category]:
            answer = get_answer_type_a(scene_graph, category, attribute)
            questions = [template.replace("<A><O>", f"{attribute} {category}") for template in template_list]
            answers = [answer for j in range(len(questions))]
            questions_all.extend(questions)
            answers_all.extend(answers)
    return questions_all, answers_all


def existing_type_b_text_generation(scene_graph):
    template_list = [
        "Are there any other <O2>s that in the same status as the <A><O>?",
        # "Is there another <O2> that has the same status as the <A><O>?",
        # "Are there any other <O2>s of the same status as the <A><O>?",
        # "Is there another <O2> of the same status as the <A><O>?",
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
            questions = [
                template.replace("<A><O>", f"{attribute} {category}").replace("<O2>", f"category")
                for template in template_list
            ]
            answers = [answer for j in range(len(questions))]
            questions_all.extend(questions)
            answers_all.extend(answers)
    return questions_all, answers_all


def existing_type_c_text_generation(scene_graph):
    template_list = [
        "Are there any <A2><O2>s to the <R> of the <A><O>?",
        # "There is a <A><O>; are there any <A2><O2>s to the <R> of it?",
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


def existing_type_d_text_generation(scene_graph):
    template_list = [
        "Are there any other <O3>s that in the same status as the <A2><O2> [that is] to the <R> of the <A><O>?",
        # "Is there another <O3> that has the same status as the <A2><O2> [that is] to the <R> of the <A><O>?",
        # "Are there any other <O3>s of the same status as the <A2><O2> [that is] to the <R> of the <A><O>?",
        # "Is there another <O3> of the same status as the <A2><O2> [that is] to the <R> of the <A><O>?",
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
                            .replace("<A2><O2>", f"{category2}")
                            .replace("<R>", relationship)
                            .replace("<O3>", category2)
                            for template in template_list
                        ]
                        answers = [answer for j in range(len(questions))]
                        questions_all.extend(questions)
                        answers_all.extend(answers)
    return questions_all, answers_all


def get_answer_type_a(scene_graph, category, attribute):
    for object in scene_graph.objects:
        if object.name == category and object.attr == attribute:
            return "Yes."
    return "No."


def get_answer_type_b(scene_graph, category, attribute):
    count = 0
    for i in range(len(scene_graph.objects)):
        if scene_graph.objects[i].attr == attribute and scene_graph.objects[i].name != category:
            count += 1
    if count == 0:
        return f"Wrong question ! No {attribute} {category}."
    elif count == 1:
        return f"No."
    else:
        return "Yes."


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
    elif len(sorted_objects[0].relationships[relationship]) <= 0:
        return "No."
    else:
        closest_object = sorted_objects[0]
        for i in range(len(closest_object.relationships[relationship])):
            if (
                closest_object.relationships[relationship][i].name == category2
                and closest_object.relationships[relationship][i].attr == attribute2
            ):
                return "Yes."
    return "No."


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
    if len(sorted_objects) == 0:
        return "Wrong question !"
    elif len(sorted_objects[0].relationships[relationship]) <= 0:
        return "No."
    else:
        closest_object = sorted_objects[0]
        for i in range(len(closest_object.relationships[relationship])):
            if (
                closest_object.relationships[relationship][i].name == category2
                and closest_object.relationships[relationship][i].attr == attribute2
            ):
                return "Yes."
    return "No."
