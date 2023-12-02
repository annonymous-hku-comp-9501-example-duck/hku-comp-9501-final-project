import json

# load json
# path = 'data/DriveLM_v2/QA_pair_v3_val.json'
path = 'data/DriveLM_v2/subset_B_train.json'
with open(path, 'r') as f:
    data = json.load(f)

# {
#     "b789de07180846cc972118ee6d1fb027": {
#         "scene_description": "",
#         "key_frame": {
#             "1537295990612404": {
#                 "Perception": {
#                     "q": [
#                         "Q: What are important objects to the back right of the ego car?",
#                         "Q: What is the status of the truck that is to the back right of the ego car?",
#                     ],
#                     "a": [
#                         "A: There are one truck, one car to the back right of the ego car.",
#                         "A: One truck is parked.",ar to the back left of the ego car, and a parked car to the front right of the ego car. The ids of these objects are <c1,CAM_BACK,1190.5,826.5>, <c2,CAM_BACK_RIGHT,1959.0,883.0>, and <c3,CAM_FRONT_RIGHT,2155.0,321.0>.",
#                         "A: No, there are no traffic elements in the front view."
#                     ],
#                     "description": {
#                         "<c1,CAM_BACK,1190.5,826.5>": "<c1,CAM_BACK,1190.5,826.5> is a moving truck to the back of the ego car.",
#                         "<c2,CAM_BACK_RIGHT,1959.0,883.0>": "<c2,CAM_BACK_RIGHT,1959.0,883.0> is a parked car to the back left of the ego car.",
#                         "<c3,CAM_FRONT_RIGHT,2155.0,321.0>": "<c3,CAM_FRONT_RIGHT,2155.0,321.0> is a parked car to the front right of the ego car."
#                     }
#                 },
#                 "Prediction and Planning": {
#                     "q": [
#                         "Q: Is <c1,CAM_BACK,1190.5,826.5> a traffic sign or a road barrier?",
#                         "Q: What is the movement of object <c1,CAM_BACK,1190.5,826.5>?",
#                     ],
#                     "a": [
#                         "A: No.",
#                         "A: Moving.",
#                     ]
#                 }
#             },

# get all scene descriptions
for key in data.keys():
    for tag in data[key]['key_frame'].keys():
        data[key]['key_frame'][tag]['Perception']['description_visual'] = {}
        for i in range(len(data[key]['key_frame'][tag]['Prediction and Planning']['q'])):
            question = data[key]['key_frame'][tag]['Prediction and Planning']['q'][i]
            if "Q: How would you visually describe" in question or "look like visually?" in question or "> look like?" in question or "Q: What is the visual description of <" in question:
                answer = data[key]['key_frame'][tag]['Prediction and Planning']['a'][i]
                tag2 = question.split("<")[1].split(">")[0]
                tag2 = "<" + tag2 + ">"
                answer = answer.replace("A: ", "")
                answer = answer.replace("It is a ", "A ")
                answer = answer.replace("It is an ", "An ")
                answer = answer.replace(f"The visual description of {tag2} is a ", "A ")
                answer = answer.replace(f"The visual description of {tag2} is an ", "An ")
                answer = answer.replace(f"The visual description of {tag2} is ", "")
                answer = answer.replace(f"The visual description of {tag2} can be described as ", "A ")
                answer = answer.replace(f"{tag2} is a ", "A ")
                answer = answer.replace(f"{tag2} can be visually described as ", "A ")
                answer = answer.replace(f"It can be visually described as a ", "A ")
                answer = answer.replace(f"It appears as a ", "A ")
                answer = answer.replace(f"The ", "A ")

                data[key]['key_frame'][tag]['Perception']['description_visual'][tag2] = answer


# save json
# path_save = 'data/DriveLM_v2/QA_pair_v3_val.json'
with open(path, 'w') as f:
    json.dump(data, f, indent=4)
                

        