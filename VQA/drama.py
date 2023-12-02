import json
import os
import glob
from pathlib import Path
import pickle as pkl

import torch

import cv2
import numpy as np

from hydra.utils import to_absolute_path

from line_profiler import LineProfiler

def profile(func):
    def wrapper(*args, **kwargs):
        profiler = LineProfiler()
        profiler.add_function(func)
        profiler.runcall(func, *args, **kwargs)
        profiler.print_stats()
    return wrapper

cv2.setNumThreads(0)


class DRAMA(torch.utils.data.Dataset):
    #@profile
    def __init__(
        self,
        data_dir,
        split="train",
        **kwargs,
    ):
        super().__init__()

        path_annotations = to_absolute_path('data/drama/integrated_output_v2.json')
        with open(path_annotations, 'r') as f:
            self.annotations = json.load(f)

        self.annotations_dict = {}
        for annotation in self.annotations:
            img_path = annotation['s3_fileUrl'].split('combined/')[1]
            self.annotations_dict[img_path] = annotation
        
        data_dir = to_absolute_path(data_dir)
        self.data_dir = data_dir + "/" + split

        # search for .pkl files in data_dir
        self.sample_paths = glob.glob(self.data_dir + "/*.pkl")[:50]

        # load all samples
        self.samples = []
        for sample_path in self.sample_paths:
            with open(sample_path, 'rb') as f:
                self.samples.append(pkl.load(f))
                # self.samples[0]['img_path']: '/data/02/kdi/internal_datasets/drama/combined/2020-0228-102841/035738/frame_035738.png'
                # get frame number from path
                frame_num = int(os.path.splitext(os.path.basename(self.samples[-1]['img_path']))[0].split('_')[-1])
                img_path = self.samples[-1]['img_path'].split('combined/')[1]
                # add annotation to sample
                self.samples[-1]['ann'] = self.annotations_dict[img_path]


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        img = self.samples[index]['img']
        caption = self.samples[index]['ann']['Caption']
        suggestion_ego = self.samples[index]['ann']['Suggestions (to the ego car)']
        
        return img, caption, suggestion_ego

if __name__ == "__main__":
    from hydra import compose, initialize

    # dataset = cProfile.run('HDD(cfg, "data/hdd/processed/info_HDD_3hz_goals.pkl", split="val",subsample=cfg.dataset.subsample)')
    dataset = DRAMA(
        'data/drama/processed', 
        split="test",
    )


    # import time
    # time.sleep(900)

    from transformers import ViltProcessor, ViltForQuestionAnswering
    from transformers import BlipProcessor, BlipForQuestionAnswering
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
    from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

    import requests
    from PIL import Image

    model_names = [
        "Salesforce/blip-vqa-base", 
        "Salesforce/blip2-flan-t5-xl",
        "Salesforce/blip2-flan-t5-xxl",
        "Salesforce/blip2-opt-2.7b",
        "Salesforce/blip2-opt-2.7b-coco",
        "Salesforce/blip2-opt-6.7b", 
        ]

    processors = []
    models = []
    for name in model_names:
        if "blip2" in name:
            processors.append(Blip2Processor.from_pretrained(name))
            models.append(Blip2ForConditionalGeneration.from_pretrained(name))
        else:
            processors.append(BlipProcessor.from_pretrained(name))
            models.append(BlipForQuestionAnswering.from_pretrained(name))


    # loop over dataset
    for i in range(len(dataset)):
        img, caption, suggestion_ego = dataset[i]
        # copy img

        # print(sample)

        # prepare image + question
        # img: np array: channel, height, width
        # to height, width, channel
        img = np.transpose(img, (1, 2, 0))
        img = img[:, :, ::-1]  # rgb to bgr
        img2 = img.copy()
        # to PIL image
        img = Image.fromarray(img)
        # text = "Why should the driver of the own vehicle brake?"
        text = "Content: The image is taken from a car driving around. Question: What should the driver do in this situation? Answer:"

        black_box = np.zeros((350, img2.shape[1], 3), dtype=np.uint8)
        # add text to black box
        cv2.putText(black_box, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


        # add prediction and ground truth to black box
        cv2.putText(black_box, "Captions: " + caption, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(black_box, "Ground truth: " + suggestion_ego, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        j = 0
        for model, processor in zip(models, processors):
            inputs = processor(img, text, return_tensors="pt")#.to("cuda")
            out = model.generate(**inputs)
            decoded_str = ''
            for o in out:
                print(processor.decode(o, skip_special_tokens=True))
                decoded_str += processor.decode(o, skip_special_tokens=True)
            cv2.putText(black_box, f"{model_names[j]}: " + decoded_str, (10, 140+j*40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # if i == 0:
            #     breakpoint()
            j += 1

        # save images
        img2 = np.concatenate((img2, black_box), axis=0)
        img2 = Image.fromarray(img2)

        folder_name = 'img_blip2/content_what_do_driver'
        Path(folder_name).mkdir(parents=True, exist_ok=True)
        
        img2.save(f'{folder_name}/test_{i}.png')

        # breakpoint()