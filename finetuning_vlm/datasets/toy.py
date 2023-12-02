from datasets import load_dataset 

import torch
from torch.utils.data import Dataset, DataLoader




class ImageCaptioningDataset(Dataset):
    def __init__(self, processor):
        self.dataset = load_dataset("ybelkada/football-dataset", split="train")
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = item["text"]
        encoding["instruction"] = "Generate caption:"
        return encoding, item["image"]