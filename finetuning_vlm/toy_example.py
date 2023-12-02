from datasets import load_dataset 
# import plt
import matplotlib.pyplot as plt
import torch

dataset = load_dataset("ybelkada/football-dataset", split="train")


dataset[0]["text"]

from torch.utils.data import Dataset, DataLoader

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
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

from PIL import Image

def collate_fn(batch):
    # pad the input_ids and attention_mask
    processed_batch = {}
    encoding = [example[0] for example in batch]
    images = [example[1] for example in batch]
    for key in encoding[0].keys():
        if key != "text" and key != "instruction":
            processed_batch[key] = torch.stack([example[key] for example in encoding])
        else:
            text_inputs = processor.tokenizer(
                [example["text"] for example in encoding], padding=True, return_tensors="pt"
            )
            processed_batch[f"input_ids_{key}"] = text_inputs["input_ids"]
            processed_batch[f"attention_mask_{key}"] = text_inputs["attention_mask"]

    # PIL images are converted to torch tensors using torchvision.transforms.functional.pil_to_tensor and resize to 840x480
    import torchvision.transforms.functional as F
    img_tensors = [F.resize(F.pil_to_tensor(img), (480, 840)) for img in images]
    processed_batch["pixel_values_pil"] = torch.stack(img_tensors)
    
    return processed_batch


from transformers import AutoProcessor, Blip2ForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("ybelkada/blip2-opt-2.7b-fp16-sharded", device_map="auto", load_in_8bit=True)

from peft import LoraConfig, get_peft_model

# Let's define the LoraConfig
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj"]
)

model = get_peft_model(model, config)
model.print_trainable_parameters()


train_dataset = ImageCaptioningDataset(dataset, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=3, collate_fn=collate_fn)


import torch

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

device = "cuda" if torch.cuda.is_available() else "cpu"

model.train()

for epoch in range(200):
  print("Epoch:", epoch)
  for idx, batch in enumerate(train_dataloader):
    input_ids = batch.pop("input_ids").to(device)
    pixel_values = batch.pop("pixel_values").to(device, torch.float16)

    outputs = model(input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=input_ids)
    
    loss = outputs.loss

    print("Loss:", loss.item())

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    # print some samples
    if idx % 100 == 0:
        img = batch["pixel_values"][0].permute(1, 2, 0).cpu().numpy()
        # normalize pixel values to be between 0 and 1
        img = img / 255.0
        # save the image
        plt.imsave(f"sample_{idx}.png", img)
        # generate caption
        generated_ids = model.generate(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            num_beams=4,
            max_length=32,
            early_stopping=True
        )
        # save the caption
        caption = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        with open(f"sample_{idx}.txt", "w") as f:
            f.write(caption)

