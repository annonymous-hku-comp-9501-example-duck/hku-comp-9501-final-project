import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from datasets.drivelm import DriveLM
from transformers import (
    AutoProcessor,
)
import torch.nn.functional as F


class DataModule(LightningDataModule):
    def __init__(
        self,
        cfg,
        shuffle=False,
        test=False,
        test_set="val",
        baseline='None',
    ):

        super().__init__()
        self.cfg = cfg
        self.test = test
        self.test_set = test_set
        self.batch_size = self.cfg.batch_size
        self.num_workers = self.cfg.num_workers
        self.shuffle = shuffle
        self.baseline = baseline

        self.processor = AutoProcessor.from_pretrained(
            self.cfg.model.name,
            use_fast=False,
        )


    def setup(self, stage=None):
        # create object of class by string of class name
        dataset_str = self.cfg.dataset.name
        dataset_class = globals()[dataset_str]

        if not self.test:
            self.train_dataset = dataset_class(
                self.cfg,
                **self.cfg.dataset,
                split="train",
            )
            val_split = "val"
            self.val_dataset = dataset_class(
                self.cfg,
                **self.cfg.dataset,
                split=val_split,
                # max_samples=100,
            )
            print(f"train_size: {len(self.train_dataset)}")
            print(f"val_size: {len(self.val_dataset)}")
        else:
            self.test_dataset = dataset_class(
                self.cfg,
                **self.cfg.dataset,
                split=self.test_set,
                test=True,
                baseline=self.baseline,
            )


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.dl_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.dl_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=16, #self.cfg.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.dl_collate_fn_test,
        )

    def dl_collate_fn(self, batch):
        """
        batch: (image, question, answer, action_tokens, image big size, training data, actions, actions_mask)

        """
        imgs = [row[0] for row in batch]
        questions = [row[1] for row in batch]
        answers = [row[2] for row in batch]
        action_tokens = [row[3] for row in batch]
        imgs_org = [row[4] for row in batch]
        training_data = [row[5] for row in batch]
        actions = [row[6] for row in batch]
        actions_mask = [row[7] for row in batch]
        action_tokens_past = [row[8] for row in batch]
        

        processed_imgs = [self.processor(images=img, padding="max_length", return_tensors="pt")[
                        "pixel_values"
                    ][0]
                    for img in imgs]

        processed_imgs = torch.stack(processed_imgs)
        question_tokens = self.processor.tokenizer(
                questions,
                # add_special_tokens=True,
                padding=True,
                # return_attention_mask=True,
                truncation=True,
                return_tensors="pt",
            )
        
        # concat question_tokens.data['input_ids'] with action_tokens_past
        if self.cfg.dataset.add_past_traj:

            question_tokens = self.processor.tokenizer(
                    questions,
                    add_special_tokens=False,
                    padding=True,
                    # return_attention_mask=True,
                    truncation=True,
                    return_tensors="pt",
                )
            
            eos = self.processor.tokenizer.eos_token_id
            # add eos token to end of action_tokens_past
            action_tokens_past = [torch.cat((action_tokens_past[i], torch.tensor([eos]))) for i in range(len(action_tokens_past))]
            
            action_tokens_past = torch.stack(action_tokens_past)
            question_tokens.data['input_ids'] = torch.cat((question_tokens.data['input_ids'], action_tokens_past), dim=1)

        answers_tokens_tmp = self.processor.tokenizer(
                answers,
                # add_special_tokens=True,
                padding=True,
                # return_attention_mask=True,
                truncation=True,
                return_tensors="pt",
            )
        
        actions = torch.stack(actions)
        actions_mask = torch.stack(actions_mask)
        # create array with 1 for action in training data and 0 else

        length = answers_tokens_tmp['input_ids'].shape[1]
        action_length = 14 # 2*6+2
        
        # if any training data is action
        if 'action' in training_data and action_length > length:
            # answers_tokens_tmp.data['input_ids'].shape[1] is length -> expand to action_length (add zeros)
            answers_tokens_tmp.data['input_ids'] = F.pad(answers_tokens_tmp.data['input_ids'], (0, action_length-length), 'constant', 0)

        length = answers_tokens_tmp['input_ids'].shape[1]

        # for all training data that is action
        action_indices = [i for i, x in enumerate(training_data) if x == 'action']
        for i in action_indices:
            # overwrite answer_tokens with action tokens
            answers_tokens_tmp['input_ids'][i] = F.pad(action_tokens[i]['input_ids'], (0, length-action_length), 'constant', 0)

        # use this when loading model with load_in_8bit=True
        # if self.cfg.model.finetuning == 'lora':
        #     processed_imgs = processed_imgs.type(torch.float16)

        # prevent out of memory error since we have some very long sequences in the dataset
        self.processor.tokenizer.model_max_length = self.cfg.max_seq_len

        return (
            processed_imgs,
            question_tokens,
            answers_tokens_tmp,
            questions,
            answers,
            imgs_org,
            actions,
            actions_mask
        )

    def dl_collate_fn_test(self, batch):
        """
        batch: (image, question, answer, action_tokens, image big size, training data, actions, actions_mask)

        """
        imgs = [row[0] for row in batch]
        questions = [row[1] for row in batch]
        answers = [row[2] for row in batch]
        action_tokens = [row[3] for row in batch]
        imgs_org = [row[4] for row in batch]
        training_data = [row[5] for row in batch]
        actions = [row[6] for row in batch]
        actions_mask = [row[7] for row in batch]
        baseline_actions = [row[8] for row in batch]
        scene_ids = [row[9] for row in batch]
        timestamps = [row[10] for row in batch]
        data_idx = [row[11] for row in batch]
        action_tokens_past = [row[12] for row in batch]
        answer_ego = [row[13] for row in batch]



        processed_imgs = [self.processor(images=img, padding="max_length", return_tensors="pt")[
                        "pixel_values"
                    ][0]
                    for img in imgs]

        processed_imgs = torch.stack(processed_imgs)
        question_tokens = self.processor.tokenizer(
                questions,
                # add_special_tokens=True,
                padding=True,
                # return_attention_mask=True,
                truncation=True,
                return_tensors="pt",
            )

        answers_tokens_tmp = self.processor.tokenizer(
                answers,
                # add_special_tokens=True,
                padding=True,
                # return_attention_mask=True,
                truncation=True,
                return_tensors="pt",
            )
        
        if self.cfg.dataset.add_past_traj:
            action_tokens_past = torch.stack(action_tokens_past)
            question_tokens.data['input_ids'] = torch.cat((question_tokens.data['input_ids'], action_tokens_past), dim=1)
        
        actions = torch.stack(actions)
        actions_mask = torch.stack(actions_mask)
        if baseline_actions[0] is not None:
            baseline_actions = torch.stack(baseline_actions)
        # create array with 1 for action in training data and 0 else

        length = answers_tokens_tmp['input_ids'].shape[1]
        action_length = 14 # 2*6+2
        
        # if any training data is action
        if 'action' in training_data and action_length > length:
            # answers_tokens_tmp.data['input_ids'].shape[1] is length -> expand to action_length (add zeros)
            answers_tokens_tmp.data['input_ids'] = F.pad(answers_tokens_tmp.data['input_ids'], (0, action_length-length), 'constant', 0)

        length = answers_tokens_tmp['input_ids'].shape[1]

        # for all training data that is action
        action_indices = [i for i, x in enumerate(training_data) if x == 'action']
        for i in action_indices:
            # overwrite answer_tokens with action tokens
            answers_tokens_tmp['input_ids'][i] = F.pad(action_tokens[i]['input_ids'], (0, length-action_length), 'constant', 0)

        # use this when loading model with load_in_8bit=True
        # if self.cfg.model.finetuning == 'lora':
        #     processed_imgs = processed_imgs.type(torch.float16)

        # prevent out of memory error since we have some very long sequences in the dataset
        self.processor.tokenizer.model_max_length = self.cfg.max_seq_len

        return (
            processed_imgs,
            question_tokens,
            answers_tokens_tmp,
            questions,
            answers,
            imgs_org,
            actions,
            actions_mask,
            baseline_actions,
            scene_ids,
            timestamps,
            data_idx,
            answer_ego,
        )
