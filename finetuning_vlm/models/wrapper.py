from pathlib import Path
from pprint import PrettyPrinter
from .VLM import VLM
from PIL import Image, ImageDraw
import numpy as np
import cv2
import json
import pickle as pkl

import torch
import pytorch_lightning as pl
from tokenizer import ActionTokenizer
from losses import PlanningLoss, CollisionLoss

from hydra.utils import get_original_cwd, to_absolute_path
import language_evaluation #https://github.com/bckim92/language-evaluation
# from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

pprint = PrettyPrinter().pprint


class VLMWrapper(pl.LightningModule):
    def __init__(
        self, cfg, working_dir=None, save_folder=None, baseline='None', test=False,
    ):  # , model_name: str, config: dict, microbatch_size: int):
        super().__init__()
        self.cfg = cfg
        self.working_dir = working_dir
        self.save_folder = save_folder
        self.baseline = baseline

        self.test_step_outputs = {'gt': [], 'pred': [], 'loss_l2': []}
        self.saved_imgs = 0

        self.save_hyperparameters()

        self.model = VLM(**self.cfg.model)

        model_type_name = self.cfg.model.name.split('/')[-1]
        discretize = self.cfg.dataset.discretize
        num_bins = 256 #self.cfg.dataset.num_bins

        self.action_tokenizer = ActionTokenizer(model_type_name, discretize, num_bins, past_traj=self.cfg.dataset.add_past_traj)

        self.planning_loss = PlanningLoss(loss_type='L2')
        #evaluate langugae outputs in language space
        self.evaluator = language_evaluation.CocoEvaluator(coco_types=["BLEU", "ROUGE_L", "CIDEr"])
        # self.evaluator = language_evaluation.CocoEvaluator(coco_types=["BLEU", "METEOR", "ROUGE_L", "CIDEr", "SPICE"])
        if test:
            self.data_info_all = pkl.load(open('data/DriveLM_v2/nuscenes_infos_temporal_val_xjw_w_annos_wo_mm.pkl', "rb"))
            

        # self.automatic_optimization = False



    def training_step(self, train_batch, idx):

        # optimizer = self.optimizers()
        # optimizer.zero_grad()

        
        images, questions, answers, questions_string, answers_string, _, actions_gt, actions_mask_gt = train_batch
        questions_input_ids = questions["input_ids"]
        # questions_attention_mask = questions["attention_mask"]
        answers_input_ids = answers["input_ids"]
        # answers_attention_mask = answers["attention_mask"]
        # print(questions_input_ids.real.shape)
        outputs = self.model(
            question_input_ids=questions_input_ids,
            answers_input_ids=answers_input_ids, 
            pixel_values=images,
            return_string=False,
        )

        # logits to tokens with argmax
        output_tokens = outputs.logits.argmax(-1)
        output_action = self.action_tokenizer.detokenize_batch(output_tokens)
        # output_action is np -> torch
        output_action = torch.from_numpy(output_action).to(self.device)

        # calculate loss
        loss_l2 = self.planning_loss(output_action, actions_gt, actions_mask_gt[..., 0])


        results_gen_dict = {}
        step = self.trainer.global_step
        # only if we have QA in the batch save results
        mask_sum_per_batch = actions_mask_gt.sum(dim=1)[..., 0]
        QA_in_batch = torch.any(mask_sum_per_batch == 0).item()
        action_in_batch = torch.any(mask_sum_per_batch > 0).item()

        if step % 20 == 0:
            generated, generated_ids = self.model(
                question_input_ids=questions_input_ids,
                answers_input_ids=answers_input_ids, 
                pixel_values=images,
                return_string=True,
            )
            epoch = self.trainer.current_epoch

            generated_ids = generated_ids[..., 1:]
            output_action = self.action_tokenizer.detokenize_batch(generated_ids)
            output_action = torch.from_numpy(output_action).to(self.device)
            loss_l2_tmp = self.planning_loss(output_action, actions_gt, actions_mask_gt[..., 0]).item()

            # get current step
            # log results in txt file for each epoch
            save_folder = './results_train/'
            Path(save_folder).mkdir(parents=True, exist_ok=True)
            with open(save_folder + f'results_{epoch}_step_{step}.txt', 'a') as f:
                f.write(f'Epoch: {epoch}\n')
                for ix in range(len(questions_string)):
                    f.write(f'Question: {questions_string[ix]}\n')
                    f.write(f'Answer: {answers_string[ix]}\n')
                    f.write(f'Generated: {generated[ix]}\n')
                    f.write(f'Loss L2: {loss_l2_tmp}\n\n')

            del generated #, results_gen, results_gen_dict

        #     results_gen = self.evaluator.run_evaluation(
        #             generated, answers_string
        #         )
        #     results_gen_dict = {
        #             f"train/{k}": v for k, v in results_gen.items()
        #         }
            
        loss = outputs.loss
        results_gen_dict.update({"train/loss": loss.detach().cpu()})
        results_gen_dict.update({"train/loss_l2_with_GT": loss_l2.detach().cpu()})
        self.log_dict(results_gen_dict, on_epoch=True, prog_bar=True)

        # loss.backward()
        # optimizer.step()

        # print lr
        # lr_scheduler = self.lr_schedulers()
        # print("LR:" + str(lr_scheduler.get_lr()[0]))
        


        del outputs, images, questions, answers, questions_string, answers_string, questions_input_ids, answers_input_ids
        torch.cuda.empty_cache()

        return loss

    def validation_step(self, val_batch, idx):
        images, questions, answers, questions_string, answers_string, _, actions_gt, actions_mask_gt = val_batch
        questions_input_ids = questions["input_ids"]
        # questions_attention_mask = questions["attention_mask"]
        answers_input_ids = answers["input_ids"]
        # answers_attention_mask = answers["attention_mask"]
        mask_sum_per_batch = actions_mask_gt.sum(dim=1)[...,0]
        QA_in_batch = torch.any(mask_sum_per_batch == 0).item()
        action_in_batch = torch.any(mask_sum_per_batch > 0).item()
        action_indices = torch.where(mask_sum_per_batch > 0)[0]

        outputs = self.model(
            question_input_ids=questions_input_ids,
            answers_input_ids=answers_input_ids, 
            pixel_values=images,
            return_string=False,
        )
        results_gen_dict = {}
        epoch = self.trainer.current_epoch

        r2, generated_ids = self.model(
            question_input_ids=questions_input_ids,
            answers_input_ids=answers_input_ids, 
            pixel_values=images,
            return_string=True,
        )

        if QA_in_batch:
            # remove action samples from answers and questions
            for i in sorted(action_indices, reverse=True):
                del(r2[i])
                del(answers_string[i])
                del(questions_string[i])

            save_folder = './results_val/'
            Path(save_folder).mkdir(parents=True, exist_ok=True)
            with open(save_folder + f'results_{epoch}.txt', 'a') as f:
                f.write(f'Epoch: {epoch}\n')
                for ix in range(len(questions_string)):
                    f.write(f'Question: {questions_string[ix]}\n')
                    f.write(f'Answer: {answers_string[ix]}\n')
                    f.write(f'Generated: {r2[ix]}\n\n')


            results_gen = self.evaluator.run_evaluation(
                r2, answers_string
            )
        
            results_gen_dict = {
                f"val/{k}": v for k, v in results_gen.items()
            }

            del r2


        if action_in_batch:
            # logits to tokens with argmax
            # output_tokens = outputs.logits.argmax(-1)
            generated_ids = generated_ids[..., 1:]
            
            output_action = self.action_tokenizer.detokenize_batch(generated_ids)
            # output_action is np -> torch
            output_action = torch.from_numpy(output_action).to(self.device)

            # calculate loss
            loss_l2 = self.planning_loss(output_action, actions_gt, actions_mask_gt[..., 0])
            results_gen_dict.update({"val/loss_l2": loss_l2.detach().cpu()})


        loss = outputs.loss

        results_gen_dict.update({"val/loss": loss.detach().cpu()})

        self.log_dict(
            results_gen_dict,
            prog_bar=False,
            rank_zero_only=False,
        )
        # self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        del outputs, images, questions, answers, questions_string, answers_string, questions_input_ids, answers_input_ids
        torch.cuda.empty_cache()


    def test_step(self, test_batch, idx):
        images, questions, answers, questions_string, answers_string, imgs_raw, actions_gt, actions_mask_gt, baseline_actions, scene_tokens, timestamp, data_idx, answer_ego = test_batch
        mask_sum_per_batch = actions_mask_gt.sum(dim=1)[...,0]
        QA_in_batch = torch.any(mask_sum_per_batch == 0).item()
        action_in_batch = torch.any(mask_sum_per_batch > 0).item()
        action_indices = torch.where(mask_sum_per_batch > 0)[0]


        if self.baseline != 'None':
            loss_l2 = self.planning_loss(baseline_actions, actions_gt, actions_mask_gt[..., 0]).item()
            for i in range(len(baseline_actions)):
                assert self.data_info_all[data_idx[i]]['scene_token'] == scene_tokens[i]
                self.data_info_all[data_idx[i]]['annos']['actions_pred'] = baseline_actions[i].cpu().detach().numpy()

        else:
            questions_input_ids = questions["input_ids"]
            # questions_attention_mask = questions["attention_mask"]
            answers_input_ids = answers["input_ids"]
            # answers_attention_mask = answers["attention_mask"]
            r2, generated_ids = self.model(
                question_input_ids=questions_input_ids,
                answers_input_ids=answers_input_ids, 
                pixel_values=images,
                return_string=True,
            )
            outputs = self.model(
                question_input_ids=questions_input_ids,
                answers_input_ids=answers_input_ids, 
                pixel_values=images,
                return_string=False,
            )



            if action_in_batch:
                generated_ids = generated_ids[..., 1:]

                # logits to tokens with argmax
                output_tokens = outputs.logits.argmax(-1)
                output_action = self.action_tokenizer.detokenize_batch(generated_ids)
                # output_action = self.action_tokenizer.detokenize_batch(generated_ids)
                # output_action is np -> torch
                output_action = torch.from_numpy(output_action).to(self.device)
            
                for i in range(len(output_action)):
                    if i in action_indices:
                        assert self.data_info_all[data_idx[i]]['scene_token'] == scene_tokens[i]
                        self.data_info_all[data_idx[i]]['annos']['actions_pred'] = output_action[i].cpu().detach().numpy()
                    else:
                        pass
                # loss_l2 = []
                # for i in range(len(output_action)):
                #     loss_l2.append(self.planning_loss(output_action[i], actions_gt[i], actions_mask_gt[i,..., 0]).item())
                
                # loss_l2_mean = np.mean(loss_l2)
                loss_l2 = self.planning_loss(output_action, actions_gt, actions_mask_gt[..., 0]).item()
                self.test_step_outputs['loss_l2'].append(loss_l2)
                


            gen_img = True

            # get epoch
            epoch = self.trainer.current_epoch
            # get current step
            # log results in txt file for each epoch
            # save_folder = 'results_test/'

            if gen_img and self.saved_imgs < 100:
                self.saved_imgs += 1
                save_path = f"{self.working_dir}/{self.save_folder}/black_image"
                Path(save_path).mkdir(parents=True, exist_ok=True)
                
                # to PIL image
                img = imgs_raw[0]
                img = img.astype(np.uint8)
                img2 = img.copy()

                img = Image.fromarray(img)
                # text = "Why should the driver of the own vehicle brake?"

                black_box = np.zeros((550, img2.shape[1], 3), dtype=np.uint8)
                # add text to black box
                cv2.putText(black_box, questions_string[0], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


                # add prediction and ground truth to black box
                cv2.putText(black_box, "Answer: " + answers_string[0], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(black_box, "Predicted: " + r2[0], (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(black_box, "Ego action: " + answer_ego[0], (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                black_box_pil = Image.fromarray(black_box)
                d = ImageDraw.Draw(black_box_pil)

                wp_image_height = 550
                wp_image_width = 600
                pixel_per_meter = 15
                for i in range(actions_gt[0].shape[0]):
                    wp_gt = actions_gt[0][i]
                    wp_gt = wp_gt.cpu().detach().numpy()
                    wp_pred = output_action[0][i]

                    wp_y_pixel = wp_image_height - 20 - int(
                        wp_gt[1] * pixel_per_meter
                    )
                    wp_x_pixel = wp_image_width // 2 + int(
                        wp_gt[0] * pixel_per_meter
                    )
                    
                    wp_y_pixel_pred = wp_image_height - 20 - int(
                        wp_pred[1] * pixel_per_meter
                    )
                    wp_x_pixel_pred = wp_image_width // 2 + int(
                        wp_pred[0] * pixel_per_meter
                    )

                    # draw wps as points
                    d.ellipse(
                        (
                            wp_x_pixel - 2,
                            wp_y_pixel - 2,
                            wp_x_pixel + 2,
                            wp_y_pixel + 2,
                        ),
                        fill="white",
                    )
                    d.ellipse(
                        (
                            wp_x_pixel_pred - 2,
                            wp_y_pixel_pred - 2,
                            wp_x_pixel_pred + 2,
                            wp_y_pixel_pred + 2,
                        ),
                        fill="red",
                    )

                black_box = np.array(black_box_pil)

                # save images
                img_all = np.concatenate((img, black_box), axis=0)
                img_all = Image.fromarray(img_all)

                # folder_name = 'img_blip2/content_what_do_driver'
                # Path(folder_name).mkdir(parents=True, exist_ok=True)
                
                img_all.save(f'{save_path}/test_{idx}.png')

                # generate img with waypoints
                save_path = f"{self.working_dir}/{self.save_folder}/waypoints"
                Path(save_path).mkdir(parents=True, exist_ok=True)
                
                # create black PIL image
                wp_image_height = 350
                wp_image_width = 600
                pixel_per_meter = 15

                img_wp = Image.new(
                    "RGB",
                    (img2.shape[1], wp_image_height),
                    color="black",
                )
                # create PIL draw object
                d = ImageDraw.Draw(img_wp)
                # breakpoint()

                for i in range(actions_gt[0].shape[0]):
                    wp_gt = actions_gt[0][i]
                    wp_gt = wp_gt.cpu().detach().numpy()
                    wp_pred = output_action[0][i]

                    wp_y_pixel = wp_image_height - 20 - int(
                        wp_gt[1] * pixel_per_meter
                    )
                    wp_x_pixel = wp_image_width // 2 + int(
                        wp_gt[0] * pixel_per_meter
                    )
                    
                    wp_y_pixel_pred = wp_image_height - 20 - int(
                        wp_pred[1] * pixel_per_meter
                    )
                    wp_x_pixel_pred = wp_image_width // 2 + int(
                        wp_pred[0] * pixel_per_meter
                    )

                    # draw wps as points
                    d.ellipse(
                        (
                            wp_x_pixel - 2,
                            wp_y_pixel - 2,
                            wp_x_pixel + 2,
                            wp_y_pixel + 2,
                        ),
                        fill="white",
                    )
                    d.ellipse(
                        (
                            wp_x_pixel_pred - 2,
                            wp_y_pixel_pred - 2,
                            wp_x_pixel_pred + 2,
                            wp_y_pixel_pred + 2,
                        ),
                        fill="red",
                    )

                img = np.concatenate([np.array(img), np.array(img_wp)], axis=0)
                img = Image.fromarray(img)

                img.save(f'{save_path}/test_{idx}.png')


            if QA_in_batch:
                for i in sorted(action_indices, reverse=True):
                    del(r2[i])
                    del(answers_string[i])
                    del(questions_string[i])
                self.test_step_outputs['gt'].extend(answers_string)
                self.test_step_outputs['pred'].extend(r2)

        
        torch.cuda.empty_cache()

    def on_test_epoch_end(self) -> None:
        if self.baseline == 'None' and len(self.test_step_outputs['gt']) > 0:

            print(len(self.test_step_outputs['gt']))
            print(len(self.test_step_outputs['pred']))
            results_all = self.evaluator.run_evaluation(self.test_step_outputs['pred'], self.test_step_outputs['gt'])
            pprint(results_all)
        else:
            results_all = {}
        # write dict to json file

        # accuracy of gt and pred (both are strings)
        accuracy = sum([1 if gt == pred else 0 for gt, pred in zip(self.test_step_outputs['gt'], self.test_step_outputs['pred'])]) / len(self.test_step_outputs['gt'])

        gt_speed = []
        pred_speed = []
        gt_steering = []
        pred_steering = []
        for gt, pred in zip(self.test_step_outputs['gt'], self.test_step_outputs['pred']):
            gt_speed.append(gt.split('. ')[1])
            pred_speed.append(pred.split('. ')[1])
            gt_steering.append(gt.split('. ')[0])
            pred_steering.append(pred.split('. ')[0])

        # accuracy of speed
        accuracy_speed = sum([1 if gt == pred else 0 for gt, pred in zip(gt_speed, pred_speed)]) / len(gt_speed)
        # accuracy of steering
        accuracy_steering = sum([1 if gt == pred else 0 for gt, pred in zip(gt_steering, pred_steering)]) / len(gt_steering)


        # mean of loss_l2
        loss_l2_mean = np.mean(self.test_step_outputs['loss_l2'])
        results_all.update({"test/loss_l2": loss_l2_mean})
        results_all.update({"test/accuracy": accuracy})
        results_all.update({"test/accuracy_speed": accuracy_speed})
        results_all.update({"test/accuracy_steering": accuracy_steering})

        Path(f"{self.working_dir}/{self.save_folder}/baseline_{self.baseline}").mkdir(
            parents=True, exist_ok=True
        )

        # write all gt and preds to txt file
        with open(f"{self.working_dir}/{self.save_folder}/baseline_{self.baseline}/results_gen.txt", "w") as f:
            for gt, pred in zip(self.test_step_outputs['gt'], self.test_step_outputs['pred']):
                f.write(f"GT: {gt}\n")
                f.write(f"Pred: {pred}\n\n")



        with open(f"{self.working_dir}/{self.save_folder}/baseline_{self.baseline}/results_gen.json", "w") as fp:
            json.dump(results_all, fp, indent=4)

        # save data_info_all
        with open(f"{self.working_dir}/{self.save_folder}/baseline_{self.baseline}/data_info_all.pkl", "wb") as fp:
            pkl.dump(self.data_info_all, fp)
    
    # Sourced from https://github.com/PyTorchLightning/pytorch-lightning/issues/5449
    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        # if self.trainer.max_steps:
        #    return self.trainer.max_steps

        # dataset = self.train_dataloader()
        dataset = self.trainer._data_connector._train_dataloader_source.dataloader()
        dataset_size = len(dataset)

        num_devices = max(1, self.trainer.num_devices)
        # if self.trainer.num_devices:
        #     num_devices = max(num_devices, self.trainer.num_devices)

        effective_batch_size = (
            dataset.batch_size * self.trainer.accumulate_grad_batches * num_devices
        )
        # print(dataset.batch_size, self.trainer.accumulate_grad_batches, num_devices)
        # print(dataset_size, effective_batch_size, self.trainer.max_epochs)
        # num_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs
        num_steps = (
            dataset_size
            * self.trainer.max_epochs
            // (self.trainer.accumulate_grad_batches * num_devices)
        )
        # print(num_steps)
        return num_steps
    

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.cfg.optimization.lr,
            weight_decay=0.1,
        )
        
        # torch.optim.AdamW(
        #     self.model.parameters(),
        #     lr=self.cfg.optimization.lr,
        #     betas=(0.9, 0.98 if self.isViT else 0.999),
        #     eps=1e-6 if self.isViT else 1e-8,
        #     weight_decay=0.2,
        # )

        # Source: https://github.com/openai/CLIP/issues/107
        # Use pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
        # lr_scheduler = CosineAnnealingWarmupRestarts(
        #     optimizer,
        #     first_cycle_steps=self.num_training_steps,
        #     cycle_mult=1.0,
        #     max_lr=self.cfg.optimization.lr,
        #     min_lr=0,
        #     warmup_steps=self.cfg.optimization.warmup_steps,
        # )

        if self.cfg.optimization.get("lr_scheduler", "None") == "None":
            return {"optimizer": optimizer} #, "lr_scheduler": lr_scheduler}

        elif self.cfg.optimization.lr_scheduler == "cosine":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.num_training_steps,
                eta_min=0,
                last_epoch=-1,
                # verbose=True
            )
        
        elif self.cfg.optimization.lr_scheduler == "cosine_restarts":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.num_training_steps//self.trainer.max_epochs,
                T_mult=1,
                eta_min=0,
                last_epoch=-1,
            )

        elif self.cfg.optimization.lr_scheduler == "cosine_warmup":
            from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

            lr_scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=self.num_training_steps,
                cycle_mult=1.0,
                max_lr=self.cfg.optimization.lr,
                min_lr=0,
                warmup_steps=self.cfg.optimization.warmup_steps,
            )
        
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
        }
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
