import yaml
import os
import hydra

from pathlib import Path
from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary
from pytorch_lightning.loggers import CSVLogger, WandbLogger, TensorBoardLogger

from src.logging_project import setup_logging, sync_wandb

from argparse import ArgumentParser
from datasets.datamodule import DataModule
from models.wrapper import VLMWrapper


@hydra.main(config_path=f"config", config_name="config")
def main(cfg):
    
    # print config
    print(OmegaConf.to_yaml(cfg))
    os.environ["WANDB_DISABLE_CODE"] = "True"
    
    # setup debug mode
    overfit = 0.0
    if cfg.debug:
        os.environ["WANDB_MODE"] = "offline"
        cfg.expname = "debug"
        # overfit = 5  # use only 5 fixed batches for debugging

    if cfg.overfit > 0:
        overfit = cfg.overfit
        
    #TODO: caching
    
    
    # if we use mutliple GPUs and want wandb online it does need too much 
    # time on the MLCLoud and the training freezes or is too slow
    # log only local and sync afterwards with wandb sync [OPTIONS] [PATH]
    # if cfg.gpus > 1:
    #     os.environ["WANDB_MODE"] = "offline"
        
    # setup logging
    pl.seed_everything(cfg.seed)
    setup_logging(cfg)

    # resume training
    resume_path = "./checkpoints/last.ckpt"
    resume_wandb = False

    # if folder for this experiment does not exist set resume to true
    # to create necessary folders to resume wandb logging later
    if not os.path.exists(resume_path):
        resume_wandb = True
    elif os.path.exists(resume_path) and cfg.resume:
        resume_wandb = True

    if os.path.exists(resume_path) and cfg.resume:
        resume_path = resume_path
    else:
        resume_path = None


    
    # setup lightning logger
    # need to change renzka here
    csvlogger = CSVLogger("log/", "CSVLogger")
    wandblogger = WandbLogger(
        project=cfg.exp_folder_name,
        name=cfg.wandb_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        entity="chonghao",
        resume=resume_wandb,
    )
    Path(f"log/TBLogger").mkdir(parents=True, exist_ok=True)
    TBlogger = TensorBoardLogger("log/", name="TBLogger")



    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=-1,
        monitor=None,
        dirpath="./checkpoints",
        filename="{epoch:03d}",
        save_last=True,
        every_n_epochs=cfg.training.val_every_n_epochs,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_summary = ModelSummary(max_depth=3)

    
    model = VLMWrapper(cfg)
        
    
    wandblogger.watch(model)
    
    dm = DataModule(cfg)
    print(f"Number of GPUS: {cfg.gpus}")
    
    if cfg.gpus >= 1:
        trainer = Trainer(
            callbacks=[checkpoint_callback, lr_monitor, model_summary],
            accelerator="gpu",
            devices=cfg.gpus,
            strategy="ddp",
            logger=[csvlogger, wandblogger],
            log_every_n_steps=2,
            check_val_every_n_epoch=cfg.training.val_every_n_epochs,
            max_epochs=cfg.training.max_epochs,
            overfit_batches=overfit,
        )
    else:
        trainer = Trainer(
            callbacks=[checkpoint_callback, lr_monitor, model_summary],
            accelerator="gpu",
            devices=1,
            logger=[csvlogger, wandblogger],
            log_every_n_steps=2,
            check_val_every_n_epoch=cfg.training.val_every_n_epochs,
            max_epochs=cfg.training.max_epochs,
            overfit_batches=overfit,
        )

    trainer.fit(model, dm, ckpt_path=resume_path)


if __name__ == "__main__":
    main()
