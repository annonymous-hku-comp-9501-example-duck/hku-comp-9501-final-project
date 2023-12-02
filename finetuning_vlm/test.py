import yaml
from pathlib import Path
from argparse import ArgumentParser
from pytorch_lightning import Trainer

# from training.datasets.carla import ActionImageDataModule
from finetuning_vlm.datasets.datamodule import DataModule
from models.wrapper import VLMWrapper
import pytorch_lightning as pl


def main(hparams, cfg, working_dir, save_folder, load_random_weights=False):
    pl.seed_everything(hparams.seed)
    dm = DataModule(
        cfg, 
        test=True, 
        test_set=hparams.split,
        baseline=hparams.baseline,
    )
    # dm = ActionImageDataModule.from_argparse_args(hparams)
    if load_random_weights:
        model = VLMWrapper(cfg, working_dir=working_dir, save_folder=save_folder)
    else:
        model = VLMWrapper.load_from_checkpoint(
            hparams.checkpoint_path, working_dir=working_dir, save_folder=save_folder, baseline=hparams.baseline, test=True
        )  # , save_path=f"{Path(hparams.checkpoint_path).parent}/shuffle_False/{Path(hparams.checkpoint_path).parts[-1].split('-')[0]}/{hparams.split}")
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
    )
    trainer.test(model, dm)


if __name__ == "__main__":
    from hydra import compose, initialize
    from omegaconf import OmegaConf

    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="outputs/01_DriveLMv2_RT2/05_actions/finetune_lora/training_data_action_full/QA_reference_description_visual/action_command_input_False/discretize_linear/num_gpus_4/bs_4/lr_0.0001/seed_1234/checkpoints/epoch=011.ckpt",
    )
    # parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load_random_weights", type=int, default=0)
    parser.add_argument("--baseline", type=str, default='None') # random, command_mean, nearest_neighbour

    args = parser.parse_args()

    if args.load_random_weights:
        args.epoch = 0
    else:
        if args.checkpoint_path.split('/')[-1] == 'last.ckpt':
            args.epoch = 'last'
        else:
            args.epoch = int(
                f"{int(args.checkpoint_path.split('epoch=')[1].split('.')[0]):03d}"
            )

    config_path = Path(args.checkpoint_path).parent.parent
    initialize(config_path=f"../{config_path}/.hydra")
    cfg = compose(config_name="config")

    hydra_yaml = f"{config_path}/.hydra/hydra.yaml"
    with open(hydra_yaml, "r") as f:
        hydra_cfg = yaml.safe_load(f)
    working_dir = hydra_cfg["hydra"]["run"]["dir"]
    save_folder = f"test/epoch_{args.epoch}/split_{args.split}"


    # working_dir = working_dir.replace("/mnt/qb/work/geiger/krenz73/coding/03_eclip/eclip/outputs", "/home/krenz/coding/03_eclip/eclip/outputs_ssh")

    main(args, cfg, working_dir, save_folder, args.load_random_weights)
