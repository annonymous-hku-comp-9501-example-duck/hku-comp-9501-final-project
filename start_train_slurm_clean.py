import os
import subprocess
import time


def make_jobsub_file(
    num_gpus,
    seed,
    bs,
    lr,
    lr_scheduler,
    num_bins,
    add_past_traj,
    action_command_input,
    command_type,
    training_data,
    discretize,
    finetune,
    num_workers,
    folder_name,
    exp_folder_name,
    exp_name,
    job_number,
    job_name,
):
    # dataset = folder_name #f'train/slurm_{time.time()}'
    # job_number = 1
    os.makedirs(f"{folder_name}/slurm/run_files/logs", exist_ok=True)
    os.makedirs(f"{folder_name}/slurm/run_files/job_files", exist_ok=True)
    job_file = f"{folder_name}/slurm/run_files/job_files/{job_number}.sh"
    qsub_template = f"""#!/bin/bash
#SBATCH --job-name={job_name}_{job_number}
#SBATCH --partition=a100
#SBATCH -o {folder_name}/slurm/run_files/logs/qsub_out{job_number}.log
#SBATCH -e {folder_name}/slurm/run_files/logs/qsub_err{job_number}.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=katrin.renz@uni-tuebingen.de
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={8*num_gpus}
#SBATCH --mem={num_gpus*40}gb
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --exclude=r2s-n28,r2s-n38
# -------------------------------

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "Job started: $dt"

echo "Current branch:"
git branch
echo "Current commit:"
git log -1
echo "Current hash:"
git rev-parse HEAD

# export OMP_NUM_THREADS=8
# export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=36
export OPENBLAS_NUM_THREADS=1

# print info about current job
scontrol show job $SLURM_JOB_ID

unset KUBERNETES_PORT
# python src/log_cpu_mem_info.py --path {folder_name} &

python finetuning_vlm_QA/train.py \
resume=1 \
debug=0 \
num_workers={num_workers} \
seed={seed} \
gpus={num_gpus} \
batch_size={bs} \
max_seq_len={max_seq_len} \
training.max_epochs=50 \
model.finetuning={finetune} \
optimization.lr={lr} \
optimization.lr_scheduler={lr_scheduler} \
dataset.filter=front_view \
dataset.QA_reference={QA_reference} \
dataset.QA_type=all \
dataset.QA_version=v4 \
dataset.training_data={training_data} \
dataset.discretize={discretize} \
dataset.num_bins={num_bins} \
dataset.action_command_input={action_command_input} \
dataset.command_type={command_type} \
dataset.add_past_traj={add_past_traj} \
hydra.run.dir={folder_name} \
exp_folder_name={exp_folder_name} \
expname={exp_name} \
wandb_name={folder_name.replace("/", "_")} \
    
"""

    with open(job_file, "w") as f:
        f.write(qsub_template)
    return job_file


def get_num_jobs(job_name="datagen", username="krenz73"):
    # print(job_name)
    num_running_jobs = int(
        subprocess.check_output(
            (
                "SQUEUE_FORMAT2='username:7,name:130' squeue --sort V | grep"
                f" {username} | grep {job_name} | wc -l"
            ),
            shell=True,
        )
        .decode("utf-8")
        .replace("\n", "")
    )
    max_num_parallel_jobs = int(open("max_num_jobs.txt", "r").read())
    return num_running_jobs, max_num_parallel_jobs


if __name__ == "__main__":
    exp_folder_name = "02_DriveLMv2_ActionTemplates"
    exp_name = (  #'05_num_workers_comparison_after_np_dataset_change' #"03_normalize_image_train_val"
        "01_stage1_generate_action_templates_longer_training"  # "02_hyperparameter_comparison_num_frames1"
    )
    job_name = "drivelm_lr"

    
    finetunes = [
        "lora",
        # "qformer",
    ]
    num_gpuss = [4]
    num_workers = 8
    max_seq_len = 60 #70
    QA_reference = "description_visual" # description
    discretizes = [
        'linear',
        # 'square',
    ]
    num_binss = [
        # 32,
        # 64,
        # 128,
        256,
        # 310,
    ]
    training_datas = [
        # 'action',
        # 'action_QA',
        # 'action_full',
        # 'both',
        # 'QA',
        'QA_and_AT',
        'AT'
    ]
    action_command_input = True
    command_types = [
        # 'None',
        # 'action_template_13',
        # 'action_template_9',
        'action_template_7',
        'action_template_5',
        # 'action_template_3',
        # 'safe_actions', 
        # 'answer_ego',
    ]
    add_past_trajs = [
        True,
        # False,
    ]

    bss = [4]  # 128]
    lrs = [
        # 5e-3,
        # 1e-3,
        5e-4, #org
        # 1e-4,
        # 5e-5,
        # 1e-5,
        # 5e-6,
        # 1e-6,
    ]
    lr_schedulers = [
        'None',
        'cosine',
        # 'cosine_warmup',
        # 'cosine_restarts',
    ]


    seeds = [1234]
    # seeds = [15, 56, 134, 263, 589, 5427]

    tot_num_jobs = (
        len(seeds)
        * len(num_gpuss)
    )

    job_number = 0

    for seed in seeds:
        for num_gpus in num_gpuss:
            for bs in bss:
                for lr in lrs:
                    for discretize in discretizes:
                        for training_data in training_datas:
                            for finetune in finetunes:
                                for lr_scheduler in lr_schedulers:
                                    for num_bins in num_binss:
                                        for add_past_traj in add_past_trajs:
                                            for command_type in command_types:
                                                if command_type == 'None':
                                                    action_command_input = False
                                                else:
                                                    action_command_input = True
                                    
                                                folder_name = f"outputs/{exp_folder_name}/{exp_name}/finetune_{finetune}/training_data_{training_data}/QA_reference_{QA_reference}/action_command_input_{action_command_input}/command_type_{command_type}/add_past_traj_{add_past_traj}/discretize_{discretize}/num_bins_{num_bins}/num_gpus_{num_gpus}/bs_{bs}/lr_scheduler_{lr_scheduler}/lr_{lr}/seed_{seed}"

                                                job_file = make_jobsub_file(
                                                    num_gpus,
                                                    seed,
                                                    bs,
                                                    lr,
                                                    lr_scheduler,
                                                    num_bins,
                                                    add_past_traj,
                                                    action_command_input,
                                                    command_type,
                                                    training_data,
                                                    discretize,
                                                    finetune,
                                                    num_workers,
                                                    folder_name,
                                                    exp_folder_name,
                                                    exp_name,
                                                    job_number,
                                                    job_name,
                                                )
                                                (
                                                    num_running_jobs,
                                                    max_num_parallel_jobs,
                                                ) = get_num_jobs(
                                                    job_name=job_name
                                                )
                                                print(
                                                    f"{num_running_jobs}/{max_num_parallel_jobs} jobs"
                                                    " are running..."
                                                )
                                                while (
                                                    num_running_jobs
                                                    >= max_num_parallel_jobs
                                                ):
                                                    (
                                                        num_running_jobs,
                                                        max_num_parallel_jobs,
                                                    ) = get_num_jobs(
                                                        job_name=job_name
                                                    )
                                                    time.sleep(5)
                                                print(
                                                    "Submitting job"
                                                    f" {job_number}/{tot_num_jobs}:"
                                                    f" {job_name}"
                                                )

                                                # print(f"sbatch {job_file}")
                                                time.sleep(5)
                                                os.system(
                                                    f"sbatch {job_file}"
                                                )
                                                job_number += 1

    training_finished = False
    while not training_finished:
        num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=job_name)
        print(f"{num_running_jobs} jobs are running... Job: {job_name}")
        time.sleep(20)
        if num_running_jobs == 0:
            training_finished = True
