#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ## 
#################################################

#SBATCH --nodes=1                   # How many nodes required? Usually 1
#SBATCH --cores=4                   # Number of CPU to request for the job
#SBATCH --mem=8GB                   # How much memory does your job require?
#SBATCH --gres=gpu:1                # Do you require GPUS? If not delete this line
#SBATCH --time=02-00:00:00          # How long to run the job for? Jobs exceed this time will be terminated
                                    # Format <DD-HH:MM:SS> eg. 5 days 05-00:00:00
                                    # Format <DD-HH:MM:SS> eg. 24 hours 1-00:00:00 or 24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL  # When should you receive an email?
#SBATCH --output=%u.%j.out          # Where should the log files go?
                                    # You must provide an absolute path eg /common/home/module/username/
                                    # If no paths are provided, the output file will be placed in your current working directory

################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=researchshort                 # The partition you've been assigned
#SBATCH --account=pradeepresearch   # The account you've been assigned (normally student)
#SBATCH --qos=research-1-qos       # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=kelltan.2019@smu.edu.sg # Who should receive the email notifications
#SBATCH --job-name=dcdjob     # Give the job a name

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Purge the environment, load the modules we require.
# Refer to https://violet.smu.edu.sg/origami/module/ for more information
module purge
module load Anaconda3/2022.05
module load CUDA/11.2.2


# Do not remove this line even if you have executed conda init
eval "$(conda shell.bash hook)"

# This command assumes that you've already created the environment previously
# We're using an absolute path here. You may use a relative path, as long as SRUN is execute in the same working directory
conda activate dcd

# If you require any packages, install it before the srun job submission.
# conda install pytorch torchvision torchaudio -c pytorch

# Submit your job to the cluster
srun --gres=gpu:1 python -m train.py \
--xpid=ued-BipedalWalker-Adversarial-Easy-v0-domain_randomization-noexpgrad \
--env_name=BipedalWalker-Adversarial-Easy-v0 \
--use_gae=True \
--gamma=0.99 \
--gae_lambda=0.9 \
--seed=88 \
--num_control_points=12 \
--recurrent_arch=lstm \
--recurrent_agent=False \
--recurrent_adversary_env=False \
--recurrent_hidden_size=1 \
--use_global_critic=False \
--lr=0.0003 \
--num_steps=2000 \
--num_processes=4 \
--num_env_steps=50000 \
--ppo_epoch=5 \
--num_mini_batch=16 \
--entropy_coef=0.001 \
--value_loss_coef=0.5 \
--clip_param=0.2 \
--clip_value_loss=False \
--adv_entropy_coef=0.01 \
--max_grad_norm=0.5 \
--algo=ppo \
--ued_algo=domain_randomization \
--use_plr=True \
--level_replay_prob=0.9 \
--level_replay_rho=0.5 \
--level_replay_seed_buffer_size=10 \
--level_replay_score_transform=rank \
--level_replay_temperature=0.1 \
--staleness_coef=0.3 \
--no_exploratory_grad_updates=True \
--use_editor=True \
--level_editor_prob=1.0 \
--level_editor_method=random \
--num_edits=3 \
--base_levels=easy \
--log_interval=10 \
--screenshot_interval=200 \
--log_grad_norm=True \
--normalize_returns=True \
--checkpoint_basis=student_grad_updates \
--archive_interval=5000 \
--reward_shaping=True \
--use_categorical_adv=True \
--use_skip=False \
--choose_start_pos=False \
--sparse_rewards=False \
--handle_timelimits=True \
--level_replay_strategy=positive_value_loss \
--test_env_names=BipedalWalker-v3,BipedalWalkerHardcore-v3,BipedalWalker-Med-Stairs-v0,BipedalWalker-Med-PitGap-v0,BipedalWalker-Med-StumpHeight-v0,BipedalWalker-Med-Roughness-v0 \
--log_dir=~/logs/accel_1 \
--test_interval=100 \
--test_num_episodes=10 \
--test_num_processes=2 \
--log_plr_buffer_stats=True \
--log_replay_complexity=True \
--checkpoint=True \
--log_action_complexity=False \
--diversity_coef=0.2 \
--diversity_transform=rank