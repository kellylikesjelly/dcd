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
module load Xvfb


# Do not remove this line even if you have executed conda init
eval "$(conda shell.bash hook)"

# This command assumes that you've already created the environment previously
# We're using an absolute path here. You may use a relative path, as long as SRUN is execute in the same working directory
conda activate dcd

# If you require any packages, install it before the srun job submission.
# conda install pytorch torchvision torchaudio -c pytorch

# Submit your job to the cluster
srun --gres=gpu:1 python -m eval \
--base_path="~/logs/accel_1" \
--xpid=ued-BipedalWalker-Adversarial-Easy-v0-domain_randomization-noexpgrad \
--env_names=BipedalWalker-Med-Roughness-v0 \
--num_episodes=10 \
--result_path="~/results/accel_1"