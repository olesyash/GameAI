#!/bin/bash

#SBATCH --partition rtx3070              ### Specify partition name where to run a job.
#SBATCH --time 2-00:00:00               ### Job running time limit. Make sure it is not exceeding the partition time li>
#SBATCH --job-name 'olesya3'             ### Name of the job. replace my_job with your desired job name
#SBATCH --output output/job-%J.out             ### Output log for running job - %J is the job number variable
#SBATCH --mail-user=ilanash@post.jce.ac.il        ### User's email for sending job status
#SBATCH --mail-type=ALL                ### Conditions when to send the email. ALL,BEGIN,END,FAIL,REQUEU,NONE

#SBATCH --gpus=1                ### number of GPUs, ask for more than 1 only if you can parallelize your code for multi>
##SBATCH --mem=16G              ### ammount of RAM memory
#SBATCH --cpus-per-task=6     ### number of CPU cores

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

### Start you code below ####
module load anaconda                    ### load anaconda module (must present when working with conda environments)
source activate env_name                  ### activating environment, environment must be configured before running the j>
python main.py                          ### replace with your own command
