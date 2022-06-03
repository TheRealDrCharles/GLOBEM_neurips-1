import sys, os
from datetime import datetime
sys.path.append("./")
sys.path.append("../")
import subprocess
import time
import itertools
from pathlib import Path

flag_compute_data = False
current_folder = os.path.dirname(os.path.realpath(__file__))

task_str = r'''#!/bin/bash

#SBATCH --mail-type=FAIL # Send email on these events
#SBATCH --mail-user=xuhaixu@uw.edu

#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --time=23:59:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --export=all
'''

task_str2 = r'''
# Modules to use (optional).
. "/gscratch/ubicomp/xuhaixu/miniconda3/etc/profile.d/conda.sh" # Enables conda shell

# Your programs to run.
conda activate behavior_modeling
'''

config_name_list = [
    # "ml_saeb",
    # "ml_canzian",
    # "ml_wahle",
    # "ml_farhan",
    # "ml_lu",
    # "ml_wang",
    # "ml_chikersal",
    # "ml_xu_interpretable", 
    # "ml_xu_personalized",

    "dl_erm_1dCNN",
    "dl_erm_2dCNN",
    "dl_erm_LSTM",
    "dl_erm_Transformer",
    "dl_erm_mixup",
    "dl_irm",
    "dl_mldg_ds_as_domain",
    "dl_mldg_person_as_domain",
    "dl_masf_ds_as_domain",
    "dl_masf_person_as_domain",
    "dl_dann_ds_as_domain",
    "dl_dann_person_as_domain",
    "dl_csd_ds_as_domain",
    "dl_csd_person_as_domain",
    "dl_siamese",
    "dl_reorder",

    # "dl_clustering_20",
    # "dl_clustering_30",
    # "dl_clustering_40",
    # "dl_clustering_50",
    # "dl_clustering_60",
    # "dl_clustering_70",
    # "dl_clustering_80",
]

df_clustering_config_list = {}
n_clusters_list = [60]
for n_clusters in n_clusters_list:
    if n_clusters not in df_clustering_config_list:
        df_clustering_config_list[n_clusters] = []
    for cluster_idx in range(n_clusters):
        config_name = f"clustering_each_cluster_config/n_clusters_{n_clusters}/dl_clustering_{n_clusters}_{cluster_idx}"
        df_clustering_config_list[n_clusters].append(config_name)

config_name_list_full = config_name_list
# config_name_list_full = []
# for n_clusters in n_clusters_list:
    # config_name_list_full += [df_clustering_config_list[n_clusters][0]]
    # config_name_list_full += df_clustering_config_list[n_clusters]
# config_name_list_full = config_name_list_full[::-1]

# eval_task_list = ["two", "allbutone", "crossgroup", "two_overlap"]
# eval_task_list = ["two"]
# eval_task_list = ["single_within_user--crosscovid--allbutone"]
# eval_task_list = ["allbutone"]
# eval_task_list = ["allbutone--crosscovid"]
# eval_task_list = ["crossgroup"]
# eval_task_list = ["crosscovid"]
# eval_task_list = ["all"]
# eval_task_list = ["two_overlap"]

# eval_task_list = ["two--allbutone--crossgroup"]

eval_task_list = ["allbutone--single_within_user","crosscovid--two_overlap"]
# eval_task_list = ["two_overlap--single_within_user"]

# eval_task_list = ["allbutone","single_within_user--crosscovid"]
# eval_task_list = ["single_within_user--crosscovid"]

# eval_task_list = ["allbutone--crosscovid"]
eval_task_list = ["two_overlap"]
# eval_task_list = ["single_within_user"]


prefix = "/gscratch/ubicomp/xuhaixu/passive_sensing_generalizability/"

slurm_folder_path = "tmp/slurm_tune_jobs/"

Path(slurm_folder_path).mkdir(parents=True, exist_ok=True)

for config_name, eval_task in itertools.product(config_name_list_full, eval_task_list):

    job_file = slurm_folder_path + f"{config_name}_{eval_task}.slurm"
    Path(os.path.split(job_file)[0]).mkdir(parents=True, exist_ok=True)
    
    # if not ("dl_reorder" in config_name):
    #     continue

    print(job_file)

    # continue

    with open(job_file, "w") as fh:
        fh.writelines(task_str)
        # if ("chikersal" in config_name and "two" in eval_task):
        #     task_str_middle = "#SBATCH --account=ubicomp\n#SBATCH --partition=gpu-2080ti\n#SBATCH --ntasks-per-node=18\n#SBATCH --mem=130G"
        # elif ("chikersal" in config_name and "allbutone" in eval_task):
        #     task_str_middle = "#SBATCH --account=cse\n#SBATCH --partition=gpu-a40\n#SBATCH --ntasks-per-node=48\n#SBATCH --mem=300G"
        if ("ml_chikersal" in config_name):
            task_str_middle = "#SBATCH --account=cse\n#SBATCH --partition=gpu-a40\n#SBATCH --ntasks-per-node=12\n#SBATCH --mem=500G"
            # task_str_middle = "#SBATCH --account=ubicomp\n#SBATCH --partition=gpu-2080ti\n#SBATCH --ntasks-per-node=12\n#SBATCH --mem=300G"
        elif ("clustering" in config_name):
            # task_str_middle = "#SBATCH --account=cse\n#SBATCH --partition=gpu-a40\n#SBATCH --ntasks-per-node=28\n#SBATCH --mem=100G"
            task_str_middle = "#SBATCH --account=cse\n#SBATCH --partition=ckpt\n#SBATCH --ntasks-per-node=4\n#SBATCH --mem=40G"
        elif ("dl_masf_person_as_domain" in job_file):
            task_str_middle = "#SBATCH --account=cse\n#SBATCH --partition=gpu-a40\n#SBATCH --ntasks-per-node=8\n#SBATCH --mem=40G"
        elif ("dl_mldg_person_as_domain" in job_file):
            task_str_middle = "#SBATCH --account=cse\n#SBATCH --partition=gpu-a40\n#SBATCH --ntasks-per-node=8\n#SBATCH --mem=40G"
        elif ("dl_irm" in job_file):
            task_str_middle = "#SBATCH --account=cse\n#SBATCH --partition=gpu-a40\n#SBATCH --ntasks-per-node=6\n#SBATCH --mem=40G"
        elif ("dl_erm_2dCNN" in job_file):
            task_str_middle = "#SBATCH --account=cse\n#SBATCH --partition=gpu-a40\n#SBATCH --ntasks-per-node=6\n#SBATCH --mem=70G"
        elif ("dl_erm_LSTM" in job_file):
            task_str_middle = "#SBATCH --account=cse\n#SBATCH --partition=gpu-a40\n#SBATCH --ntasks-per-node=6\n#SBATCH --mem=40G"
        elif ("dl_erm_Transformer" in job_file):
            task_str_middle = "#SBATCH --account=ubicomp\n#SBATCH --partition=gpu-2080ti\n#SBATCH --ntasks-per-node=8\n#SBATCH --mem=40G"
        elif ("ml_xu_personalized" in job_file):
            task_str_middle = "#SBATCH --account=cse\n#SBATCH --partition=gpu-a40\n#SBATCH --ntasks-per-node=19\n#SBATCH --mem=200G"
        elif ("ml_xu_interpretable" in job_file):
            task_str_middle = "#SBATCH --account=cse\n#SBATCH --partition=gpu-a40\n#SBATCH --ntasks-per-node=19\n#SBATCH --mem=120G"
        # elif ("dl_reorder" in job_file):
            # task_str_middle = "#SBATCH --account=ubicomp\n#SBATCH --partition=gpu-2080ti\n#SBATCH --ntasks-per-node=10\n#SBATCH --mem=100G"
        elif ("ml_" in config_name):
            task_str_middle = "#SBATCH --account=cse\n#SBATCH --partition=ckpt\n#SBATCH --ntasks-per-node=10\n#SBATCH --mem=70G"
            # task_str_middle = "#SBATCH --account=ubicomp\n#SBATCH --partition=gpu-2080ti\n#SBATCH --ntasks-per-node=10\n#SBATCH --mem=100G"
        else:
        # if True:
            task_str_middle = "#SBATCH --account=cse\n#SBATCH --partition=ckpt\n#SBATCH --ntasks-per-node=10\n#SBATCH --mem=50G"
        fh.writelines(task_str_middle)
        fh.writelines(f"\n#SBATCH --output={prefix}{slurm_folder_path}{config_name}_{eval_task}_log.log # where STDOUT goes\n#SBATCH --job-name=eval_{config_name.replace('/','_')}_{eval_task}\n")
        fh.writelines(task_str2)
        fh.writelines(f"\npython {prefix}evaluation/model_train_eval.py --config_name={config_name} --eval_task={eval_task}\n")
        # fh.writelines(f"\npython {prefix}evaluation/model_train_eval.py --config_name={config_name} --eval_task={eval_task} --pred_target=dep_endterm\n")

    os.system("sbatch %s" %job_file)
    time.sleep(1)