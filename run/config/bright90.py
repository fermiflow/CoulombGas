import subprocess
import time

folder = "/data/xiehao/"

def submitjob(program, partition, ngpus, jobname, run=False, wait=None):
    job="#!/bin/bash -l\n\n" \
        "#SBATCH --partition=%s\n" \
        "#SBATCH --gres=gpu:%d\n" \
        "#SBATCH --nodes=1\n" \
        "#SBATCH --time=100:00:00\n" \
        "#SBATCH --job-name=%s\n" \
        "\n" \
        % (partition, ngpus, jobname)

    if wait is not None:
        pass

    job += "#export XLA_PYTHON_CLIENT_PREALLOCATE=false\n" \
           "echo The current job ID is $SLURM_JOB_ID\n" \
           "echo Running on $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST\n" \
           "echo Using $SLURM_NTASKS_PER_NODE tasks per node\n" \
           "echo A total of $SLURM_NTASKS tasks is used\n" \
           "echo List of CUDA devices: $CUDA_VISIBLE_DEVICES\n" \
           "echo\n\n"

    job += "echo ==== Job started at `date` ====\n"
    job += "echo\n"

    job += program

    job += "\necho\n"
    job += "echo ==== Job finished at `date` ====\n"

    # Generate the job file
    jobfile = open("jobfile.sh", "w")
    jobfile.write(job)
    jobfile.close()

    # Submit the job 
    if run:
        cmd = ["sbatch", "./jobfile.sh"]
        time.sleep(0.1)
    else:
        cmd = ["vim", "jobfile.sh"]
    subprocess.check_call(cmd)

    return None