import subprocess 
import time 

folder = "/data/home/scv1691/archive/"

def submitjob(program, partition, ngpus, jobname, run=False, wait=None):
    del partition

    job="#!/bin/bash -l\n\n" \
        "module load anaconda/2020.11\n" \
        "module load cuda/11.2\n" \
        "conda activate myenv\n\n"

    if wait is not None:
        pass

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
        cmd = ["sbatch", "--gpus=%d" % ngpus, "./jobfile.sh"]
        time.sleep(0.1)
    else:
        cmd = ["vim", "jobfile.sh"]
    subprocess.check_call(cmd)

    return None