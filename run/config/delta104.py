import subprocess 
import time 

def submitjob(program, ngpus, jobname, run=False, wait=None):
    job="#!/bin/bash -l\n\n" \
        "#PBS -l nodes=1:ppn=1:gpus=%d\n" \
        "#PBS -l walltime=240:00:00\n" \
        "#PBS -j oe\n" \
        "#PBS -V\n\n" \
        % ngpus

    if wait is not None:
        dependency = "#PBS -W=afterok:%d\n" % wait
        job += dependency 

    job += "ncpu=`cat $PBS_NODEFILE | wc -l`\n" \
           "echo Running on `uniq -c $PBS_NODEFILE`\n" \
           "cd $PBS_O_WORKDIR\n" \
           "echo Running from $PBS_O_WORKDIR\n" \
           "echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES\n" \
           "echo\n\n"

    job += "echo ==== Job started at `date` ====\n"
    job += "echo\n"

    job += program

    job += "\necho\n"
    job += "echo ==== Job finished at `date` ====\n"

    # Generate the job file
    jobfile = open("jobfile.pbs", "w")
    jobfile.write(job)
    jobfile.close()

    # Submit the job 
    if run:
        cmd = ["qsub", "-V", "jobfile.pbs"]
        time.sleep(0.1)
    else:
        cmd = ["vim", "jobfile.pbs"]
    subprocess.check_call(cmd)

    return None
