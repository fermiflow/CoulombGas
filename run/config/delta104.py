import subprocess 
import time 

def submitjob(program, args, jobname, run=False, wait=None):
    job="#!/bin/bash -l\n\n" \
        "#PBS -l nodes=1:ppn=1:gpus=1\n" \
        "#PBS -l walltime=24:00:00\n" \
        "#PBS -N %s\n" \
        "#PBS -o %s\n" \
        "#PBS -j oe\n" \
        "#PBS -V\n\n" \
        % (jobname, jobname + ".log")

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
    job += "python %s " % program
    for param, value in args.items():
        job += "--%s %s " % (param, value)
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
