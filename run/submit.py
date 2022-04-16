import os 

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-run", action='store_true', help="Run or not")
    parser.add_argument("-waitfor", type=int, help="wait for this job for finish")
    input = parser.parse_args()

    # Select appropriate configs for the current machine. ##########################
    import socket, getpass
    username, hostname = getpass.getuser(), socket.gethostname()
    print ("The current location: %s@%s" % (username, hostname))
    import importlib
    try:
        conf = importlib.import_module("config.%s" % hostname)
    except ModuleNotFoundError:
        print ("Error! Where am I?")
    ################################################################################

    repo = "CoulombGas"

    from pygit2 import Repository
    branch = Repository('.').head.shorthand

    # Program and arguments ########################################################
    program0 = "python ../main.py"

    args = {"folder": os.path.join(conf.folder, repo, branch, ""),
            "n": 57, "dim": 2, "rs": 1.0, "Theta": 0.15,
            "Emax": 49,
            "nlayers": 2, "modelsize": 16, "nheads": 4, "nhidden": 32,
            "depth": 2, "spsize": 16, "tpsize": 16,
            "Gmax": 15, "kappa": 10,
            "mc_therm": 10, "mc_steps": 50, "mc_stddev": 0.1,
            "hutchinson": True,
            "lr": 1e-3,
            "sr": True, "damping": 1e-3, "max_norm": 1e-3,
            "batch": 512, "num_devices": 1, "acc_steps": 16,
            "epoch_finished": 0, "epoch": 3000,
            }
    ################################################################################

    # The folder for saving the (standard) output of the job.
    jobdir = '../jobs/'

    for rs in [1.0]:
        args["rs"] = rs
        jobid = input.waitfor 

        jobname = jobdir 
        program = program0
        for param, value in args.items():
            jobname += "%s_%s_" % (param, value)
            if isinstance(value, bool):
                program += (" --%s" % param if value else "")
            elif value is None:
                pass
            else:
                program += " --%s %s" % (param, value)
        jobname = jobname[:-1] 

        partition = "a100"
        ngpus = args.get("num_devices", 0)
        jobid = conf.submitjob(program, partition, ngpus, jobname, run=input.run, wait=None)