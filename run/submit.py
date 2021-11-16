import os.path 

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

    # Parameters ###################################################################
    n, dim, rs, Theta = 49, 2, 1.0, 0.15
    Emax = 36
    nlayers, modelsize, nheads, nhidden = 2, 16, 4, 32
    depth, spsize, tpsize = 2, 16, 16
    Gmax, kappa = 15, 10
    mc_therm, mc_steps, mc_stddev = 10, 50, 0.1
    hutchinson = True
    lr = 1e-3
    sr, damping, max_norm = True, 1e-3, 1e-3
    batch, num_devices, acc_steps, epoch_finished, epoch = 512, 8, 16, 0, 3000
    ################################################################################

    program0 = 'python ../main.py'

    # The folder for saving the (standard) output of the job.
    jobdir='../jobs/'

    for rs in [10.0]:
        jobid = input.waitfor 

        args = {"n": n, "dim": dim, "rs": rs, "Theta": Theta,
                "Emax": Emax,
                "nlayers": nlayers, "modelsize": modelsize, "nheads": nheads, "nhidden": nhidden,
                "depth": depth, "spsize": spsize, "tpsize": tpsize,
                "Gmax": Gmax, "kappa": kappa,
                "mc_therm": mc_therm, "mc_steps": mc_steps, "mc_stddev": mc_stddev,
                "hutchinson": hutchinson,
                "lr": lr,
                "sr": sr, "damping": damping, "max_norm": max_norm,
                "batch": batch, "num_devices": num_devices, "acc_steps": acc_steps,
                "epoch_finished": epoch_finished, "epoch": epoch
                }
        jobname = jobdir 
        program = program0
        for param, value in args.items():
            jobname += "%s_%s_" % (param, value)
            if isinstance(value, bool):
                program += (" --%s" % param if value else "")
            else:
                program += " --%s %s" % (param, value)
        jobname = jobname[:-1] 

        jobid = conf.submitjob(program, args["num_devices"], jobname, run=input.run, wait=None)
