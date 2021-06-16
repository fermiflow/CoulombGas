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
    param1 = 2.0
    param2 = 6
    param3list = [0]
    ################################################################################

    program = '../test.py'

    # The folder for saving the (standard) output of the job.
    jobdir='../jobs/'

    nickname = 'pmap'
    resfolder = '/data1/wanglei/heg/' + nickname  + '/' 
    #cmd = ['mkdir', '-p', resfolder]
    #subprocess.check_call(cmd)
    
    for param3 in param3list:
        jobid = input.waitfor 

        args = {"param1": param1,
                "param2": param2,
                "param3": param3, 
                }
        jobname = jobdir 
        for param, value in args.items():
            jobname += "%s_%s_" % (param, value)
        jobname = jobname[:-1] 

        jobid = conf.submitjob(program, args, jobname, run=input.run, wait=None)
