import pickle
import os

def ckpt_filename(epoch, path):
    return os.path.join(path, "epoch_%06d.pkl" % epoch)

def load_checkpoint(filename):
    with open(filename, "rb") as f:
        ckpt = pickle.load(f)
    return ckpt

def save_checkpoint(ckpt, filename):
    with open(filename, "wb") as f:
        pickle.dump(ckpt, f)
