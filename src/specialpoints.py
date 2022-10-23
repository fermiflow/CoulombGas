import numpy as np

def Monkhorst_Pack(dim, Nk):
    if dim == 2:
        if Nk == 1:
            twists = [np.array([0., 0.])]
            weights = [1.]
        elif Nk == 2:
            twists = [np.array([1/4, 1/4])]
            weights = [1.]
        elif Nk == 3:
            twists = [np.array([0., 0.]),
                      np.array([1/3, 0.]),
                      np.array([1/3, 1/3]),
            ]
            weights = [1/9, 4/9, 4/9]
        elif Nk == 4:
            twists = [np.array([1/8, 1/8]),
                      np.array([3/8, 1/8]),
                      np.array([3/8, 3/8]),
            ]
            weights = [1/4, 1/2, 1/4]
    return twists, weights