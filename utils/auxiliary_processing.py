import numpy as np


def random_range(a=0, b=1):
    return np.random.rand() * (b - a) + a