import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal


def get_acq_fn():

    return NoAF



class AcquisitionFunctionWrapper():
    def __init__(self, model, l2r):
        self.model = model
        self.l2r = l2r

    def __call__(self, x):
        raise NotImplementedError()
    
    def update(self, data):
        self.fit(data)

    def fit(self, data):
        self.model.fit(data, reset=True)
    def save(self,path):
        torch.save(self.model.state_dict(),path)
    def load(self,path):
        self.model.load_state_dict(torch.load(path))


class NoAF(AcquisitionFunctionWrapper):
    def __call__(self, x):
        return self.l2r(self.model(x))
    def eval(self, x):
        mean, _ = self.model.eval(x)
        return self.l2r(mean)   

