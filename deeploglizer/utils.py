import sys
import torch
import os
import numpy as np
import json 
import pickle
import random
import hashlib
import logging
from datetime import datetime


def tensor2flatten_arr(tensor):
    return tensor.data.cpu().numpy().flatten()

def set_device(gpu=-1):
    if gpu != -1 and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu}')
    else:
        device = torch.device('cpu')
    return device
