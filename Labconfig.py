# 1. Python 标准库
import copy
import glob
import os
from datetime import datetime
from functools import partial
import gc
import time

# 2. 第三方科学计算与数据处理库
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.stats import qmc
from tqdm import trange, tqdm

# 3. PyTorch 深度学习与分布式训练生态
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import torch.autograd as autograd 
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import _LRScheduler

# 4. 领域特定与本地模块 (根据需要取消注释)
# import deepwave