from typing import List, Tuple, Union, Optional, Callable, Literal
from functools import partial
import torch
import torch as t, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import torchvision as tv
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os