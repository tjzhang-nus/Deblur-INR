import torch
import torch.nn as nn
from .common import *


def fcn(num_input_channels=200, num_output_channels=1, num_hidden=1000):
    model = nn.Sequential()
    model.add(nn.Linear(num_input_channels, num_hidden, bias=True))
    model.add(nn.ReLU6())
    #
    model.add(nn.Linear(num_hidden, num_output_channels))
    #    model.add(nn.ReLU())
    model.add(nn.Softmax())
    #
    # model.add(nn.Sigmoid())
    return model

def fcn_linear(num_input_channels=200, num_output_channels=1, num_hidden=1000):
    model = nn.Sequential()
    model.add(nn.Linear(num_input_channels, num_hidden, bias=True))
    # model.add(nn.ReLU6())
    #
    model.add(nn.Linear(num_hidden, num_output_channels))
    #    model.add(nn.ReLU())
    model.add(nn.Softmax())
    #
    # model.add(nn.Sigmoid())
    return model


# import torch
# import torch.nn as nn
# from .common import *
# from .normalizer import Normalizer, Mydropout
#
# def fcn(num_input_channels=200, num_output_channels=1, num_hidden=1000):
#
#
#     model = nn.Sequential()
#     model.add(nn.Linear(num_input_channels, num_hidden,bias=True))
#     model.add(nn.ReLU6())
#     # model.add(nn.Dropout(p=0.3))
# #
#     model.add(nn.Linear(num_hidden, num_output_channels))
#     # model.add(nn.ReLU())
#     #
#     # model.add(nn.Sigmoid())
#     model.add(nn.Softmax())
#     # model.add(Normalizer(p=1))
# #
#     return model










