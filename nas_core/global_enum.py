from enum import Enum


class Activation(Enum):
    Identity = 0,
    ReLU = 1,
    LeakyReLU = 2,
    Sigmoid = 3,
    ReLU6 = 4,
    HardSwish = 5,
    HardSigmoid = 6,
    Tanh = 7,
    PReLU = 8,


class Padding(Enum):
    SAME = -100,
    VALID = -200,


class CheckpointKey(object):
    Epoch = "epoch",
    ModelState = "state_dict",
    CoreMetric = "best_top1",
    OptimizerState = "optimizer_dict",
    ShadowModelState = "shadow_dict",
