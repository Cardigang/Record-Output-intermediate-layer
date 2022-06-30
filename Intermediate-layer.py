# 开发者 Alwyn
# 开发时间 2022/6/30 12:48
import torch
import torch.nn as nn
import torchvision

model = nn.Sequential(
            nn.Conv2d(3, 9, 1, 1, 0, bias=False),
            nn.BatchNorm2d(9),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

# The ReLU's Output
x = torch.rand([2, 3, 224, 224])
for i in range(len(model)):
    x = model[i](x)
    if i == 2:
        ReLu_out = x
print('ReLu_out.shape：\n\t',ReLu_out.shape)
print('x.shape：\n\t',x.shape)

from collections import OrderedDict

import torch
from torch import nn


# torchvision.models._utils.IntermediateLayerGetter
class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


# example
m = torchvision.models.resnet18(pretrained=True)
# extract layer1 and layer3, giving as names `feat1` and feat2`
new_m = torchvision.models._utils.IntermediateLayerGetter(m, {'layer1': 'feat1', 'layer3': 'feat2'})
out = new_m(torch.rand(1, 3, 224, 224))
print([(k, v.shape) for k, v in out.items()])

#Hook
class TestForHook(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_1 = nn.Linear(in_features=2, out_features=2)
        self.linear_2 = nn.Linear(in_features=2, out_features=1)
        self.relu = nn.ReLU()
        self.relu6 = nn.ReLU6()
        self.initialize()

    def forward(self, x):
        linear_1 = self.linear_1(x)
        linear_2 = self.linear_2(linear_1)
        relu = self.relu(linear_2)
        relu_6 = self.relu6(relu)
        layers_in = (x, linear_1, linear_2)
        layers_out = (linear_1, linear_2, relu)
        return relu_6, layers_in, layers_out

features_in_hook = []
features_out_hook = []

def hook(module, fea_in, fea_out):
    features_in_hook.append(fea_in)
    features_out_hook.append(fea_out)
    return None

net = TestForHook()

"""
# 第一种写法，按照类型勾，但如果有重复类型的layer比较复杂
net_chilren = net.children()
for child in net_chilren:
    if not isinstance(child, nn.ReLU6):
        child.register_forward_hook(hook=hook)
"""

"""
推荐下面我改的这种写法，因为我自己的网络中，在Sequential中有很多层，
这种方式可以直接先print(net)一下，找出自己所需要那个layer的名称，按名称勾出来
"""
layer_name = 'relu_6'
for (name, module) in net.named_modules():
    if name == layer_name:
        module.register_forward_hook(hook=hook)

print(features_in_hook)  # 勾的是指定层的输入
print(features_out_hook)  # 勾的是指定层的输出