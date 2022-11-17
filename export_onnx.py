import json

from tqdm import tqdm
import numpy as np

import logging
import torch.onnx
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import os
from torchvision import transforms, datasets
from layers import MvPrunConv2d
from utils.utils import AverageMeter, accuracy, profile, replace_module_by_names, change_threshold
from model_zoo import ProxylessNASNets, MobileNetV3
from data_providers.imagenet import ImagenetDataProvider
import math
from torch.ao.quantization.qconfig import (
    add_module_to_qconfig_obs_ctr,
    default_dynamic_qconfig,
    float16_dynamic_qconfig,
    float_qparams_weight_only_qconfig,
    float_qparams_weight_only_qconfig_4bit,
    activation_is_memoryless
    )
def __set_model():
    net_config = json.load(open('net.config', 'r'))
    
    if net_config['name'] == ProxylessNASNets.__name__:
        model = ProxylessNASNets.build_from_config(net_config)
    elif net_config['name'] == MobileNetV3.__name__:
        model = MobileNetV3.build_from_config(net_config)
    else:
        raise ValueError("Not supproted network type: %s" % net_config['name'])
    
    init = torch.load("init", map_location="cpu")["state_dict"]
    model.load_state_dict(init)
    return model

def test(model,criterion, data_loader):
    
    model.eval()
    test_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    epoch_iterator = tqdm(data_loader,
                    desc="Validating... (loss=X.X) (Top1=X.X) (Top5=X.X)",
                    bar_format="{l_bar}{r_bar}",
                    dynamic_ncols=True)
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs, labels
            output = model(inputs)
            test_loss = criterion(output, labels)
            test_losses.update(test_loss.item(), inputs.size(0))
            
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            
            
    print("Test Average Loss: %2.5f"  % test_losses.avg)
    print("Test Top1 Accuracy: %2.5f" % top1.avg)

def convert_ONNX(model):
    model.eval()
    dummy_input = torch.randn(1, 3, 160, 160, requires_grad = True)

    torch.onnx.export(model,
            dummy_input,
            "Test.onnx",
            export_params = True,
            opset_version = 10,
            do_constant_folding = True,
            input_names = ['modelInput'],
            output_names = ['modelOutput'],
            dynamic_axes = {'modelInput': {0:'batch_size'},
                            'modelOutput': {0: 'batch_size'}}
            )
    print(" ")
    print("Model has been converted to ONNX")

if __name__ == "__main__":
    if os.path.exists('run.config'):
        image_size = json.load(open('run.config'))['image_size']
    else:
        image_size = 224

    model = __set_model()
    convert_ONNX(model)
