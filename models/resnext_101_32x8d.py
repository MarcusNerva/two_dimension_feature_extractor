#!/usr/bin/python3
# @Author  : MarcusNerva
# @Email   : yehanhua20@mails.ucas.ac.cn
from .utils_ import _resnet, Bottleneck

def get_resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet(False, 'resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

def get_mul_resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet(True, 'resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

