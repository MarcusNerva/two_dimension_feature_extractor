#!/usr/bin/python3
# @Author  : MarcusNerva
# @Email   : yehanhua20@mails.ucas.ac.cn
from .resnext_101_32x8d import get_resnext101_32x8d, get_mul_resnext101_32x8d
from .utils_ import trans


def models_factory(model_name):
    if model_name == 'resnext101_32x8d':
        ret = get_resnext101_32x8d(pretrained=True)
    else:
        raise NotImplementedError

    return ret

def models_factory_mullevel(model_name):
    if model_name == 'resnext101_32x8d':
        ret = get_mul_resnext101_32x8d(pretrained=True)
    else:
        raise NotImplementedError

