from .alexnet import alexnet
from .vggnet import vgg11, vgg13, vgg16, vgg19
from .zfnet import zfnet 
from .googlenet_v1 import googlenet
from .xception import xception
from .resnet import  resnet34, resnet50, resnet101, resnext50_32x4d, resnext101_32x8d
from .densenet import densenet121, densenet161, densenet169, densenet201
from .dla import dla34
from .mobilenet_v3 import mobilenet_v3_small, mobilenet_v3_large
from .shufflenet_v2 import shufflenet_v2_x0_5, shufflenet_v2_x1_0
from .efficientnet_v2 import efficientnetv2_l, efficientnetv2_m, efficientnetv2_s
from .convnext import convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge
from .vision_kansformer import kit_base_patch16_224,kit_base_patch16_224_in21k
from .vision_transformer import vit_base_patch16_224, vit_base_patch32_224, vit_large_patch16_224, vit_base_patch16_224_in21k
from .vit2 import vit_base_patch16_224_ema
from .swin_transformer import swin_tiny_patch4_window7_224, swin_small_patch4_window7_224, swin_base_patch4_window7_224
from .repvit import repvit_m0_6,repvit_m0_9,repvit_m1_0
cfgs = {
    'alexnet': alexnet,
    'zfnet': zfnet,
    'vgg': vgg16,
    'vgg_tiny': vgg11,
    'vgg_small': vgg13,
    'vgg_big': vgg19,
    'googlenet': googlenet,
    'xception': xception,    
    'resnet_small': resnet34,
    'resnet': resnet50,
    'resnet_big': resnet101,
    'resnext': resnext50_32x4d,
    'resnext_big': resnext101_32x8d,
    'densenet_tiny': densenet121,
    'densenet_small': densenet161,
    'densenet': densenet169,
    'densenet_big': densenet121,
    'dla': dla34, 
    'mobilenet_v3': mobilenet_v3_small,
    'mobilenet_v3_large': mobilenet_v3_large,
    'shufflenet_small':shufflenet_v2_x0_5,
    'shufflenet': shufflenet_v2_x1_0,
    'efficient_v2_small': efficientnetv2_s,
    'efficient_v2': efficientnetv2_m,
    'efficient_v2_large': efficientnetv2_l,
    'convnext_tiny': convnext_tiny,
    'convnext_small': convnext_small,
    'convnext': convnext_base,
    'convnext_big': convnext_large,
    'convnext_huge': convnext_xlarge,
    'kansformer1': kit_base_patch16_224,
    'kansformer2': kit_base_patch16_224_in21k,
    'vision_transformer_small': vit_base_patch32_224,  
    'vit2': vit_base_patch16_224_ema, 
    'vision_transformer1': vit_base_patch16_224,
    'vision_transformer2': vit_base_patch16_224_in21k,
    'vision_transformer_big': vit_large_patch16_224,
    'swin_transformer_tiny': swin_tiny_patch4_window7_224,
    'swin_transformer_small': swin_small_patch4_window7_224,
    'swin_transformer': swin_base_patch4_window7_224,
    'repvitm06': repvit_m0_6,
    'repvitm09': repvit_m0_9,
    'repvitm10': repvit_m1_0
}

def find_model_using_name(model_name, num_classes):   
    return cfgs[model_name](num_classes)

 
