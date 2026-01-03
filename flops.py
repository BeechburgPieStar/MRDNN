
import thop
import torch
# from Model.HCLNN import *
from Model.HCLNN_SCAF_6 import *


# net = HCLNNmodel()
net = HCLNNmodel_scaf_6()

flops, params = thop.profile(net, inputs=(torch.randn(1, 1, 4, 12),))


print('flops: ', flops, 'params: ', params)