import numpy as np
import torch
import time
from tqdm import tqdm
import os
from mobileone import reparameterize_model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')

import yaml


def computeTime(model):
    inputs = torch.randn(1, 3, 352, 352)
    
    model = model.to(device)
    inputs = inputs.to(device)

    if True:
        model = model.to(device)
        input1=torch.randn(1,3,352,352).to(device)
        from thop import profile,clever_format
        flops,params=profile(model,inputs=(input1,input1,input1))
        flops,params=clever_format([flops,params],"%.3f")
        print(flops,params)

    model.eval()
    # model = reparameterize_model(model)
    time_spent = []
    for idx in tqdm(range(1000)):
        start_time = time.time()
        with torch.no_grad():
            _ = model(inputs,inputs,inputs)

        if device == 'cuda:0':
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        if idx > 10:
            time_spent.append(time.time() - start_time)
    print('Average speed: {:.4f} fps'.format(1/ np.mean(time_spent)*1))


torch.backends.cudnn.benchmark = True

from allNet.v2_3_19_v2_nosmooth_conv_VT import Net
model = Net(inference=False)

computeTime(model)
