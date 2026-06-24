
import torch
import torch.nn.functional as F
import numpy as np
from LMFNet.v2_3_19_v2_nosmooth_conv import Net
import sys
import os
sys.path.append('C:/Users/32525/Desktop/lightvdt/lib')
from LMFNet.lib.dataset import Data
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt

import time
from LMFNet.lib.loss import muti_bce_loss_fusion
from torchsummary import summary
from datetime import datetime
from tqdm import tqdm
import cv2

device = torch.device('cuda:0')

if __name__=='__main__':
    mode_path = "/home/cuifengyu/Instance/LightVDT/model/2025-03-21-VT/epoch_400.pth"

    data = Data(root='/home/cuifengyu/data/VDT-2048_dataset/Test/',mode='test')
    dataset = DataLoader(data,batch_size=1,shuffle=True,num_workers=4)
    net = Net(inference=False).to(device)
    net.eval
    net.load_state_dict(torch.load(mode_path,map_location=device),strict=True)
    input0=torch.randn(1,3,352,352).to(device)
   
    # summary(net,input_data=(input0,input0,input0))
    if True:
        model = net.to(device)
        input1=torch.randn(1,3,224,224).to(device)
        from thop import profile,clever_format
        flops,params=profile(model,inputs=(input0,input0,input0))
        flops,params=clever_format([flops,params],"%.3f")
        print(flops,params)
    # print(net)
    
    TIMESTAMP = "{0:%Y-%m-%d-VDT-multilevel_last2}".format(datetime.now())
    print(TIMESTAMP)
    out_path = '/home/cuifengyu/Instance/LightVDT/output/' + TIMESTAMP

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    time_s = time.time()
    image_num = len(dataset)
    with torch.no_grad():
        epoch_loss=0
        for rgb, t, d, mask, (H, W), name in tqdm(dataset):
            silent = net(rgb.to(device).float(),t.to(device).float(),d.to(device).float())
            Loss = muti_bce_loss_fusion(silent,mask.float().to(device))

            epoch_loss = epoch_loss+Loss.data
            
            silent = F.interpolate(silent[0],size=(H,W),mode='bilinear',align_corners=True)
            pred = np.squeeze(torch.sigmoid(silent).cpu().data.numpy())
            pred = (pred-pred.min())/(pred.max()-pred.min())

            # pred = np.where(pred<0.27,0,pred)
            # # &&&&&&&&&&&&&&&&特征可视化
            # fig, axes = plt.subplots(2, 1, figsize=(3, 6))
            # axes[0].imshow(mask[0, :, :, 0])
            # axes[1].imshow(255 * pred)
            # plt.subplots_adjust(right=0.95, left=0.05, bottom=0, top=1, hspace=0.2, wspace=0.2)
            # plt.title("final")
            # plt.savefig('/home/cuifengyu/Instance/LightVDT/plt_show/testGS.png')
            
            # &&&&&&&&&&&&&&&&&&&&
            cv2.imwrite(os.path.join(out_path, name[0][:-4] + '.png'), 255 * pred)
            # print(name[0][:-4] + '.png')
    time_e = time.time()
    print('speed: %f FPS' % (image_num / (time_e - time_s)))
    print('loss_=%f' % (epoch_loss/1000))
