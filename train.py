from datetime import datetime
from LMFNet.lib.dataset import Data
from torch.utils.data import DataLoader
from LMFNet.v2_3_19_v2_nosmooth_conv import Net
import torch
import torch.optim as optim
from torchsummary import summary
from tensorboardX import SummaryWriter
import sys
sys.path.append(r'C:/Users/32525/Desktop/lightvdt/lib')
import loss
from tqdm import tqdm
import os
import json

device = torch.device('cuda:0')

if __name__=='__main__':
    img_root = '/home/cuifengyu/data/VDT-2048_dataset/Train/'
    TIMESTAMP = "{0:%Y-%m-%d-VD_352}".format(datetime.now())
    print(TIMESTAMP)
    save_path = '/home/cuifengyu/Instance/LightVDT/model/' + TIMESTAMP
    if not os.path.exists(save_path): os.makedirs(save_path)

    lr = 0.0005
    batch_size =2
    epoch = 300
    num_params = 0
    data = Data(img_root)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)
    net = Net(pretrained=True).to(device)
    input0=torch.randn(1,3,352,352).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # summary(net,input_data=[input0,input0,input0]) 
    
    # if True:
    #     model = net.to(device)
    #     from thop import profile,clever_format
        
    #     flops,params=profile(model,inputs=(input0,input0,input0),verbose=False)
    #     flops,params=clever_format([flops,params],"%.3f")
    #     print(flops,params)

    hyperparameters = {
        'learning_rate': lr,
        'batch_size': batch_size,
        'num_epochs': epoch,
        'optimizer':'adam',
        'backboon':'mv2',
        'dataset':'VDT',
        'introduntion':'224size'
    }
    # 定义文件路径
    hyperparameters_file_path = save_path+'/superparameter.json'

    # 保存超参数到 JSON 文件
    with open(hyperparameters_file_path, 'w') as json_file:
        json.dump(hyperparameters, json_file, indent=4)


    iter_num = len(loader)
    train_log_dir = '/home/cuifengyu/Instance/LightVDT/logs/' + TIMESTAMP
    writer = SummaryWriter(train_log_dir)
    min_loss=100
    for epochi in range(1,epoch+1):
        net.train()
        epoch_loss=0
        net.zero_grad()
        for i,da in tqdm(enumerate(loader),total=len(loader),dynamic_ncols=True):
            Rgb = da[0].type(torch.FloatTensor).to(device)
            Thermal = da[1].type(torch.FloatTensor).to(device)
            Depth = da[2].type(torch.FloatTensor).to(device)
            mask = da[3].type(torch.FloatTensor).to(device) 

            sal = net(Rgb,Thermal,Depth)
            Loss = loss.muti_bce_loss_fusion(sal,mask)

            epoch_loss = epoch_loss+Loss.data
            Loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print('epoch-%2d_ave_loss: %7.6f' % (epochi, (epoch_loss / (i+1))))
        writer.add_scalar('loss/ep', epoch_loss/(i+1), epochi)
        if (epochi%50==0):
            torch.save(net.state_dict(), '%s/epoch_%d.pth' % (save_path, epochi))
        if (epochi >=100 ) and (epoch_loss/i<min_loss):
            min_loss = epoch_loss / i
            torch.save(net.state_dict(), '%s/epoch_%d.pth' % (save_path, epochi))
          
           
    torch.save(net.state_dict(), '%s/final.pth' % save_path)
        