from torch import nn
import mobileone  
from LMFNet.MobileNetV2 import mobilenet_v2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes//ratio, 1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
        super(BasicConv2d, self).__init__()

       
        self.conv = nn.Sequential(nn.Conv2d(in_planes, in_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,groups=in_planes),
                              nn.Conv2d(in_planes,out_planes,kernel_size=1))
        
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    
class SDI(nn.Module):

    def __init__(self, channel,outchannel):
        super(SDI,self).__init__()

        # self.convs = nn.ModuleList(
        #     [nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1) for _ in range(5)])
        self.Conv = nn.Conv2d(channel,outchannel,1,1)
        

    def forward(self, xs, anchor):
        ans = torch.ones_like(anchor)
        target_size = anchor.shape[-1]

        for i, x in enumerate(xs):
            if x.shape[-1] > target_size:
                x = F.adaptive_avg_pool2d(x, (target_size, target_size))
            elif x.shape[-1] < target_size:
                x = F.interpolate(x, size=(target_size, target_size),
                                      mode='bilinear', align_corners=True)

            ans = ans * x
        ans = self.Conv(ans)
        return ans



class FusionM(nn.Module):
    def __init__(self,channel_everlear):
        super(FusionM,self).__init__()
        ch1,ch2,ch3,ch4,ch5= channel_everlear
        self.conv_dt1 = BasicConv2d(ch1*2,ch1,kernel_size=3,stride=1,padding=1)
        self.conv_dt2 = BasicConv2d(ch2*2,ch2,kernel_size=3,stride=1,padding=1)
        self.conv_dt3 = BasicConv2d(ch3*2,ch3,kernel_size=3,stride=1,padding=1)
        self.conv_dt4 = BasicConv2d(ch4*2,ch4,kernel_size=3,stride=1,padding=1)
        self.conv_dt5 = BasicConv2d(ch5*2,ch5,kernel_size=3,stride=1,padding=1)


        self.conv_tv1 = BasicConv2d(ch1*2,ch1,kernel_size=3,stride=1,padding=1)
        self.conv_tv2 = BasicConv2d(ch2*2,ch2,kernel_size=3,stride=1,padding=1)
        self.conv_tv3 = BasicConv2d(ch3*2,ch3,kernel_size=3,stride=1,padding=1)
        self.conv_tv4 = BasicConv2d(ch4*2,ch4,kernel_size=3,stride=1,padding=1)
        self.conv_tv5 = BasicConv2d(ch5*2,ch5,kernel_size=3,stride=1,padding=1)


    def forward(self,v,d,t):
        #D、T fusion
        dt1 = self.conv_dt1(torch.cat((d[0],t[0]),dim=1))
        dt2 = self.conv_dt2(torch.cat((d[1],t[1]),dim=1))
        dt3 = self.conv_dt3(torch.cat((d[2],t[2]),dim=1))
        dt4 = self.conv_dt4(torch.cat((d[3],t[3]),dim=1))
        dt5 = self.conv_dt5(torch.cat((d[4],t[4]),dim=1))
        dt=[dt1,dt2,dt3,dt4,dt5]
        
        #D、T result fusion with V
        tv1 = self.conv_tv1(torch.cat((dt[0],v[0]),dim=1))
        tv2 = self.conv_tv2(torch.cat((dt[1],v[1]),dim=1))
        tv3 = self.conv_tv3(torch.cat((dt[2],v[2]),dim=1))
        tv4 = self.conv_tv4(torch.cat((dt[3],v[3]),dim=1))
        tv5 = self.conv_tv5(torch.cat((dt[4],v[4]),dim=1))
        tv = [tv1,tv2,tv3,tv4,tv5]
        # semantic_pred = [dt,tv]
        # fig, axes = plt.subplots(2, 5, figsize=(20, 12))

        # for num, i in enumerate(semantic_pred):
        #     for n, m in enumerate(i):
        #         m = m.cpu()
        #         B, C, H, W = m.shape
        #         sum_tensor = torch.sum(m, dim=1)
        #         mean_tensor = sum_tensor / C
        #         mean_tensor = mean_tensor.detach().numpy()
        #         axes[num, n].imshow(mean_tensor[0, :, :])
        #         # axes[num, n].set_title(f' output_m{num, n}')
        # plt.subplots_adjust(hspace=0.05, wspace=0.1,right=0.95,left=0.05,bottom=0,top=1)
        # plt.savefig('Instance/LightVDT/plt_show/test_fusion.png')
        return tv
    
class Enhance(nn.Module):
    def __init__(self,delt=0.75, channel=32, channel_everlear=[]) -> None:
        super(Enhance,self).__init__()
        ch1,ch2,ch3,ch4,ch5= channel_everlear
        self.ca_0 = ChannelAttention(ch1)
        self.sa_0 = SpatialAttention()

        self.ca_1 = ChannelAttention(ch2)
        self.sa_1 = SpatialAttention()

        self.ca_2 = ChannelAttention(ch3)
        self.sa_2 = SpatialAttention()

        self.ca_3 = ChannelAttention(ch4)
        self.sa_3 = SpatialAttention()

        self.ca_4 = ChannelAttention(ch5)
        self.sa_4 = SpatialAttention()

        self.Translayer_0 = BasicConv2d(ch1, channel, 1,padding=0)
        self.Translayer_1 = BasicConv2d(ch2, channel,1,padding=0)
        self.Translayer_2 = BasicConv2d(ch3, channel, 1,padding=0)
        self.Translayer_3 = BasicConv2d(ch4, channel, 1,padding=0)
        self.Translayer_4 = BasicConv2d(ch5, channel, 1,padding=0)

        self.sdi_0 = SDI(channel,ch1)
        self.sdi_1 = SDI(channel,ch2)
        self.sdi_2 = SDI(channel,ch3)
        self.sdi_3 = SDI(channel,ch4)
        self.sdi_4 = SDI(channel,ch5)

       

    def forward(self,tv,f5):
        
        f0,f1,f2,f3,f4 = tv
        f0 = self.ca_0(f0) * f0
        f0 = self.sa_0(f0) * f0
        f0 = self.Translayer_0(f0)

        f1 = self.ca_1(f1) * f1
        f1 = self.sa_1(f1) * f1
        f1 = self.Translayer_1(f1)

        f2 = self.ca_2(f2) * f2
        f2 = self.sa_2(f2) * f2
        f2 = self.Translayer_2(f2)

        f3 = self.ca_3(f3) * f3
        f3 = self.sa_3(f3) * f3
        f3 = self.Translayer_3(f3)

        f4 = self.ca_4(f4) * f4
        f4 = self.sa_4(f4) * f4
        f4 = self.Translayer_4(f4)

        f41 = self.sdi_4([f0,f1, f2, f3, f4], f4)
        f31 = self.sdi_3([f0,f1, f2, f3, f4], f3)
        f21 = self.sdi_2([f0,f1, f2, f3, f4], f2)
        f11 = self.sdi_1([f0,f1, f2, f3, f4], f1)
        f01 = self.sdi_0([f0,f1, f2, f3, f4], f0)

        # f41 = self.sdi_4([ f3, f4], f4)
        # f31 = self.sdi_3([ f3, f4], f3)
        # f21 = self.sdi_2([ f2], f2)
        # f11 = self.sdi_1([ f1], f1)
        # f01 = self.sdi_0([ f0], f0)



        return [f01,f11,f21,f31,f41]
    
def d_module(channel1=512, channel2=256):
    return nn.Sequential(
            nn.Dropout2d(p=0.2),
            BasicConv2d(channel1, channel2, kernel_size=3, padding=1,),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )



class Decoder(nn.Module):
    def __init__(self,delt=0.75,channel_everlear=[]):
        super(Decoder,self).__init__()
        ch1,ch2,ch3,ch4,ch5=channel_everlear
        self.Conv_s5 = BasicConv2d(ch4,1,1,1)
        self.Conv_s4 = BasicConv2d(ch3,1,1,1)
        self.Conv_s3 = BasicConv2d(ch2,1,1,1)
        self.Conv_s2 = BasicConv2d(ch1,1,1,1)

        self.decoder5 = d_module(ch5,ch4)
        self.decoder4 = d_module(ch4,ch3)
        self.decoder3 = d_module(ch3,ch2)
        self.decoder2 = d_module(ch2,ch1)
        self.decoder1 = nn.Sequential(
                            nn.Dropout2d(p=0.2),
                            BasicConv2d(ch1, ch1, kernel_size=3, padding=1, dilation=1),
                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  
                            nn.Conv2d(ch1, 1, kernel_size=3, padding=1)
                        )
        
    
    def forward(self,enhanceResult,F_v):
        s5 = self.decoder5(enhanceResult[4]+F_v[4])
        s4 = self.decoder4(enhanceResult[3]+s5+F_v[3])
        s3 = self.decoder3(enhanceResult[2]+s4+F_v[2])
        s2 = self.decoder2(enhanceResult[1]+s3+F_v[1])
        s1 = self.decoder1(enhanceResult[0]+s2+F_v[0])
        B,C,W,H = s1.shape

        s5 = F.interpolate(self.Conv_s5(s5),(W,H))
        s4 = F.interpolate(self.Conv_s4(s4),(W,H))
        s3 = F.interpolate(self.Conv_s3(s3),(W,H))
        s2 = F.interpolate(self.Conv_s2(s2),(W,H))


        return s1,s2,s3,s4,s5


class Net(nn.Module) :
    def __init__(self,inference=False):
        super(Net, self).__init__()
        # mobileong_s0 channel
        # channel_everlear=[16,48,128,256,256]
        # self.mobile_v = mobileone.mobileone(num_classes=2,inference_mode=inference,num_blocks_per_stage=[2,8,5,5])
        # self.mobile_d = mobileone.mobileone(2,inference_mode=inference,num_blocks_per_stage=[2,8,1,1])
        # self.mobile_t = mobileone.mobileone(2,inference_mode=inference,num_blocks_per_stage=[2,8,5,1])

        # mobileV2 channelevery lear
        channel_everlear=[16,24,32,96,320]
        self.mobile_v = mobilenet_v2()
        self.mobile_d = mobilenet_v2()
        self.mobile_t = mobilenet_v2()

        self.Fusion = FusionM(channel_everlear=channel_everlear)
        self.Enhance = Enhance(channel_everlear=channel_everlear)
        self.Decoder = Decoder(channel_everlear=channel_everlear)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,v,t,d):
        F_v = self.mobile_v(v)
        # print(mobilenet_v2())
        F_d = self.mobile_v(d) #原来用的都是mobile_v,
        F_t = self.mobile_v(t)

        fusionResult = self.Fusion(F_v,F_d,F_t)
        enhanceResult = self.Enhance(fusionResult,F_v[4])
        finalResult,s2,s3,s4,s5 = self.Decoder(enhanceResult,F_v)

        # semantic_pred = [fusionResult, enhanceResult, [finalResult, s2, s3, s4, s5]]
        # fig, axes = plt.subplots(3, 5, figsize=(20, 12))

        # for num, i in enumerate(semantic_pred):
        #     for n, m in enumerate(i):
        #         m = m.cpu()
        #         B, C, H, W = m.shape
        #         sum_tensor = torch.sum(m, dim=1)
        #         mean_tensor = sum_tensor / C
        #         mean_tensor = mean_tensor.detach().numpy()
        #         axes[num, n].imshow(mean_tensor[0, :, :])
        #         axes[num, n].axis('off')  # 关闭坐标轴

        # plt.subplots_adjust(hspace=0.05, wspace=0.1, right=0.95, left=0.05, bottom=0, top=1)
        # plt.savefig('Instance/LightVDT/plt_show/test_all.png', bbox_inches='tight', pad_inches=0)

        # pred = [F_v]
        # fig, axes = plt.subplots(8, 8)
        
        # for num, i in enumerate(pred):
        #     for n, m in enumerate(i):
        #         m = m.cpu()
        #         B, C, H, W = m.shape
        #         mean_tensor = m
        #         mean_tensor = (mean_tensor-torch.min(mean_tensor))/(torch.max(mean_tensor)-torch.min(mean_tensor))*255
        #         mean_tensor = mean_tensor.detach().numpy()
        #         mean_tensor = np.round(mean_tensor).astype(np.uint8)
        #         # mean_tensor = np.stack((mean_tensor,mean_tensor,mean_tensor),axis=1)
        #         mean_tensor = np.transpose(mean_tensor,(0,2,3,1))
        #         for i in range(mean_tensor.shape[3]):
        #             num_channel = int(i/8)
        #             num_line = int(i%8)
        #             axes[num_channel, num_line].imshow(mean_tensor[0,:, :, int(i)], cmap='hot')
        #             #axes[num_channel, num_line].set_title(f' output_m{num, n}')
        #         plt.subplots_adjust(hspace=0.05, wspace=0.1, right=0.95, left=0.05, bottom=0, top=1)
        #         plt.savefig('Instance/LightVDT/plt_show/test.png')

        return [finalResult,s2,s3,s4,s5]
    def load_pretrained_model(self,pth):
        model_dict = self.mobile_v.state_dict()
        pretrained_dict = {k: v for k, v in pth.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.mobile_v.load_state_dict(model_dict)
        self.mobile_t.load_state_dict(model_dict)
        self.mobile_d.load_state_dict(model_dict)
        print('loading pretrained model success!')



