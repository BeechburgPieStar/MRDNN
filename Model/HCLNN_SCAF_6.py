

import torch
from torch import nn

from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP, NCP

import torch.nn.functional as F

class LNN(nn.Module):
    def __init__(self, c_in=4, c_out=8):
        super(LNN, self).__init__()
        wiring = AutoNCP(19, c_out)  # 16 units, 1 motor neuron
        self.model = CfC(c_in, wiring, proj_size=c_out)
        self.hx = None

    def forward(self, gradient):
        # gradient = torch.transpose(gradient, 2, 1)
        gradient, self.hx = self.model.forward(gradient, self.hx)
        self.hx = self.hx.detach()
        out = gradient[:, -1, :]
        return out

class HCLNNmodel_scaf_6(nn.Module):
    def __init__(self):
        super(HCLNNmodel_scaf_6, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features=8,out_features=16),
            nn.Tanh(),
            nn.Linear(in_features=16,out_features=2)
        )

        self.LNN = LNN()

        self.channel_1 = 1
        self.channel_2 = 2
        self.channel_3 = self.channel_2
        self.channel_4 = self.channel_2
        self.channel_5 = self.channel_2
        self.channel_6 = self.channel_2
        self.channel_7 = self.channel_2
        self.channel_8 = self.channel_2
        self.channel_9 = self.channel_2
        self.channel_10 = self.channel_2
        self.channel_11 = self.channel_2
        self.channel_12 = self.channel_2

        self.cov_1 = nn.Conv2d(in_channels=self.channel_1, out_channels=self.channel_2,kernel_size=1)
        # self.cov_2 = nn.Conv2d(in_channels=self.channel_1, out_channels=self.channel_2,kernel_size=1)

        self.cov_3 = nn.Conv2d(in_channels=self.channel_2, out_channels=self.channel_3, kernel_size=3, padding=(1,1))
        self.cov_4 = nn.Conv2d(in_channels=self.channel_2, out_channels=self.channel_2, kernel_size=3)

        self.cov_5 = nn.Conv2d(in_channels=self.channel_3, out_channels=self.channel_3, kernel_size=3, padding=(1,1))
        self.cov_6 = nn.Conv2d(in_channels=self.channel_2, out_channels=self.channel_2, kernel_size=3, padding=(1,1))

        self.cov_7 = nn.Conv2d(in_channels=self.channel_3, out_channels=self.channel_3, kernel_size=3, padding=(1,1))
        self.cov_8 = nn.Conv2d(in_channels=self.channel_2, out_channels=self.channel_2, kernel_size=3, padding=(1,1))

        # self.fc1 = nn.Linear(self.channel_3, int(self.channel_3 * 0.5))
        # self.fc2 = nn.Linear(int(self.channel_3 * 0.5),self.channel_3)
        #
        # self.fc3 = nn.Linear(self.channel_2, int(self.channel_2 * 0.5))
        # self.fc4 = nn.Linear(int(self.channel_2 * 0.5),self.channel_2)
        #
        # self.fc5 = nn.Linear(self.channel_3, int(self.channel_2 * 0.5))
        # self.fc6 = nn.Linear(int(self.channel_2 * 0.5), self.channel_3)
        #
        # self.fc7 = nn.Linear(self.channel_2, int(self.channel_2 * 0.5))
        # self.fc8 = nn.Linear(int(self.channel_2 * 0.5),self.channel_2)
        #
        # self.fc9 = nn.Linear(self.channel_4, int(self.channel_4 * 0.5))
        # self.fc10 = nn.Linear(int(self.channel_4 * 0.5),self.channel_4)
        #
        # self.fc11 = nn.Linear(self.channel_6, int(self.channel_6 * 0.5))
        # self.fc12 = nn.Linear(int(self.channel_6 * 0.5),self.channel_6)
        #
        # self.fc13 = nn.Linear(self.channel_5, int(self.channel_5 * 0.5))
        # self.fc14 = nn.Linear(int(self.channel_5 * 0.5),self.channel_5)
        #
        # self.fc15 = nn.Linear(self.channel_7, int(self.channel_7 * 0.5))
        # self.fc16 = nn.Linear(int(self.channel_7 * 0.5),self.channel_7)

        self.fc1 = nn.Linear(self.channel_3, self.channel_3 * 2)
        self.fc2 = nn.Linear(self.channel_3 * 2,self.channel_3)

        self.fc3 = nn.Linear(self.channel_2, self.channel_2 * 2)
        self.fc4 = nn.Linear(self.channel_2 * 2,self.channel_2)

        self.fc5 = nn.Linear(self.channel_3, self.channel_3 * 2)
        self.fc6 = nn.Linear(self.channel_3 * 2, self.channel_3)

        self.fc7 = nn.Linear(self.channel_2, self.channel_2 * 2)
        self.fc8 = nn.Linear(self.channel_2 * 2,self.channel_2)

        self.fc9 = nn.Linear(self.channel_4, self.channel_4 * 2)
        self.fc10 = nn.Linear(self.channel_4 * 2,self.channel_4)

        self.fc11 = nn.Linear(self.channel_6, self.channel_6 * 2)
        self.fc12 = nn.Linear(self.channel_6 * 2,self.channel_6)

        self.fc13 = nn.Linear(self.channel_5, self.channel_5 * 2)
        self.fc14 = nn.Linear(self.channel_5 * 2,self.channel_5)

        self.fc15 = nn.Linear(self.channel_7, self.channel_7 * 2)
        self.fc16 = nn.Linear(self.channel_7 * 2,self.channel_7)

        self.sigmod = nn.Sigmoid()

        self.pad1 = torch.nn.ReflectionPad2d(1)

        self.cov_9 = nn.Conv2d(in_channels=self.channel_3, out_channels=self.channel_3, kernel_size=3)

        self.cov_10 = nn.Conv2d(in_channels=self.channel_2, out_channels=self.channel_2, kernel_size=(1,3))
        self.cov_11 = nn.Conv2d(in_channels=self.channel_3, out_channels=self.channel_3, kernel_size=(3,5))

        self.cov_12 = nn.Conv2d(in_channels=self.channel_2, out_channels=self.channel_2, kernel_size=3, padding=(1,1))
        self.cov_13 = nn.Conv2d(in_channels=self.channel_3, out_channels=self.channel_3, kernel_size=3, padding=(1,1))
        self.cov_14 = nn.Conv2d(in_channels=self.channel_4, out_channels=self.channel_4, kernel_size=3, padding=(1,1))

        self.cov_15 = nn.Conv2d(in_channels=self.channel_2, out_channels=self.channel_2, kernel_size=3, padding=(1,1))
        self.cov_16 = nn.Conv2d(in_channels=self.channel_3, out_channels=self.channel_3, kernel_size=3, padding=(1,1))
        self.cov_17 = nn.Conv2d(in_channels=self.channel_4, out_channels=self.channel_4, kernel_size=3, padding=(1,1))
        self.cov_18 = nn.Conv2d(in_channels=self.channel_3, out_channels=self.channel_3, kernel_size=3)
        self.cov_19 = nn.Conv2d(in_channels=self.channel_3, out_channels=self.channel_3, kernel_size=(3,5))
        self.cov_20 = nn.Conv2d(in_channels=self.channel_2, out_channels=self.channel_2, kernel_size=(1,3))

        self.cov_24 = nn.Conv2d(in_channels=self.channel_6, out_channels=self.channel_6, kernel_size=3, padding=(1,1))
        self.cov_25 = nn.Conv2d(in_channels=self.channel_5, out_channels=self.channel_5, kernel_size=3, padding=(1,1))
        self.cov_26 = nn.Conv2d(in_channels=self.channel_7, out_channels=self.channel_7, kernel_size=3, padding=(1,1))

        self.cov_28 = nn.Conv2d(in_channels=self.channel_6, out_channels=self.channel_6, kernel_size=3, padding=(1,1))
        self.cov_29 = nn.Conv2d(in_channels=self.channel_5, out_channels=self.channel_5, kernel_size=3, padding=(1,1))
        self.cov_30 = nn.Conv2d(in_channels=self.channel_7, out_channels=self.channel_7, kernel_size=3, padding=(1,1))

        self.cov_32 = nn.Conv2d(in_channels=self.channel_6, out_channels=self.channel_6, kernel_size=3)
        self.cov_33 = nn.Conv2d(in_channels=self.channel_6, out_channels=self.channel_6, kernel_size=(3,5))
        self.cov_34 = nn.Conv2d(in_channels=self.channel_5, out_channels=self.channel_5, kernel_size=(1,3))

        self.cov_38 = nn.Conv2d(in_channels=self.channel_10, out_channels=self.channel_10, kernel_size=3, padding=(1,1))
        self.cov_39 = nn.Conv2d(in_channels=self.channel_9, out_channels=self.channel_9, kernel_size=3, padding=(1,1))
        self.cov_40 = nn.Conv2d(in_channels=self.channel_11, out_channels=self.channel_11, kernel_size=3, padding=(1,1))

        self.cov_42 = nn.Conv2d(in_channels=self.channel_10, out_channels=1, kernel_size=1)
        self.cov_43 = nn.Conv2d(in_channels=self.channel_9, out_channels=1, kernel_size=1)
        self.cov_44 = nn.Conv2d(in_channels=self.channel_11, out_channels=1, kernel_size=1)


    def forward(self, x):

        x = self.cov_1(x)


        x1 = self.cov_3(x)
        x2 = self.cov_4(x)

        x1 = self.cov_5(x1)
        x2 = self.cov_6(x2)

        x1 = self.cov_7(x1)
        x2 = self.cov_8(x2)

        LH_x1 = F.adaptive_max_pool2d(x1, (1, 1))
        LH_x2 = x2.mean(dim=1,keepdim=True)

        LH_x1 = self.sigmod(self.fc2(self.fc1(LH_x1.squeeze(2).squeeze(2))))
        LH_x2 = LH_x2.repeat(1, self.channel_3, 1, 1)  # 重复第二维

        LH_1 = LH_x1.unsqueeze(2).unsqueeze(2) * LH_x2

        LH_1 = self.pad1(LH_1)
        SCAF1_1 = LH_1 + x1 #B,channel3,4,12

        HL_x1 = self.cov_9(x1)
        HL_x1 = HL_x1.mean(dim=1, keepdim=True)
        HL_x1 = HL_x1.repeat(1, self.channel_2, 1, 1)

        HL_x2 = F.adaptive_max_pool2d(x2, (1, 1))
        HL_x2 = self.sigmod(self.fc4(self.fc3(HL_x2.squeeze(2).squeeze(2))))

        HL_2 = HL_x2.unsqueeze(2).unsqueeze(2) * HL_x1
        SCAF1_2 = HL_2 + x2 #B,channel2,2,10

        x11 = self.cov_11(x1)
        x12 = self.cov_10(x2)

        x_cat1 = x11 + x12

        x2 = self.cov_12(SCAF1_2)
        x1 = self.cov_13(SCAF1_1)
        x3 = self.cov_14(x_cat1)

        x2 = self.cov_15(x2)
        x1 = self.cov_16(x1)
        x3 = self.cov_17(x3)

        LH_x1 = F.adaptive_max_pool2d(x1, (1, 1))
        LH_x1 = self.sigmod(self.fc6(self.fc5(LH_x1.squeeze(2).squeeze(2))))
        LH_x2 = x2.mean(dim=1, keepdim=True)
        LH_x2 = LH_x2.repeat(1, self.channel_3, 1, 1)
        LH_2_1_1 = LH_x1.unsqueeze(2).unsqueeze(2) * LH_x2
        LH_2_1_1 = self.pad1(LH_2_1_1)
        LH_x3 = x3.mean(dim=1, keepdim=True)
        LH_x3 = LH_x3.repeat(1, self.channel_3, 1, 1)
        LH_2_1_2 = LH_x1.unsqueeze(2).unsqueeze(2) * LH_x3
        LH_2_1_2 = self.pad1(LH_2_1_2)
        LH_2_1_2 = self.pad1(LH_2_1_2)[:,:,1:5,:]
        SCAF2_1 = LH_2_1_1 + LH_2_1_2 + x1

        HL_x2 = F.adaptive_max_pool2d(x2,(1, 1))
        HL_x2 = self.sigmod(self.fc8(self.fc7(HL_x2.squeeze(2).squeeze(2))))
        HL_x1 = self.cov_18(x1)
        HL_x1 = HL_x1.mean(dim=1, keepdim=True)
        HL_x1 = HL_x1.repeat(1, self.channel_2, 1, 1)
        HL_2_2_1 = HL_x2.unsqueeze(2).unsqueeze(2) * HL_x1
        LH_x3 = x3.mean(dim=1, keepdim=True)
        LH_x3 = LH_x3.repeat(1, self.channel_2, 1, 1)
        LH_2_2_2 = HL_x2.unsqueeze(2).unsqueeze(2) * LH_x3
        LH_2_2_2 = self.pad1(LH_2_2_2)[:,:,1:3,:]
        SCAF2_2 = HL_2_2_1 + LH_2_2_2 + x2

        HL_x3 = F.adaptive_max_pool2d(x3, (1, 1))
        HL_x3 = self.sigmod(self.fc10(self.fc9(HL_x3.squeeze(2).squeeze(2))))
        HL_x1 = self.cov_19(x1)
        HL_x1 = HL_x1.mean(dim=1, keepdim=True)
        HL_x1 = HL_x1.repeat(1, self.channel_4, 1, 1)
        HL_2_3_1 = HL_x3.unsqueeze(2).unsqueeze(2) * HL_x1
        HL_x2 = self.cov_20(x2)
        HL_x2 = HL_x2.mean(dim=1, keepdim=True)
        HL_x2 = HL_x2.repeat(1, self.channel_4, 1, 1)
        HL_2_3_2 = HL_x3.unsqueeze(2).unsqueeze(2) * HL_x2
        SCAF2_3 = HL_2_3_1 + HL_2_3_2 + x3

        x1 = self.cov_24(SCAF2_1)
        x2 = self.cov_25(SCAF2_2)
        x3 = self.cov_26(SCAF2_3)

        x1 = self.cov_28(x1)
        x2 = self.cov_29(x2)
        x3 = self.cov_30(x3)

        LH_x1 = F.adaptive_max_pool2d(x1, (1, 1))
        LH_x1 = self.sigmod(self.fc12(self.fc11(LH_x1.squeeze(2).squeeze(2))))
        LH_x2 = x2.mean(dim=1, keepdim=True)
        LH_x2 = LH_x2.repeat(1, self.channel_6, 1, 1)
        LH_3_1_1 = LH_x1.unsqueeze(2).unsqueeze(2) * LH_x2
        LH_3_1_1 = self.pad1(LH_3_1_1)
        LH_x3 = x3.mean(dim=1, keepdim=True)
        LH_x3 = LH_x3.repeat(1, self.channel_6, 1, 1)
        LH_3_1_2 = LH_x1.unsqueeze(2).unsqueeze(2) * LH_x3
        LH_3_1_2 = self.pad1(LH_3_1_2)
        LH_3_1_2 = self.pad1(LH_3_1_2)[:,:,1:5,:]
        SCAF_3_1 = LH_3_1_1 + LH_3_1_2 + x1
        HL_x2 = F.adaptive_max_pool2d(x2, (1, 1))
        HL_x2 = self.sigmod(self.fc14(self.fc13(HL_x2.squeeze(2).squeeze(2))))
        HL_x1 = self.cov_32(x1)
        HL_x1 = HL_x1.mean(dim=1, keepdim=True)
        HL_x1 = HL_x1.repeat(1, self.channel_5, 1, 1)
        HL_3_2_1 = HL_x2.unsqueeze(2).unsqueeze(2) * HL_x1
        LH_x3 = x3.mean(dim=1, keepdim=True)
        LH_x3 = LH_x3.repeat(1, self.channel_5, 1, 1)
        LH_3_2_2 = HL_x2.unsqueeze(2).unsqueeze(2) * LH_x3
        LH_3_2_2 = self.pad1(LH_3_2_2)[:,:,1:3,:]
        SCAF_3_2 = HL_3_2_1 + LH_3_2_2 + x2
        HL_x3 = F.adaptive_max_pool2d(x3, (1, 1))
        HL_x3 = self.sigmod(self.fc16(self.fc15(HL_x3.squeeze(2).squeeze(2))))
        HL_x1 = self.cov_33(x1)
        HL_x1 = HL_x1.mean(dim=1, keepdim=True)
        HL_x1 = HL_x1.repeat(1, self.channel_7, 1, 1)
        HL_3_3_1 = HL_x3.unsqueeze(2).unsqueeze(2) * HL_x1
        HL_x2 = self.cov_34(x2)
        HL_x2 = HL_x2.mean(dim=1, keepdim=True)
        HL_x2 = HL_x2.repeat(1, self.channel_7, 1, 1)
        HL_3_3_2 = HL_x3.unsqueeze(2).unsqueeze(2) * HL_x2

        SCAF_3_3 = HL_3_3_1 + HL_3_3_2 + x3

        x1 = self.cov_38(SCAF_3_1)
        x2 = self.cov_39(SCAF_3_2)
        x3 = self.cov_40(SCAF_3_3)

        x1 = self.cov_42(x1)
        x2 = self.cov_43(x2)
        x3 = self.cov_44(x3)


        x2 = self.pad1(x2)
        x3 = self.pad1(x3)
        x3 = self.pad1(x3)[:,:,1:5,:]

        x_SCAF = x1 + x2 + x3
        x_SCAF = x_SCAF.squeeze(1)
        x_SCAF = x_SCAF.transpose(1,2)

        x = self.LNN(x_SCAF)

        # x = x.squeeze()

        x = self.fc(x)
        return x

if __name__ == '__main__':
    a=torch.tensor(range(4*1*4*12)).reshape((4,1,4,12)).float()
    model = HCLNNmodel_scaf_6()
    model(a)