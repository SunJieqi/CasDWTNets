import torch
import torch.nn as nn
from DWT_Tools.DWT_IDWT_layer import *
from math import log


class Channel_pooling_fusion_layer(nn.Module):
    def __init__(self, inplanes, level,
                 wavename="haar"):
        super(Channel_pooling_fusion_layer, self).__init__()
        """
             Channel pooling fusion layer based on 1D DWT. 
        """
        # self.inchannel = 2 * level * inplanes
        # self.outchannel = inplanes
        self.inchannel = inplanes
        self.outchannel = int(inplanes / (2 * level))

        self.dwt = DWT_1D(wavename=wavename)
        self.level = level
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        t = int(abs(log(inplanes, 2) + 1) / 2)
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        # b,c,h,w ----> b,c,h * w -----> b,h * w,c
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(b, h * w, self.inchannel)
        for i in range(self.level):
            x, y = self.dwt(x)
            x = x.reshape(b, h, w, self.outchannel).permute(0, 3, 1, 2).contiguous()
            y = y.reshape(b, h, w, self.outchannel).permute(0, 3, 1, 2).contiguous()
            vector1 = self.conv(self.avg_pool(x).squeeze(-1).transpose(-1, -2))
            vector2 = self.conv(self.avg_pool(y).squeeze(-1).transpose(-1, -2))
            attention_vectors = torch.cat([vector1, vector2], dim=1)
            attention_vectors = self.softmax(attention_vectors)
            attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
            x_1 = x.unsqueeze_(dim=1)
            y_1 = y.unsqueeze_(dim=1)
            feas = torch.cat([x_1, y_1], dim=1)
            fea_v = (feas * attention_vectors).sum(dim=1)

            return fea_v


class CasDWTM(nn.Module):

    def __init__(self, planes, level=1, spatial_wavelet="bior2.2", channel_wavelet="haar"):
        super(CasDWTM, self).__init__()
        """
            Cascade wavelet transform module. 
        """
        self.dwt = DWT_2D(wavename=spatial_wavelet)
        self.Channel_pooling_fusion_layer = Channel_pooling_fusion_layer(2 * planes, level=1, wavename=channel_wavelet)
        self.fc = nn.Sequential(nn.Linear(planes, int(planes / 2)), nn.ReLU(True), nn.Linear(int(planes / 2), planes))
        self.softmax = nn.Softmax(dim=1)
        self.level = level
        self.planes = planes

    def forward(self, input):
        LL, LH, HL, HH = self.dwt(input)
        x_2 = torch.abs(LH).unsqueeze_(dim=1)
        x_3 = torch.abs(HL).unsqueeze_(dim=1)
        x_4 = torch.abs(HH).unsqueeze_(dim=1)
        feas = torch.cat([x_2, x_3, x_4], dim=1)
        vector2 = self.fc(LH.mean(-1).mean(-1)).unsqueeze_(dim=1)
        vector3 = self.fc(HL.mean(-1).mean(-1)).unsqueeze_(dim=1)
        vector4 = self.fc(HH.mean(-1).mean(-1)).unsqueeze_(dim=1)
        attention_vectors = torch.cat([vector2, vector3, vector4], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        out = torch.cat([fea_v + LL, LL], 1)
        x = self.Channel_pooling_fusion_layer(out)
        return x
