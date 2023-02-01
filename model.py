import torch
import torch.nn as nn
import torch.nn.functional as F

class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)

class BasicBlock_encoder(nn.Module):
    def __init__(self, chann=16,
                 dropprob=0.35, kernel=3, dilated=1, encoder_stage=False):
        super().__init__()

        # self.encoder_stage = encoder_stage

        self.conv3x3_1 = nn.Conv2d(chann, chann, (3, 3), stride=1, padding=(1, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x3_2 = nn.Conv2d(chann, chann, (3, 3), stride=1, padding=(1, 1), bias=True)
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

    def forward(self, input):

        output = self.conv3x3_1(input)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x3_2(output)
        output = self.bn2(output)

        output = F.relu(output + input)

        return output

class non_bottleneck_1d_2(nn.Module):
    def __init__(self, chann, dropprob, kernel, dilated, encoder_stage=False, last=True):
        super().__init__()

        self.encoder_stage = encoder_stage

        if kernel==3:
            self.conv1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)
            self.conv1_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

            self.conv2_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                       dilation=(dilated, 1))
            self.conv2_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                       dilation=(1, dilated))
        elif kernel==5:
            self.conv1_1 = nn.Conv2d(chann, chann, (5, 1), stride=1, padding=(2, 0), bias=True)
            self.conv1_2 = nn.Conv2d(chann, chann, (1, 5), stride=1, padding=(0, 2), bias=True)

            self.conv2_1 = nn.Conv2d(chann, chann, (5, 1), stride=1, padding=(2 * dilated, 0), bias=True,
                                       dilation=(dilated, 1))
            self.conv2_2 = nn.Conv2d(chann, chann, (1, 5), stride=1, padding=(0, 2 * dilated), bias=True,
                                       dilation=(1, dilated))
        elif kernel==7:
            self.conv1_1 = nn.Conv2d(chann, chann, (7, 1), stride=1, padding=(3, 0), bias=True)
            self.conv1_2 = nn.Conv2d(chann, chann, (1, 7), stride=1, padding=(0, 3), bias=True)

            self.conv2_1 = nn.Conv2d(chann, chann, (7, 1), stride=1, padding=(3 * dilated, 0), bias=True,
                                       dilation=(dilated, 1))
            self.conv2_2 = nn.Conv2d(chann, chann, (1, 7), stride=1, padding=(0, 3 * dilated), bias=True,
                                       dilation=(1, dilated))

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
        self.last = last

    def forward(self, input):
        output = self.conv1_1(input)
        if self.encoder_stage:
            output = F.relu(output)

        output = self.conv1_2(output)
        output = self.bn1(output)
        if self.encoder_stage:
            output = F.relu(output)

        output = self.conv2_1(output)
        if self.encoder_stage:
            output = F.relu(output)
        output = self.conv2_2(output)
        if self.last == False:
            output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        if self.encoder_stage:
            return F.relu(output + input)  # +input = identity (residual connection)
        else:
            return output       ########## lk 2021.02.05

################################### Encoder corresponding to ResNet-34 ###################################
class Encoder_v0_762(nn.Module):
    def __init__(self, num_classes=1,
                 dp=DownsamplerBlock,
                 block=BasicBlock_encoder,
                 channels=[3, 16, 64, 128, 256],
                 dropprob=0.1, rates=[1, 1, 1, 1, 1, 1],
                 predict=False):
        super().__init__()

        self.predict = predict
        self.stage_1_0 = dp(channels[0], channels[1])

        self.stage_2_0 = dp(channels[1], channels[2])
        self.stage_2_1 = block(channels[2], dropprob, rates[0])
        self.stage_2_2 = block(channels[2], dropprob, rates[1])
        self.stage_2_3 = block(channels[2], dropprob, rates[2])
        self.stage_2_4 = block(channels[2], dropprob, rates[3])
        self.stage_2_5 = block(channels[2], dropprob, rates[1])
        self.stage_2_6 = block(channels[2], dropprob, rates[2])
        self.stage_2_7 = block(channels[2], dropprob, rates[3])

        self.stage_3_0 = dp(channels[2], channels[3])
        self.stage_3_1 = block(channels[3], dropprob, rates[1])
        self.stage_3_2 = block(channels[3], dropprob, rates[2])
        self.stage_3_3 = block(channels[3], dropprob, rates[3])
        self.stage_3_4 = block(channels[3], dropprob, rates[1])
        self.stage_3_5 = block(channels[3], dropprob, rates[2])
        self.stage_3_6 = block(channels[3], dropprob, rates[3])

        self.stage_4_0 = dp(channels[3], channels[4])
        self.stage_4_1 = block(channels[4], dropprob, rates[4])
        self.stage_4_2 = block(channels[4], dropprob, rates[5])

    def forward(self, input):
        stage_1_0 = self.stage_1_0(input)

        stage_2_0 = self.stage_2_0(stage_1_0)
        stage_2_1 = self.stage_2_1(stage_2_0)
        stage_2_2 = self.stage_2_2(stage_2_1)
        stage_2_3 = self.stage_2_3(stage_2_2)
        stage_2_4 = self.stage_2_4(stage_2_3)
        stage_2_5 = self.stage_2_5(stage_2_4)
        stage_2_6 = self.stage_2_6(stage_2_5)
        stage_2_last = self.stage_2_7(stage_2_6)

        stage_3_0 = self.stage_3_0(stage_2_last)
        stage_3_1 = self.stage_3_1(stage_3_0)
        stage_3_2 = self.stage_3_2(stage_3_1)
        stage_3_3 = self.stage_3_3(stage_3_2)
        stage_3_4 = self.stage_3_4(stage_3_3)
        stage_3_5 = self.stage_3_5(stage_3_4)
        stage_3_last = self.stage_3_6(stage_3_5)

        stage_4_0 = self.stage_4_0(stage_3_last)
        stage_4_1 = self.stage_4_1(stage_4_0)
        stage_4_last = self.stage_4_2(stage_4_1)

        output = [stage_1_0, stage_2_last, stage_3_last, stage_4_last]

        return output

################################### CarNet corresponding to ResNet-34 ####################################
class CarNet34(nn.Module):
    def __init__(self,
                 Encoder=Encoder_v0_762,
                 dp=DownsamplerBlock,
                 block=BasicBlock_encoder,
                 num_classes=1,
                 channels=[3, 16, 64, 128, 256],
                 dropprob=[0, 0],
                 rates=[1, 1, 1, 1, 1, 1],
                 kernels=[3, 3, 3],
                 predict=False,
                 decoder_block=non_bottleneck_1d_2):
        super().__init__()

        self.encoder = Encoder(num_classes=num_classes, dp=dp, block=block,
                               channels=channels, dropprob=dropprob[0], rates=rates, predict=predict)

        compress_channels = 32  ### channels[2] // 2

        self.conv1x1_1 = nn.Conv2d(channels[4], compress_channels, (1, 1))
        self.deconv_1 = nn.ConvTranspose2d(compress_channels, compress_channels, kernel_size=3,
                                           stride=4, padding=1, output_padding=3, bias=True)

        self.conv1x1_2 = nn.Conv2d(channels[3], compress_channels, (1, 1))
        self.deconv_2 = nn.ConvTranspose2d(compress_channels, compress_channels, kernel_size=3,
                                           stride=2, padding=1, output_padding=1, bias=True)

        self.conv1x1_3 = nn.Conv2d(channels[2], compress_channels, (1, 1))

        self.conv_1 = decoder_block(compress_channels, dropprob=dropprob[1], kernel=kernels[1],
                                          dilated=rates[0], encoder_stage=False, last=True)

        self.conv1x1_4 = nn.Conv2d(compress_channels, num_classes, (1, 1))
        self.deconv_3 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3,
                                           stride=4, padding=1, output_padding=3, bias=True)
        self.conv_2 = decoder_block(num_classes, dropprob=dropprob[1], kernel=kernels[2],
                                          dilated=rates[0], encoder_stage=False, last=True)

    def forward(self, input):

        encoder = self.encoder(input)
        ### 不同阶段的输出
        stage_1, stage_2, stage_3, stage_4 = (encoder)[0],  (encoder)[1], (encoder)[2], (encoder)[3]

        stage_4 = self.conv1x1_1(stage_4)
        s_4to2 = self.deconv_1(stage_4)

        stage_3 = self.conv1x1_2(stage_3)
        s_3to2 = self.deconv_2(stage_3)

        s2 = self.conv1x1_3(stage_2)
        s_4_3_2 = self.conv_1(s_4to2 + s_3to2 + s2)

        s_4_3_2 = self.conv1x1_4(s_4_3_2)
        s = self.deconv_3(s_4_3_2)
        s = self.conv_2(s)

        return s


