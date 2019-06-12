import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual
    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, activate_before_residual))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, dropRate=0.1):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        self.nChannels = 2 * nChannels[3]
        self.route1 = nn.Sequential(nn.Conv2d(1, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False),
                               NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True), 
                            NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate), 
                            NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate), 
                            nn.BatchNorm2d(nChannels[3], momentum=0.001), 
                            nn.LeakyReLU(negative_slope=0.1, inplace=True), 
                            nn.AdaptiveAvgPool2d(1))
        self.route2 = nn.Sequential(nn.Conv2d(1, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False),
                            nn.AvgPool2d(1, stride=(6,1)),
                            NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True), 
                            NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate), 
                            NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate), 
                            nn.BatchNorm2d(nChannels[3], momentum=0.001), 
                            nn.LeakyReLU(negative_slope=0.1, inplace=True), 
                            nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(self.nChannels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        x1, x2 = torch.split(x, [80, x.shape[2] - 80], dim=2)
        out1 = self.route1(x1)
        out2 = self.route2(x2)
        out = torch.cat([out1, out2], dim=1)
        out = out.view(-1, self.nChannels)
        return self.fc(out)