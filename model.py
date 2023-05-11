import torch.nn as nn
import torch
from torchvision.models.resnet import resnet34 as _resnet34
import torch.onnx

def _make_resnet_backbone(resnet):
    pretrained = nn.Module()
    pretrained.conv0_0 = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.layer1
    )

    pretrained.conv1_0 = resnet.layer2
    pretrained.conv2_0 = resnet.layer3
    pretrained.conv3_0 = resnet.layer4

    # for param in pretrained.parameters():
    #    param.requires_grad = True

    return pretrained


def resnet34(pretrained=True, **kwargs):
    """# This docstring shows up in hub.help()
    Resnet18 model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load pretrained weights
    model = _resnet34(pretrained=pretrained, **kwargs)
    return model


class Upscale_Block(nn.Module):
    def __init__(self, input_channel, out):
        super().__init__()
        self.cv1 = nn.Conv2d(
            input_channel, out, 3, stride=(1, 1), padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out)
        self.cv2 = nn.Conv2d(out, out, 3, stride=(1, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out)
        # self.upscale = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.relu = nn.ReLU(inplace=True)

        self.cv3 = nn.Conv2d(out, out, 3, stride=(1, 1), padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out)
        # self.cv4 = nn.Conv2d(out, out, 3, stride=(2, 2), padding=1, bias=False)
        self.cv4 = nn.Conv2d(out, out, 3, stride=(1, 1), padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(out)


    def forward(self, x):
        out = self.cv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.cv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.cv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.cv4(out)
        out = self.bn4(out)
        return out


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class NestedUNet(nn.Module):
    def __init__(self, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [64, 128, 256, 512, 1024]

        model = resnet34()
        self.encoder = _make_resnet_backbone(model)

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.conv0_0 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv4_0 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = Upscale_Block(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv0_2 = VGGBlock(
            nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0]
        )
        self.conv1_2 = VGGBlock(
            nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1]
        )
        self.conv2_2 = Upscale_Block(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = VGGBlock(
            nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0]
        )
        self.conv1_3 = Upscale_Block(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = Upscale_Block(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])

    def forward(self, input):
        # input_layer = self.conv0_0(input)
        x0_0 = self.encoder.conv0_0(input)
        x1_0 = self.encoder.conv1_0(x0_0)
        x2_0 = self.encoder.conv2_0(x1_0)
        x3_0 = self.encoder.conv3_0(x2_0)
        x4_0 = self.conv4_0(x3_0)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        out = self.up(x0_4)
        out = self.final(out)
        return out
