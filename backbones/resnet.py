import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.ops import deform_conv2d

BatchNorm2d = nn.BatchNorm2d

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# --- Deformable Conv2d Wrapper ---
class DeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, deformable_groups=1, bias=False):
        super(DeformConv2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # Weight initialization
        nn.init.kaiming_uniform_(self.weight, a=1)
        if bias:
            nn.init.constant_(self.bias, 0)

    def forward(self, x, offset, mask=None):
        return deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
            #deformable_groups=self.deformable_groups
        )

def constant_init(module, constant, bias=0):
    nn.init.constant_(module.weight, constant)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None):
        super(BasicBlock, self).__init__()
        self.with_dcn = dcn is not None
        self.with_modulated_dcn = False
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if self.with_dcn:
            fallback_on_stride = dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = dcn.get('modulated', False)

        if not self.with_dcn or fallback_on_stride:
            self.conv2 = conv3x3(planes, planes)
        else:
            deformable_groups = dcn.get('deformable_groups', 1)
            offset_channels = 27 if self.with_modulated_dcn else 18
            self.conv2_offset = nn.Conv2d(
                planes, deformable_groups * offset_channels,
                kernel_size=3, stride=stride, padding=1)
            self.conv2 = DeformConv2d(
                planes, planes, kernel_size=3, padding=1,
                deformable_groups=deformable_groups, bias=False)

        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if not self.with_dcn:
            out = self.conv2(out)
        else:
            offset_mask = self.conv2_offset(out)
            if self.with_modulated_dcn:
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, -9:, :, :].sigmoid()
                out = self.conv2(out, offset, mask)
            else:
                offset = offset_mask
                out = self.conv2(out, offset)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None):
        super(Bottleneck, self).__init__()
        self.with_dcn = dcn is not None
        self.with_modulated_dcn = False
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)

        if self.with_dcn:
            fallback_on_stride = dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = dcn.get('modulated', False)
        else:
            fallback_on_stride = False

        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            deformable_groups = dcn.get('deformable_groups', 1)
            offset_channels = 27 if self.with_modulated_dcn else 18
            self.conv2_offset = nn.Conv2d(
                planes, deformable_groups * offset_channels,
                kernel_size=3, stride=stride, padding=1)
            self.conv2 = DeformConv2d(
                planes, planes, kernel_size=3, padding=1,
                stride=stride, deformable_groups=deformable_groups, bias=False)

        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if not self.with_dcn:
            out = self.conv2(out)
        else:
            offset_mask = self.conv2_offset(out)
            if self.with_modulated_dcn:
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, -9:, :, :].sigmoid()
                out = self.conv2(out, offset, mask)
            else:
                offset = offset_mask
                out = self.conv2(out, offset)

        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, 
                 dcn=None): #, stage_with_dcn=(False, False, False, False)
        self.dcn = dcn
        # fixed stage_with_dcn [False, True, True, True]
        #self.stage_with_dcn = stage_with_dcn
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dcn=dcn)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dcn=dcn)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dcn=dcn)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
        #self.smooth = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=1)    

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if self.dcn is not None:
            for m in self.modules():
                if isinstance(m, (Bottleneck, BasicBlock)):
                    if hasattr(m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dcn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, dcn=dcn)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dcn=dcn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x2, x3, x4, x5


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['resnet18']), strict=False)
    return model

def deformable_resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2],
                    dcn=dict(modulated=True,
                            deformable_groups=1,
                            fallback_on_stride=False),
                   **kwargs) # stage_with_dcn=[False, True, True, True], 
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['resnet18']), strict=False)
    return model


def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['resnet34']), strict=False)
    return model


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['resnet50']), strict=False)
    return model


def deformable_resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model with deformable conv.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   dcn=dict(modulated=True,
                            deformable_groups=1,
                            fallback_on_stride=False),
                   **kwargs) # stage_with_dcn=[False, True, True, True], 
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['resnet50']), strict=False)
    return model


def resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['resnet101']), strict=False)
    return model


def resnet152(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['resnet152']), strict=False)
    return model
    
    

if __name__ == "__main__":
    # Test input
    dummy_input = torch.randn(4, 3, 224, 224)  # batch size 4, image size 224x224

    # Test resnet50 (standard)
    model = resnet50(pretrained=False)
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_input)
    print("\nStandard ResNet50 outputs:")
    for i, out in enumerate(outputs, start=2):
        print(f"x{i}.shape = {out.shape}")

    # Test deformable_resnet50 (should same output as before)
    model = deformable_resnet50(pretrained=False)
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_input)
    print("\nDeformable ResNet50 outputs:")
    for i, out in enumerate(outputs, start=2):
        print(f"x{i}.shape = {out.shape}")

