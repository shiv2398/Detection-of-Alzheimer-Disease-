import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision
import torchvision.models as models
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
class ResNet(nn.Module):
    def __init__(self,depth=18,num_classes=10):
        super(ResNet, self).__init__()
        self.wt50=wt50=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        self.wt34=wt34=torchvision.models.ResNet34_Weights.IMAGENET1K_V1
        self.wt101=wt101=torchvision.models.ResNet101_Weights.IMAGENET1K_V1
        self.wt152=wt152=torchvision.models.ResNet152_Weights.IMAGENET1K_V1
        if depth == 18:
            self.resnet = models.resnet18(pretrained=True)
        elif depth == 34:
            self.resnet = models.resnet34(weights=self.wt34)
        elif depth == 50:
            self.resnet = models.resnet50(weights=self.wt50)
        elif depth == 101:
            self.resnet = models.resnet101(weights=self.wt101)
        elif depth == 152:
            self.resnet = models.resnet152(weights=self.wt152)
        else:
            raise ValueError("Unsupported ResNet depth: {}".format(depth))
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    def forward(self, x):
        x = self.resnet(x)
        return x

