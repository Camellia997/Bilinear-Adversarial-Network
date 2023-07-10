# ------------------------------------------------------------------------------
# Modified based on torchvision.models.resnet.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torchvision import models

__all__ = ['d', 'm', 'bcnn_d', 'bcnn_m', 'bcnn_dd', 'bcnn_mm']



class d(nn.Module):
    def __init__(self, pretrained=False):
        nn.Module.__init__(self)
        self.model = models.vgg16(pretrained=pretrained)
        self.features = nn.Sequential(*list(self.model.features.children())[:-1])

    def forward(self, x):
        x = self.features(x)

        return x, x, x

    @property
    def out_features(self):
        return 512

class m(nn.Module):
    def __init__(self, pretrained=False):
        nn.Module.__init__(self)
        self.model = models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self._out_features = self.model.fc.in_features

    def forward(self, x):
        x = self.features(x)

        return x, x, x

    @property
    def out_features(self):
        return self._out_features

class bcnn_m(nn.Module):
    def __init__(self, pretrained=False):
        nn.Module.__init__(self)
        self.model = models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self._out_features = self.model.fc.in_features
    
    def forward(self, x):
        x = self.features(x)
        batch_size, feature_dim, feature_size = x.size(0), x.size(1), x.size(2) * x.size(3)
        x = x.view(batch_size, feature_dim, feature_size)
        x = torch.bmm(x, torch.transpose(x, 1, 2)) / feature_size
        # x = x.view(batch_size, -1)
        # x = torch.nn.functional.normalize(torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10))
        x = x.view(batch_size, feature_dim, feature_dim, 1)

        """Bilinear
        torch.transpose(x, 1, 2) - exchange the dimension 1 and 2 of X (dimension is 0, 1, 2)
        torch.bmm - multiplication of matrix, (N, 512, 7**2) * (N, 7**2, 512) -> (N, 512, 512) ((p, m, n) * (p, n, a) -> (p, m, a))
        """

        return x, x, x

    @property
    def out_features(self):
        return self._out_features

class bcnn_d(nn.Module):
    def __init__(self, pretrained=False):
        nn.Module.__init__(self)
        self.features = models.vgg16(pretrained=pretrained).features
        self.features = nn.Sequential(*list(self.features.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        batch_size, feature_dim, feature_size = x.size(0), x.size(1), x.size(2) * x.size(3)
        x = x.view(batch_size, feature_dim, feature_size)
        x = torch.bmm(x, torch.transpose(x, 1, 2)) / feature_size
        # x = x.view(batch_size, -1)
        # x = torch.nn.functional.normalize(torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10))
        x = x.view(batch_size, feature_dim, feature_dim, 1)

        return x, x, x
        
    @property
    def out_features(self):
        return 512

class bcnn_mm(nn.Module):
    def __init__(self, pretrained=False):
        nn.Module.__init__(self)
        self.a = models.resnet50(pretrained=pretrained)
        self.b = models.resnet50(pretrained=pretrained)
        self.features1 = nn.Sequential(*list(self.a.children())[:-2])
        self.features2 = nn.Sequential(*list(self.b.children())[:-2])
    
    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        batch_size, feature_dim, feature_size = x1.size(0), x1.size(1), x1.size(2) * x1.size(3)
        x1 = x1.view(batch_size, feature_dim, feature_size)
        x2 = x2.view(batch_size, feature_dim, feature_size)
        x = torch.bmm(x1, torch.transpose(x2, 1, 2)) / feature_size
        # x = x.view(batch_size, -1)
        # x = torch.nn.functional.normalize(torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10))
        x = x.view(batch_size, feature_dim, feature_dim, 1)
        x1 = x1.view(batch_size, feature_dim, feature_size, 1)
        x2 = x2.view(batch_size, feature_dim, feature_size, 1)

        return x, x1, x2

    @property
    def out_features(self):
        return self.a.fc.in_features

class bcnn_dd(nn.Module):
    def __init__(self, pretrained=False):
        nn.Module.__init__(self)
        self.a = models.vgg16(pretrained=pretrained)
        self.b = models.vgg16(pretrained=pretrained)
        self.features1 = nn.Sequential(*list(self.a.features.children())[:-1])
        self.features2 = nn.Sequential(*list(self.b.features.children())[:-1])
    
    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        batch_size, feature_dim, feature_size = x1.size(0), x1.size(1), x1.size(2) * x1.size(3)
        x1 = x1.view(batch_size, feature_dim, feature_size)
        x2 = x2.view(batch_size, feature_dim, feature_size)
        x = torch.bmm(x1, torch.transpose(x2, 1, 2)) / feature_size
        # x = x.view(batch_size, -1)
        # x = torch.nn.functional.normalize(torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10))
        x = x.view(batch_size, feature_dim, feature_dim, 1)
        x1 = x1.view(batch_size, feature_dim, feature_size, 1)
        x2 = x2.view(batch_size, feature_dim, feature_size, 1)

        return x, x1, x2

    @property
    def out_features(self):
        return 512

# class bcnn_md(nn.Module):
#     def __init__(self, pretrained=False):
#         nn.Module.__init__(self)
#         self.m = models.resnet50(pretrained=pretrained)
#         self.mfeatures = nn.Sequential(*list(self.m.children())[:-2])
#         self.d = models.vgg16(pretrained=pretrained)
#         self.dfeatures = nn.Sequential(*list(self.d.features.children())[:-1])

#     def forward(self, x):
#         x_m = self.mfeatures(x)
#         x_d = self.dfeatures(x)
#         batch_size = x.size(0)
#         feature_dim_m, feature_size_m = x_m.size(1), x_m.size(2) * x_m.size(3)
#         feature_dim_d, feature_size_d = x_d.size(1), x_d.size(2) * x_d.size(3)

#         x_m = x.view(batch_size , feature_dim_m, feature_size_m)
#         x_d = x.view(batch_size , feature_dim_d, feature_size_d)
#         x = (torch.bmm(x_m, torch.transpose(x_d, 1, 2)) / torch.sqrt(feature_size_m * feature_size_d)).view(batch_size, -1)
#         x = torch.nn.functional.normalize(torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10))
#         x = x.view(N, feature_dim_m, feature_dim_d, 1)

#         return x
        
#     @property
#     def out_features(self):
#         return self.m.fc.in_features

