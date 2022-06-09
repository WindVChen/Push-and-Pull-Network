import torch.nn as nn
import torch
import pretrainedmodels


class PPNet(nn.Module):
    """
    Build a P^2Net model.
    """

    def __init__(self, base_encoder: str, dim=2048, pred_dim=512, pretrained=True, baseline=True, num_classes=23, proxy_per_cls=3):
        """
        :param base_encoder: backbone architecture like "resnet50", "densenet121", and so on
        :param dim: feature dimension, set to 2048
        :param pred_dim: feature dimension, set to 512
        :param pretrained: whether use pretrained model, suggested to set to True
        :param baseline: whether train/inference on just the baseline model
        :param num_classes: number of subclasses, e.g. for FGSC-23, it should be set to 23
        :param proxy_per_cls: number of proxies per subclass, e.g. for FGSC-23, the optimal is 3
        """
        super(PPNet, self).__init__()

        # create the encoder
        if pretrained:
            print("use pretrained model!")
            self.encoder = pretrainedmodels.__dict__[base_encoder](pretrained='imagenet')
        else:
            self.encoder = pretrainedmodels.__dict__[base_encoder](pretrained=None)

        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])

        if baseline:
            print("Train on baseline!")
            self.classify_fc = nn.Linear(512 * 4, num_classes)
        else:
            self.classify_fc = nn.Linear(512, num_classes)

        self.avg_push = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                      nn.BatchNorm1d(pred_dim, affine=False),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(pred_dim, dim, bias=False),
                                      nn.BatchNorm1d(dim, affine=False),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(dim, pred_dim, bias=False),
                                      nn.BatchNorm1d(pred_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(pred_dim, dim))

        self.max_push = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                      nn.BatchNorm1d(pred_dim, affine=False),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(pred_dim, dim, bias=False),
                                      nn.BatchNorm1d(dim, affine=False),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(dim, pred_dim, bias=False),
                                      nn.BatchNorm1d(pred_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(pred_dim, dim))

        self.pull = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.prototype = prototype(512, proxy_per_cls*num_classes)

    def forward(self, x1, x2, baseline=True):
        """
        :param x1: first augmented views
        :param x2: second augmented views
        :param baseline: whether train/inference on just the baseline model
        :return: outputs of the last classification layer, outputs of the dual-branch push-out part and their flipped
        ones,  outputs of the pull-in part and the proxeis
        """
        if not baseline:
            # compute features for one view
            z1 = self.encoder(x1)
            z2 = self.encoder(x2)

            m1 = self.max_push(torch.flatten(self.maxpool(z1), 1))
            m2 = self.max_push(torch.flatten(self.maxpool(z2), 1))

            a1 = self.avg_push(torch.flatten(self.avgpool(z1), 1))
            a2 = self.avg_push(torch.flatten(self.avgpool(z2), 1))

            pt1 = torch.flatten(self.avgpool(self.pull(z1)), 1)
            pt2 = torch.flatten(self.avgpool(self.pull(z2)), 1)

            t1 = self.classify_fc(pt1)
            t2 = self.classify_fc(pt2)

            return t1, t2, [m1, torch.flip(m1, [0])], [m2, torch.flip(m2, [0])], \
                   [a1, torch.flip(a1, [0])], [a2, torch.flip(a2, [0])], [m1.detach(), m2.detach()], [a1.detach(),
                                                                                                      a2.detach()], [
                       self.prototype.prototype, pt1, pt2]

        else:
            '''baseline'''
            z1 = self.encoder(x1)
            z1 = self.avgpool(z1)
            z1 = z1.view(z1.size(0), -1)
            t1 = self.classify_fc(z1)

            return t1, None, None, None, None, None, None, None


class emptyLayer(nn.Module):
    def __init__(self):
        super(emptyLayer, self).__init__()

    def forward(self, x):
        return x


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class prototype(nn.Module):
    def __init__(self, dim, num=23):
        super(prototype, self).__init__()
        protos = torch.nn.Parameter(torch.empty([num, dim]))
        self.prototype = torch.nn.init.xavier_uniform_(protos, gain=1)
