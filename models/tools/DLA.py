import torch
import torch.nn.functional as F
import torch.nn as nn


class LocalAffinity(nn.Module):

    def __init__(self, dilations=[1]):
        super(LocalAffinity, self).__init__()
        self.dilations = dilations
        weight = self._init_aff()
        self.register_buffer('kernel', weight)

    def _init_aff(self):
        # initialising the shift kernel
        weight = torch.zeros(8, 1, 3, 3, device='cuda')

        weight[0, 0, 0, 0] = 1
        weight[1, 0, 0, 1] = 1
        weight[2, 0, 0, 2] = 1

        weight[3, 0, 1, 0] = 1
        weight[4, 0, 1, 2] = 1

        weight[5, 0, 2, 0] = 1
        weight[6, 0, 2, 1] = 1
        weight[7, 0, 2, 2] = 1

        self.weight_check = weight.clone()
        return weight

    def forward(self, x):
        self.weight_check = self.weight_check.type_as(x)
        assert torch.all(self.weight_check.eq(self.kernel))

        B, K, H, W = x.size()
        x = x.view(B * K, 1, H, W)

        x_affs = []
        for d in self.dilations:
            x_pad = F.pad(x, [d] * 4, mode='replicate')
            x_aff = F.conv2d(x_pad, self.kernel, dilation=d)
            x_affs.append(x_aff)

        x_aff = torch.cat(x_affs, 1)
        return x_aff.view(B, K, -1, H, W)


class LocalAffinityCopy(LocalAffinity):

    def _init_aff(self):
        # initialising the shift kernel
        weight = torch.zeros(8, 1, 3, 3, device='cuda')

        weight[0, 0, 0, 0] = 1
        weight[1, 0, 0, 1] = 1
        weight[2, 0, 0, 2] = 1

        weight[3, 0, 1, 0] = 1
        weight[4, 0, 1, 2] = 1

        weight[5, 0, 2, 0] = 1
        weight[6, 0, 2, 1] = 1
        weight[7, 0, 2, 2] = 1

        self.weight_check = weight.clone()
        return weight


class LocalAffinityAbs(LocalAffinity):

    def forward(self, x):
        x = super(LocalAffinityAbs, self).forward(x)
        return torch.abs(x)


class DLAModule(nn.Module):
    def __init__(self, num_iter=5, dilations=[1]):
        # Dilated Local Affinity
        super(DLAModule, self).__init__()
        self.num_iter = num_iter
        self.aff_loc = LocalAffinity(dilations)

    def forward(self, feature):
        # feature: [BxCxHxW]
        n, c, h, w = feature.size()

        for _ in range(self.num_iter):
            f = self.aff_loc(feature)  # [BxCxPxHxW]
            # dim2 represent the p neighbor-pixels' value

            abs = torch.abs(feature.unsqueeze(2) - f)
            aff = torch.exp(-torch.mean(abs, dim=1, keepdim=True))
            # aff = F.cosine_similarity(feature.unsqueeze(2), f, dim=1).unsqueeze(1)

            aff = aff/torch.sum(aff, dim=2, keepdim=True)  # [Bx1xPxHxW]
            # print(aff[0, 0, :, h//2, w//2])
            # dim2 represent the p neighbor-pixels' affinity

            feature = torch.sum(f * aff, dim=2)

        return feature


class RefineModule(nn.Module):
    def __init__(self, num_iter=10, dilations=[1]):
        super(RefineModule, self).__init__()

        self.num_iter = num_iter
        self.aff_abs = LocalAffinityAbs(dilations)
        self.aff_loc = LocalAffinityCopy(dilations)

    def forward(self, x, mask):
        # x: [BxKxHxW]
        # mask: [BxCxHxW]
        B, K, H, W = x.size()
        _, C, _, _ = mask.size()

        abs = self.aff_abs(x)
        aff = torch.exp(-torch.mean(abs, dim=1, keepdim=True))

        for _ in range(self.num_iter):
            m = self.aff_loc(mask)  # [BxCxPxHxW]
            mask = torch.sum(m * aff, dim=2)/torch.sum(aff, dim=2)

        return mask


if __name__ == '__main__':
    feature = torch.randn([4, 64, 256, 256], device='cuda')

    model = DLAModule()
    output = model(feature)
    print(output.shape)