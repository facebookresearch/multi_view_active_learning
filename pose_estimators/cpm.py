import torch
import torch.manifold.patch
from torch.hub import load_state_dict_from_url
from torchvision import models

from .pose_estimator import PoseEstimator


class ConvolutionalPoseMachines(PoseEstimator):
    """
    Implementation of the CPM model in this paper:
    Wei, Shih-En, et al. "Convolutional pose machines." CVPR. 2016.
    """

    def load_pretrained_weights(self, path_to_weights):
        if path_to_weights == "default":
            state_dict = load_state_dict_from_url(
                "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"
            )
            self.vgg19.backbone.load_state_dict(state_dict)
        else:
            raise NotImplementedError()

    def __init__(self, num_joints, load_pretrained_vgg=True):
        super().__init__(num_joints=num_joints)

        self.vgg19 = VGG19(pretrained=load_pretrained_vgg)  # backbone
        self.stage1 = FirstStage(128, self.num_joints)
        self.stage2 = Stage(self.num_joints + 128, self.num_joints)
        self.stage3 = Stage(self.num_joints + 128, self.num_joints)
        self.stage4 = Stage(self.num_joints + 128, self.num_joints)
        self.stage5 = Stage(self.num_joints + 128, self.num_joints)
        self.stage6 = Stage(self.num_joints + 128, self.num_joints)

    def forward(self, x):
        features = self.vgg19(x)

        stage1 = self.stage1(features)
        stage2 = self.stage2(torch.cat([features, stage1], dim=1))
        stage3 = self.stage3(torch.cat([features, stage2], dim=1))
        stage4 = self.stage4(torch.cat([features, stage3], dim=1))
        stage5 = self.stage5(torch.cat([features, stage4], dim=1))
        stage6 = self.stage6(torch.cat([features, stage5], dim=1))

        return torch.stack([stage1, stage2, stage3, stage4, stage5, stage6], dim=1)

    def compute_batch_loss(self, data):
        images_batch = data["images"].cuda()
        channel, w, h = (
            images_batch.shape[2],
            images_batch.shape[3],
            images_batch.shape[4],
        )
        images_batch = images_batch.reshape([-1, channel, w, h])
        # BATCH_SIZE * NUM_CAMS, 6, 19, W, H
        heatmaps = self.forward(images_batch)
        gt_heatmap = data["gt_heatmap"].cuda()
        # 6 stages for CPM.
        channel, w, h = gt_heatmap.shape[2], gt_heatmap.shape[3], gt_heatmap.shape[4]
        gt_heatmap = gt_heatmap.reshape([-1, channel, w, h])
        gt_heatmap = torch.stack([gt_heatmap] * 6, dim=1)
        batch_loss = self.loss.pose_2d_mse(heatmaps, gt_heatmap)
        return batch_loss


class Stage(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Stage, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_dim, 128, kernel_size=7, padding=3)
        self.conv2 = torch.nn.Conv2d(128, 128, kernel_size=7, padding=3)
        self.conv3 = torch.nn.Conv2d(128, 128, kernel_size=7, padding=3)
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=7, padding=3)
        self.conv5 = torch.nn.Conv2d(128, 128, kernel_size=7, padding=3)
        self.conv6 = torch.nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.conv7 = torch.nn.Conv2d(128, output_dim, kernel_size=1, padding=0)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x), inplace=True)
        x = torch.nn.functional.relu(self.conv2(x), inplace=True)
        x = torch.nn.functional.relu(self.conv3(x), inplace=True)
        x = torch.nn.functional.relu(self.conv4(x), inplace=True)
        x = torch.nn.functional.relu(self.conv5(x), inplace=True)
        x = torch.nn.functional.relu(self.conv6(x), inplace=True)
        x = self.conv7(x)
        return x


class FirstStage(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FirstStage, self).__init__()
        self.stage1_1 = torch.nn.Conv2d(input_dim, 512, kernel_size=1, padding=0)
        self.stage1_2 = torch.nn.Conv2d(512, output_dim, kernel_size=1, padding=0)

    def forward(self, x):
        x = torch.nn.functional.relu(self.stage1_1(x), inplace=True)
        x = self.stage1_2(x)
        return x


class VGG19(torch.nn.Module):
    def __init__(self, pretrained=False):
        super(VGG19, self).__init__()
        self.backbone = models.vgg19(pretrained=pretrained).features[:27]
        self.conv5_1 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = torch.nn.Conv2d(512, 128, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.nn.functional.relu(self.conv5_1(x), inplace=True)
        x = torch.nn.functional.relu(self.conv5_2(x), inplace=True)
        x = torch.nn.functional.relu(self.conv5_3(x), inplace=True)
        return x
