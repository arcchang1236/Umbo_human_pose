import torch
import numpy as np

# class JointsMSELoss(torch.nn.Module):
#     def __init__(self):
#         super(JointsMSELoss, self).__init__()
#         self.criterion = torch.nn.MSELoss().cuda()

#     def forward(self, output, target):
#         num_joints = output.size(0)
#         loss = torch.Tensor([0.])

#         for idx in range(num_joints):
#             if output[idx] == 0:
#                 continue
#             loss += self.criterion(output[idx], target[idx]/100.0)

#         return loss / num_joints


def MPJPE(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))