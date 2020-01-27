import torch


class ReconstructionLoss(torch.nn.Module):

    def __init__(self, coarse_hole_alpha, coarse_nohole_alpha, refine_hole_alpha, refine_nohole_alpha):
        super(ReconstructionLoss, self).__init__()
        self.coarse_hole_alpha = coarse_hole_alpha
        self.coarse_nohole_alpha = coarse_nohole_alpha
        self.refine_hole_alpha = refine_hole_alpha
        self.refine_nohole_alpha = refine_nohole_alpha

    def forward(self, image, coarse, refined, mask):
        mask_flat = mask.view(mask.size(0), -1)
        loss_a = self.coarse_hole_alpha * torch.mean(torch.abs(image - coarse) * mask / mask_flat.mean(1).view(-1, 1, 1, 1))
        loss_b = self.coarse_nohole_alpha * torch.mean(torch.abs(image - coarse) * (1 - mask) / (1 - mask_flat.mean(1).view(-1, 1, 1, 1)))
        loss_c = self.refine_hole_alpha * torch.mean(torch.abs(image - refined) * mask / mask_flat.mean(1).view(-1, 1, 1, 1))
        loss_d = self.refine_nohole_alpha * torch.mean(torch.abs(image - refined) * (1 - mask) / (1 - mask_flat.mean(1).view(-1, 1, 1, 1)))
        return loss_a + loss_b + loss_c + loss_d
