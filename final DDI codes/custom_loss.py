import torch
from torch import nn
import torch.nn.functional as F


class SigmoidLoss(nn.Module):

    def forward(self, p_scores, n_scores, n2_scores):
        p_loss = - F.logsigmoid(p_scores).mean()
        n_loss = - F.logsigmoid(-n_scores).mean()
        n2_loss = - F.logsigmoid(-n2_scores).mean()

        return (0.5*(p_loss + n_loss) + 0.5*(p_loss + n2_loss))/ 2, p_loss, n_loss, n2_loss
        # return (p_loss + n_loss) / 2, p_loss, n_loss

