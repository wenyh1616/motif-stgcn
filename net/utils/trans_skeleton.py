import torch
import torch.nn as nn


class TransSkeleton(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):

        N, C, T, V = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(N * T, V, C)
        x_t = x.permute(0, 2, 1).contiguous()

        vec_prod = torch.matmul(x, x_t)
        sum_sqa = torch.sum(torch.mul(x, x), dim=2)
        sum_sqa_ex = sum_sqa.repeat(1, 1, vec_prod.shape[2]).view(-1, V, V)
        sum_sqb_ex = sum_sqa.repeat(1, 1, vec_prod.shape[1]).view(-1, V, V)
        sum_sqb_ex = sum_sqb_ex.permute(0, 2, 1).contiguous()
        sq_ed = sum_sqb_ex + sum_sqa_ex - 2 * vec_prod
        zero_vec = torch.zeros_like(sq_ed)
        ed = torch.where(sq_ed > 0, sq_ed, zero_vec)

        ed = ed.view(N, T, V * V).mean(1)
        max_ed, ind = torch.max(ed, 1)
        max_ed = max_ed.view(N, -1)
        ed = max_ed - ed
        ed = ed.view(N, V, V)

        x_diag = torch.sum(ed, 1)
        x_diag = x_diag.repeat(1, V).view(-1, V, V)
        ed = torch.div(ed, x_diag)

        ed = ed.view(N, V, V)

        return ed.contiguous()
