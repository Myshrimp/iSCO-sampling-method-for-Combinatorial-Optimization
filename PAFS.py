import torch
import numpy as np
import  torch.nn as nn
from tqdm import tqdm
import time
class BaseSampler():
    def __init__(self, args, U=1, log_g=None, seed=0, device=torch.device("cpu")):
        self.U = U
        self.ess_ratio = 1#args.ess_ratio
        self.log_g = log_g
        self.device = device
        self.rng = np.random.default_rng(seed)
        self._steps = 0
        self._lens = []
        self._accs = []
        self._hops = []

    def step(self, x, model):
        raise NotImplementedError

    @property
    def accs(self):
        return self._accs[-1]

    @property
    def hops(self):
        return self._hops[-1]

    @property
    def lens(self):
        return self._lens[-1]

    @property
    def avg_lens(self):
        ratio = self.ess_ratio
        return sum(self._lens[int(self._steps * (1 - ratio)):]) / int(self._steps * ratio)

    @property
    def avg_accs(self):
        ratio = self.ess_ratio
        #print('ratio:',ratio,'steps:',self._steps)
        return sum(self._accs[int(self._steps * (1 - ratio)):]) / int(self._steps * ratio)

    @property
    def avg_hops(self):
        ratio = self.ess_ratio
        return sum(self._hops[int(self._steps * (1 - ratio)):]) / int(self._steps * ratio)


class PathAuxiliaryFastSampler(BaseSampler):
    def __init__(self, args=None, U=1, log_g=None, seed=0, device=torch.device("cpu")):
        super().__init__(args, U, log_g, seed, device)


    def step(self, x, model):
        R = int(self.rng.integers(1, 2 * self.U, 1))

        bsize = x.shape[0]
        x_rank = len(x.shape) - 1

        x = x.requires_grad_()
        #print('x fed:',x)
        score_x = model(x)
        grad_x = torch.autograd.grad(score_x.sum(), x)[0].detach()
        #print('grad_x:',grad_x)

        b_idx = torch.arange(bsize).to(x.device)
        delta_list = []
        with torch.no_grad():
            cur_x = x.clone()
            idx_list = []
            for step in range(R):
                delta_x = -(2.0 * cur_x - 1.0)
                delta_list.append(delta_x)
                score_change_x = self.log_g(delta_x * grad_x)
                score_change_x = score_change_x - torch.logsumexp(score_change_x, dim=-1, keepdim=True)
                prob_x_local = torch.softmax(score_change_x, dim=-1)
                #print('(PAFS)prob_x_local:',prob_x_local)
                index = torch.multinomial(prob_x_local, 1).view(-1)
                idx_list.append(index.view(-1, 1))
                #cur_x[b_idx, index] = 1.0 - cur_x[b_idx, index]
                cur_x=model.flip_state(cur_x,b_idx,index)
                #print('cur_x:',cur_x)
            y = cur_x
        y = y.requires_grad_()
        score_y = model(y)
        grad_y = torch.autograd.grad(score_y.sum(), y)[0].detach()

        with torch.no_grad():
            r_idx = torch.arange(R).to(x.device).view(1, -1)
            b_idx = b_idx.view(-1, 1)
            idx_list = torch.cat(idx_list, dim=1)  # bsize x max_r

            # fwd from x -> y
            traj = torch.stack(delta_list, dim=1)  # bsize x max_r x dim
            score_fwd = self.log_g(traj * grad_x.unsqueeze(1))
            log_fwd = torch.log_softmax(score_fwd, dim=-1)
            log_fwd = torch.sum(log_fwd[b_idx, r_idx, idx_list], dim=-1) + score_x.view(-1)

            # backwd from y -> x
            delta_y = -(2.0 * y - 1.0)
            delta_list.append(delta_y)
            traj = torch.stack(delta_list[1:], dim=1)  # bsize x max_r x dim
            score_backwd = self.log_g(traj * grad_y.unsqueeze(1))
            log_backwd = torch.log_softmax(score_backwd, dim=-1)
            log_backwd = torch.sum(log_backwd[b_idx, r_idx, idx_list], dim=-1) + score_y.view(-1)

            log_acc = log_backwd - log_fwd
            accepted = (log_acc.exp() >= torch.rand_like(log_acc)).float().view(-1, *([1] * x_rank))
            new_x = y * accepted + (1.0 - accepted) * x
            #print('accepted:',accepted)
        accs = torch.clamp(log_acc.exp(), max=1).mean().item()

        self._steps += 1
        self._lens.append(R)
        self._accs.append(accs)
        self._hops.append(torch.abs(x - new_x).sum().item() / bsize)
        return new_x
