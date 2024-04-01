import torch
import numpy as np
import  torch.nn as nn
from tqdm import tqdm
import time
class BaseSampler():
    def __init__(self, args, U=1, log_g=None, seed=0, device=torch.device("cpu")):
        self.U = U
        self.ess_ratio = 0#args.ess_ratio
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
        return sum(self._accs[int(self._steps * (1 - ratio)):]) / int(self._steps * ratio)

    @property
    def avg_hops(self):
        ratio = self.ess_ratio
        return sum(self._hops[int(self._steps * (1 - ratio)):]) / int(self._steps * ratio)

class PathAuxiliarySampler(BaseSampler):
    def __init__(self, args=0, U=1.0, log_g=None, seed=0, device=torch.device("cpu")):
        super().__init__(args, U, log_g, seed, device)
        self.average_acceptance_rate=0
        self.size=0
        self.all_accepts=[]
        self.U=U
    def step(self, x, model):
        #R = int(self.rng.integers(1, 2 * self.U, 1))
        R=torch.poisson(torch.tensor(self.U)) #Poisson distribution
        R=int(R.detach().numpy())
        bsize = x.shape[0]
        self.size=x.shape[1]
        x_rank = len(x.shape) - 1
        x = x.requires_grad_()
        Zx, Zy = 1., 1.
        b_idx = torch.arange(bsize).to(x.device)
        cur_x = x.clone()
        with torch.no_grad():
            for step in range(R):
                score_change_x = self.log_g(model.change(cur_x))
                if step == 0:
                    Zx = torch.logsumexp(score_change_x, dim=1)
                prob_x_local = torch.softmax(score_change_x, dim=-1)
                index = torch.multinomial(prob_x_local, 1).view(-1)
                cur_x[b_idx, index] = 1 - cur_x[b_idx, index]
            y = cur_x

        score_change_y = self.log_g(model.change(y))
        Zy = torch.logsumexp(score_change_y, dim=1)

        log_acc = Zx - Zy
        accepted = (log_acc.exp() >= torch.rand_like(log_acc)).float().view(-1, *([1] * x_rank))
        new_x = y * accepted + (1.0 - accepted) * x

        accs = torch.clamp(log_acc.exp(), max=1).mean().item()


        self._steps += 1
        self._lens.append(R)
        self._accs.append(accs)
        self._hops.append(torch.abs(x - new_x).sum().item() / bsize)
        self.all_accepts.append(accs)
        self.average_acceptance_rate=np.sum(self.all_accepts)/len(self.all_accepts)
        return new_x

    def update_U(self):
        self.U = np.clip(self.U + 0.001 * (self.average_acceptance_rate - 0.574), a_min=1, a_max=self.size)
        #print("average_acceptance_rate:", self.average_acceptance_rate)



class BaseSamplerPAS(object):
    def __init__(self, seed=100):
        self.rng = np.random.default_rng(seed)
        self._reset()

    def _reset(self):
        self.logp = []
        self.trace = []
        self.elapse = 0
        self.succ = 0

    def step(self, model, t, x):
        raise NotImplementedError

    def sample(self, model, T=1000, method='Gibbs', *args, **kwargs):
        self._reset()
        self.x = model.init_state()
        self.energy = model.energy(self.x)

        # progress_bar = tqdm(range(T))
        # progress_bar.set_description(f"[{method}]")
        t_begin = time.time()
        for t in range(T):
            self.step(model, t, *args, **kwargs)
            self.logp.append([self.energy.item()])
            self.trace.append([model.trace(self.x)])
        t_end = time.time()
        self.elapse += t_end - t_begin
        return self.logp, self.trace, self.elapse, self.succ
class MSASampler(BaseSamplerPAS):
    """
    Multi Step Accurate Sampler
    """

    def __init__(self, R=1, seed=100):
        super().__init__(seed=seed)
        self.R = R
        self.energy_change = None
        self.Z = None
        self.average_acceptance_rate=0
        self.size=5
        self.accs=[]
    def step(self, model, t,*args,**kwargs):
        R = int(self.rng.integers(1, 2 * self.R, 1))
        x = self.x
        #print('initial x:',x)
        indices = []
        probs=[]
        # print('self.R:',self.R)
        # print('R',R)
        for t in range(R):
            if t == 0:
                if self.Z is None:
                    energy_change = model.change(x)
                    Zx = torch.logsumexp(- energy_change / 2, dim=-1)
                else:
                    energy_change = self.energy_change
                    Zx = self.Z
            else:
                energy_change = model.change(x)
            #print('energy change:',energy_change)
            prob = torch.exp(-energy_change / 2)
            #print('prob:',prob)
            probs.append(prob)
            idx = torch.multinomial(prob, 1, replacement=False) #originally replacement=True
            #print('index:',idx)
            indices.append(idx)
            print('original x:',x)
            x = model.flip_state(x, idx)
            print('flipped x:',x)
        #print('indices:',indices)
        #print('probs:',probs)
        energy_change_y = model.change(x)
        Zy = torch.logsumexp(- energy_change_y / 2, dim=-1)
        self.accs.append(torch.exp(Zx-Zy).detach().numpy())
        #print('exp(Zx-Zy):',torch.exp(Zx-Zy))
        if self.rng.random() < torch.exp(Zx - Zy):
            #print(x)
            self.x = x
            self.energy = model.energy(self.x)
            self.succ += 1
            self.energy_change = energy_change_y
            self.Z = Zy

            # if model.energy(model.current_x) < model.energy(self.x):
            #     model.current_x = self.x
            #     model.distances.append(model.get_total_distance(model.current_x))
            # else:
            #     if np.random.rand() < np.exp(
            #             -(model.energy(self.x) - model.energy(model.current_x)).detach().numpy() / model.t):
            #         model.current_x = self.x

        return x
    def update_U(self,avg):
        self.R = np.clip(self.R + 0.001 * (avg - 0.574), a_min=1, a_max=self.size)
        #print('updated self.R:',self.R,'AVG:',self.average_acceptance_rate)


