from torchpwl import MonoPWL, Calibrator
import torch
import math
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import copy
import random

N = 1
C = 1
K = 20
x = torch.linspace(-2, 2, steps=100).unsqueeze(1)


def ans(x):
    p = 1.5 * math.pi
    a = x + torch.sin(p * x) / p
    return -a


torch.manual_seed(11)

xa = torch.linspace(-2, 2, steps=30).unsqueeze(0)
c = torch.nn.Sequential(Calibrator(keypoints=xa, monotonicity=[-1]), torch.nn.Linear(1, 1, bias=True),)


def get_param_jiggle(module: torch.nn.Module, seed: int, std=1.0) -> [torch.Tensor]:
    jiggle = []
    generator = torch.random.manual_seed(seed)
    for param in module.parameters():
        jiggle.append(torch.empty_like(param.data).normal_(mean=0, std=std, generator=generator))
    return jiggle


def apply_param_jiggle(module: torch.nn.Module, jiggle: [torch.Tensor]):
    params = list(module.parameters())
    assert len(jiggle) == len(params)
    for param, jig in zip(params, jiggle):
        param.data.add_(jig)


def estimate_gradients(module: torch.nn.Module, eval_fn, population: int, std: float = 1.0):
    base_loss = eval_fn(module)
    grad = None
    sd = module.state_dict()
    cloned = copy.deepcopy(module)
    for seed in range(population):
        cloned.load_state_dict(sd)
        jiggle = get_param_jiggle(cloned, seed=seed, std=std)
        apply_param_jiggle(cloned, jiggle)
        exp_loss = eval_fn(cloned)
        delta = exp_loss - base_loss
        for e in jiggle:
            e.mul_(delta)
        if grad is None:
            grad = jiggle
        else:
            for g, j in zip(grad, jiggle):
                g.add_(j)
    for param, grad in zip(module.parameters(), grad):
        param.grad = grad / population / (std ** 2)
    return base_loss


# opt = torch.optim.Adam(c.parameters(), 0.1, betas=(0.6, 0.99), weight_decay=0.0001)
opt = torch.optim.SGD(c.parameters(), 0.3, momentum=0.5, weight_decay=0.0001)


def eval_fn(module):
    x = torch.linspace(-2, 2, steps=100).unsqueeze(1)
    y_pred = module(x)
    y_actual = ans(x)
    loss = ((y_actual - y_pred) ** 2).mean()
    return loss.item()


ims = []
fig = plt.figure()
for e in range(200):

    if e % 5 == 0:
        x = torch.linspace(-2, 2, steps=100).unsqueeze(1)
        y_pred = c(x)
        y_actual = ans(x)
        fit = plt.plot(x.detach().squeeze().numpy(), y_pred.detach().squeeze().numpy(), "b")
        points = plt.plot(x.detach().squeeze().numpy(), y_actual.detach().squeeze().numpy(), "r")
        ims.append(fit + points)

    opt.zero_grad()
    base_loss = estimate_gradients(module=c, eval_fn=eval_fn, population=500, std=0.003)
    # print([e.grad for e in c.parameters()])
    # opt.zero_grad()
    # x = torch.linspace(-2, 2, steps=100).unsqueeze(1)
    # y_pred = c(x)
    # y_actual = ans(x)
    # loss = ((y_actual - y_pred) ** 2).mean()
    # loss.backward()
    # print([e.grad for e in c.parameters()])

    # exit()
    opt.step()
    print(base_loss)

animation.ArtistAnimation(fig, ims, repeat=True).save("xx.gif", writer="pillow")
import torch.jit
