import torch
import numpy as np
import torchpwl


def plot_and_add(ims, pwl, x_start=-7.0, x_end=7.0):
    plt.ylim((x_start, x_end))
    plt.xlim((x_start, x_end))
    x = torch.linspace(x_start, x_end, steps=1000).unsqueeze(1)
    y = pwl(x)
    brk = plt.plot(list(pwl.x_positions.squeeze(0)), list(pwl(pwl.x_positions.view(-1, 1)).squeeze(1)), "or",)
    xb, yb = get_batch()
    data = plt.plot(xb.squeeze(1).numpy(), yb.squeeze(1).numpy(), "xg")
    curve = plt.plot(list(x), list(y), "b")
    ims.append(data + curve + brk)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import matplotlib.animation as animation

    ims = []
    fig = plt.figure()
    pwl = torchpwl.MonoPWL(num_channels=1, num_breakpoints=10)

    opt = torch.optim.Adam(params=pwl.parameters(), lr=0.2)

    cc = {}

    def get_batch(bs=100):
        if not cc or 1:
            x = np.random.normal(0, scale=2, size=(bs, 1))
            y = np.random.normal(0, scale=0.1, size=(bs, 1)) + np.where(x > 1, x, 1)
            # y = 3*np.sin(2*x) + 1 + x + np.random.normal(0, scale=1, size=(bs, 1))
            cc["d"] = [torch.Tensor(x), torch.Tensor(y)]
        return cc["d"]

    plot_and_add(ims, pwl)
    for e in range(300):
        if e % 10 == 0:
            plot_and_add(ims, pwl)
        x, y = get_batch()
        pred = pwl(x)
        loss = torch.mean((y - pred) ** 2) + 100 * torch.mean(
            torch.clamp(0.1 - torch.abs(pwl.get_spreads()), 0, 100) ** 2
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())

    animation.ArtistAnimation(fig, ims, repeat=True).save("gifs/sample_fit.gif", writer="pillow")
