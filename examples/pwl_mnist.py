"""
MNIST Demo with PWL activation.

REQUIRES:
torchvision
sorch


RESULTS:
Comparable to RELU, no significant different, in fact it evolves to RELU when using just 1 breakpoint.

"""
import sys
import os
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torchpwl
from torchvision.datasets import mnist
import torchvision.transforms as T
from sorch.trainer import trainer


def get_mnist_loader(batch_size=16, num_workers=0, train=True):
    """Build and return a data loader."""
    transform = []
    transform.append(T.Resize(32))
    transform.append(T.ToTensor())
    # transform.append(lambda x: torch.cat((x, x, x), dim=0))

    dataset = mnist.MNIST("~/datasets/MNIST", transform=T.Compose(transform), train=train)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return data_loader


class Lambda(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *input):
        return self.fn(*input)


def get_cnn(activation_fn):
    return torch.nn.Sequential(
        torch.nn.BatchNorm2d(1),
        torch.nn.Conv2d(1, 32, 3, 1, 1),
        torch.nn.BatchNorm2d(32),
        activation_fn(32),
        torch.nn.Conv2d(32, 32, 3, 1, 1),
        torch.nn.BatchNorm2d(32),
        activation_fn(32),
        torch.nn.AvgPool2d(2),
        torch.nn.Conv2d(32, 64, 3, 1, 1),
        torch.nn.BatchNorm2d(64),
        activation_fn(64),
        torch.nn.Conv2d(64, 64, 3, 1, 1),
        torch.nn.BatchNorm2d(64),
        activation_fn(64),
        torch.nn.AvgPool2d(2),
        torch.nn.Conv2d(64, 128, 3, 1, 1),
        torch.nn.BatchNorm2d(128),
        activation_fn(128),
        torch.nn.Conv2d(128, 128, 3, 1, 1),
        torch.nn.BatchNorm2d(128),
        activation_fn(128),
        torch.nn.AvgPool2d(2),
        torch.nn.Conv2d(128, 256, 3, 1, 1),
        torch.nn.BatchNorm2d(256),
        activation_fn(256),
        Lambda(lambda x: x.mean(2).mean(2)),
        torch.nn.Linear(256, 10),
    )


def maybe_cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x


def get_universal_pwl_fn(num_breakpoints):
    """Returns a PWL activation generator fn. All the activations share the same PWL"""
    pwl = maybe_cuda(torchpwl.MonoPWL(num_features=1, num_breakpoints=num_breakpoints, monotonicity=1))

    def act_fn(chans):
        def fn(x):
            old_sh = list(x.shape)
            return pwl(x.view(-1, 1)).view(*old_sh)

        return Lambda(fn)

    return act_fn, pwl


def plot_and_add(ims, pwl, x_start=-7.0, x_end=7.0):
    pwl.cpu()
    plt.ylim((x_start, x_end))
    plt.xlim((x_start, x_end))
    x = torch.linspace(x_start, x_end, steps=1000).unsqueeze(1)
    y = pwl(x)
    brk = plt.plot(list(pwl.x_positions.squeeze(0)), list(pwl(pwl.x_positions.view(-1, 1)).squeeze(1)), "or",)
    curve = plt.plot(list(x), list(y), "b")
    ims.append(curve + brk)
    pwl.cuda()


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import matplotlib.animation as animation

    ims = []
    fig = plt.figure()

    mnist.MNIST("~/datasets/MNIST", download=True, train=True, transform=None)
    activation_fn = lambda channels: Lambda(lambda x: F.relu(x))
    # 10 times slower bit it works... Does not slow down significantly when increasing num_breakpoints from 1 to 10.
    # Will need a native op to speed up unfortnuately.
    activation_fn, upwl = get_universal_pwl_fn(1)
    # activation_fn = lambda channels: torchpwl.MonoPWL(num_features=channels, num_breakpoints=1, monotonicity=1)
    cnn = maybe_cuda(get_cnn(activation_fn))
    opt = torch.optim.SGD(params=list(cnn.parameters()) + list(upwl.parameters()), lr=0.1, momentum=0.5)

    def update_fn(trainer, batch):
        x, labels = map(maybe_cuda, batch)

        logits = cnn(x)
        loss = F.cross_entropy(
            logits, target=labels
        )  # + 50 * torch.mean(torch.clamp(0.3 - torch.abs(upwl.get_spreads()), 0, 100)**2)
        if trainer.state.is_training:
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = torch.mean(torch.eq(torch.argmax(logits, dim=1), labels).float())

        trainer.state.add_scalar_data_point("loss", loss.item(), smoothing=0.98)
        trainer.state.add_scalar_data_point("acc", acc.item(), smoothing=0.98)

    st = trainer.STrainer(
        update_fn,
        info_string_components=trainer.DEFAULT_INFO_STRING_COMPONENTS + ("acc: {acc:.3f}",),
        modules=[cnn],
    )
    st.state.acc = "l"

    for e in range(30):
        plot_and_add(ims, upwl)
        st.train(get_mnist_loader(batch_size=512, num_workers=2, train=True), steps=10)
        # st.evaluate(
        #     get_mnist_loader(batch_size=512, num_workers=2, train=False))
    if ims:
        animation.ArtistAnimation(fig, ims).save("gifs/upwl.gif", writer="pillow")
