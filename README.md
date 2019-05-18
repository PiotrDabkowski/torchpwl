### Piecewise Linear Functions (PWL) implementation in PyTorch

Piecewise Linear Functions (PWLs) can be used to approximate any 1D function. 
PWLs are built with a configurable number of line segments - the more segments the more accurate the approximation.
This package implements PWLs in PyTorch and as such you can optimize them using standard gradient descent
and even use them as the activation function in your neural nets. 

For example here you can see a PWL (blue) fitting to the dataset (green). Red dots are breakpoints - they connect separate PWL segments:


![](examples/gifs/demo.gif)


##### Installation:

`pip install torchpwl`

##### Example usage:

```python
import torchpwl

# Create a PWL consisting of 3 segments for 5 channels - each channel will have its own PWL function.
pwl = torchpwl.PWL(num_channels=5, num_breakpoints=3)
x = torch.Tensor(11, 5).normal_()
y = pwl(x)

```

Monotonicity is also supported via `MonoPWL`:

```python
# Create a PWL with monotonicity constraint (here increasing).
mono_increasing_pwl = torchpwl.MonoPWL(num_channels=5, num_breakpoints=3, monotonicity=+1)
x = torch.Tensor(11, 5).normal_()
y = mono_increasing_pwl(x)
```

See the class documentations for more details.

##### More examples:

Below you can see a PWL fitting to the sine. 
![](examples/gifs/pure_sine_fit.gif)

See the examples/gif directory for more examples.

##### Performance

The current implementation is pretty slow - at least 10 times slower than ReLU.
If needed we can always implement it as a native op.

