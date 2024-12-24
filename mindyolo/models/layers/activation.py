"""
Custom activation operators.
"""
from mindspore import nn, ops


class Swish(nn.Cell):
    """
    Swish activation function: x * sigmoid(βx). If beta equals 1, you can use nn.SiLU instead.
    """

    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def construct(self, x):
        return x * ops.sigmoid(self.beta * x)


class SiLU(nn.Cell):
    r"""
    Applies the silu linear unit function element-wise.

    .. math::

        \text{SiLU}(x) = x * \sigma(x),

    where :math:`x_i` is an element of the input, :math:`\sigma(x)` is Sigmoid function.

    .. math::

        \text{sigmoid}(x_i) = \frac{1}{1 + \exp(-x_i)},

    SiLU Activation Function Graph:

    .. image:: ../images/SiLU.png
        :align: center

    Inputs:
        - **input** (Tensor) - `input` is :math:`x` in the preceding formula.
          Input with the data type float16 or float32. Tensor of any dimension.

    Outputs:
        Tensor, with the same type and shape as the `input`.

    Raises:
        TypeError: If dtype of `input` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> input = Tensor(np.array([-1, 2, -3, 2, -1]), mindspore.float16)
        >>> silu = nn.SiLU()
        >>> output = silu(input)
        >>> print(output)
        [-0.269  1.762  -0.1423  1.762  -0.269]
    """

    def __init__(self):
        """Initialize SiLU."""
        super(SiLU, self).__init__()
        self.fused_op = True

    def construct(self, x):
        if self.fused_op:
            return ops.function.silu(x)
        return x * ops.sigmoid(x)
