import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter

__all__ = ['EMA']


class EMA(nn.Cell):
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weight is necessary for some training schemes to perform well.
    """

    def __init__(self, model, ema_model, decay=0.9999, updates=0):
        super(EMA, self).__init__()
        # Create EMA
        self.ema = ema_model
        self.ema.set_train(False)
        self.weight = ms.ParameterTuple(list(model.get_parameters()))
        self.ema_weight = ms.ParameterTuple(list(ema_model.get_parameters()))
        self.updates = Parameter(Tensor(updates, ms.float32), requires_grad=False)  # number of EMA updates
        self.decay_value = decay
        self.assign = ops.Assign()
        self.hyper_map = ops.HyperMap()

    def decay(self, x):
        # decay exponential ramp (to help early epochs)
        return self.decay_value * (1 - ops.exp(ops.neg(x) / 2000))

    @ms.ms_function
    def update(self):
        # Update EMA parameters
        def update_param(d, ema_v, weight):
            if weight.dtype == ms.int32:
                return self.assign(ema_v, weight)
            else:
                return self.assign(ema_v, weight * (1. - d) + ema_v * d)

        updates = ops.assign_add(self.updates, 1)
        d = self.decay(self.updates)
        success = self.hyper_map(ops.partial(update_param, d), self.ema_weight, self.weight)
        updates = ops.depend(updates, success)

        return updates

    @ms.ms_function
    def clone_from_model(self):
        updates = ops.assign_add(self.updates, 1)
        success = self.hyper_map(ops.assign, self.ema_weight, self.weight)
        updates = ops.depend(updates, success)
        return updates

    def load_param_from_dict(self, ckpt):
        if isinstance(ckpt, str) and ckpt.endswith("ckpt"):
            param_dict_ema = ms.load_checkpoint(ckpt)
        elif isinstance(ckpt, dict):
            param_dict_ema = ckpt
        else:
            raise NotImplementedError(f"input ckpt type not support, {ckpt}")

        for w in self.ema_weight:
            if w.name in param_dict_ema:
                ops.assign(w, param_dict_ema[w.name])
            else:
                print(f"EMA.load_param_from_dict: [{w.name}] not load.")
