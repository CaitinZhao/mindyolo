import mindspore as ms
from mindspore import ops, nn


@ops.constexpr
def tensor(x):
    ms.Tensor(x)


def normalize(x, mean=(114, 114, 114), std=(255, 255, 255)):
    if len(x.shape) == 3:
        x = ops.expand_dims(x, 0)
    return (x - tensor(mean).reshape(1, 1, 1, 3)) / tensor(std).reshape(1, 1, 1, 3)


def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right"""
    y = ops.Identity()(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, multi_label=False, max_det=300, max_wh = 4096):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,5+c) tensor per image [xyxy, conf, cls]
    """
    nc = prediction.shape[2] - 5  # number of classes
    nms = ops.NMSWithMask(iou_thres)
    res = []
    for i in range(prediction.shape[0]):  # image index, image inference
        x = prediction[i]
        # Detections matrix nx5 (xyxy, conf)
        box = xywh2xyxy(x[:, :4])
        # Detections matrix nx6 (xyxy, conf, cls)
        obj_conf = (x[:, 4] > conf_thres).astype(x.dtype) * x[:, 4]
        if multi_label and nc > 1:
            box = ops.repeat_elements(box, nc, 0)
            cls_conf = (x[:, 5:] * ops.expand_dims(obj_conf, 1)).reshape(-1)
            cls = ops.tile(ops.arange(nc), (x.shape[0],))
        else:  # best class only
            cls_conf = x[:, 5:].max(1) * obj_conf
            cls =x[:, 5:].argmax(1)
        # filter top 2000
        _, idx = ops.TopK()(cls_conf, 2000)
        c_conf = cls_conf[idx]
        box = box[idx]
        cls = cls[idx]
        c_box = box + ops.expand_dims(cls.astype(box.dtype) * max_wh, 1)

        # nms
        nms_box, nms_idx, mask = nms(ops.concat((c_box, ops.expand_dims(c_conf, 1)), 1))  # NMS for per sample
        c_mask = mask.astype(nms_box.dtype) * (c_conf > conf_thres).astype(nms_box.dtype)
        c_score = c_conf * c_mask

        # filter top max_det
        _, idx = ops.TopK()(c_score, max_det)
        box = box[idx]
        c_score = c_score[idx]
        c_v = cls[idx].astype(box.dtype)
        res.append(ops.concat((box, ops.expand_dims(c_score, 1), ops.expand_dims(c_v, 1)), 1))
    res = ops.stack(res, 0)  # res: (batch_size, num_sample, 6), 6 is xyxy, score, cls
    return res


class EvalNet(nn.Cell):
    def __init__(self, net, mean=(114, 114, 114), std=(255, 255, 255), conf_thres=0.25, iou_thres=0.45, max_det=300):
        super(EvalNet, self).__init__()
        self.net = net
        self.mean = mean
        self.std = std
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

    def construct(self, img):
        x = normalize(img, self.mean, self.std)
        x = ops.transpose(x, (0, 3, 1, 2))
        out = self.net(x)
        res = non_max_suppression(out, self.conf_thres, self.iou_thres, True, self.max_det)
        return res


if __name__ == '__main__':
    import numpy as np
    from mindyolo.utils.metrics import non_max_suppression as nm
    x = np.load("E:\WorkPlace\mindyolo\deploy\\net_out.npy")[:4]
    np.save("pred.npy", x)
    xm = ms.Tensor(x)
    # print(xm.shape)
    # y = ops.repeat_elements(xm[0][:2, :4], 4, 0)
    # print(y)
    # print(y.reshape(-1))
    # print(y.shape)
    out = non_max_suppression(xm, conf_thres=0.001, iou_thres=0.65, multi_label=True, max_det=300)
    out = out.asnumpy()
    mask = out[:,:,4] > 0.001
    for i, o in enumerate(out):
        p = o[mask[i]]
        print(p.shape, p)
    print("="*10)
    q = nm(x, conf_thres=0.001, iou_thres=0.65, multi_label=True)
    for i in q:
        print(i.shape, i)
    t = i - p
    print(t[:,3])
    # print(q[0][t > 0])

