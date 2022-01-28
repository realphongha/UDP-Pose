import cv2
import torch
import numpy as np
import time
import torchvision


def letterbox(img, new_shape=(640, 640)):
    H, W = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / H, new_shape[1] / W)
    nH, nW = round(H * r), round(W * r)
    pH, pW = np.mod(new_shape[0] - nH, 32) / 2, np.mod(new_shape[1] - nW, 32) / 2

    if (H, W) != (nH, nW):
        img = cv2.resize(img, (nW, nH), interpolation=cv2.INTER_LINEAR)

    top, bottom = round(pH - 0.1), round(pH + 0.1)
    left, right = round(pW - 0.1), round(pW + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img


def scale_boxes(boxes, orig_shape, new_shape):
    H, W = orig_shape
    nH, nW = new_shape
    gain = min(nH / H, nW / W)
    pad = (nH - H * gain) / 2, (nW - W * gain) / 2

    boxes[:, ::2] -= pad[1]
    boxes[:, 1::2] -= pad[0]
    boxes[:, :4] /= gain
    
    boxes[:, ::2].clamp_(0, orig_shape[1])
    boxes[:, 1::2].clamp_(0, orig_shape[0])
    return boxes.round()


def xywh2xyxy(x):
    boxes = x.clone()
    boxes[:, 0] = x[:, 0] - x[:, 2] / 2
    boxes[:, 1] = x[:, 1] - x[:, 3] / 2
    boxes[:, 2] = x[:, 0] + x[:, 2] / 2
    boxes[:, 3] = x[:, 1] + x[:, 3] / 2
    return boxes


def xyxy2xywh(x):
    y = x.clone()
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2cs(x, y, w, h, image_width, image_height, pixel_std=200):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5
    
    aspect_ratio = image_width * 1.0 / image_height

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

# def non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None):
#     candidates = pred[..., 4] > conf_thres 

#     max_wh = 4096
#     max_nms = 30000
#     max_det = 300

#     output = [torch.zeros((0, 6), device=pred.device)] * pred.shape[0]

#     for xi, x in enumerate(pred):
#         x = x[candidates[xi]]

#         if not x.shape[0]: continue

#         # compute conf
#         x[:, 5:] *= x[:, 4:5]   # conf = obj_conf * cls_conf

#         # box
#         box = xywh2xyxy(x[:, :4])

#         # detection matrix nx6
#         conf, j = x[:, 5:].max(1, keepdim=True)
#         x = torch.cat([box, conf, j.float()], dim=1)[conf.view(-1) > conf_thres]

#         # filter by class
#         if classes is not None:
#             x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

#         # check shape
#         n = x.shape[0]
#         if not n: 
#             continue
#         elif n > max_nms:
#             x = x[x[:, 4].argsort(descending=True)[:max_nms]]

#         # batched nms
#         c = x[:, 5:6] * max_wh
#         boxes, scores = x[:, :4] + c, x[:, 4]
#         keep = ops.nms(boxes, scores, iou_thres)

#         if keep.shape[0] > max_det:
#             keep = keep[:max_det]

#         output[xi] = x[keep]

#     return output


def yolo2xyxy(size, box):
    image_width, image_height = size[1], size[0]
    center_X, center_y, width, height = box
    x1 = (center_X-width/2)*image_width
    x2 = (center_X+width/2)*image_width
    y1 = (center_y-height/2)*image_height
    y2 = (center_y+height/2)*image_height
    x1, y1, x2, y2 = round(x1-1), round(y1-1), round(x2-1), round(y2-1)
    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    if x2 >= image_width: x2 = image_width-1
    if y2 >= image_height: y2 = image_height-1
    return x1, y1, x2, y2
