import time
import cv2
import onnx
import onnxruntime
import numpy as np
from matplotlib import pyplot as plt
from draw_box_utils import draw_box


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def scale_img(img: np.ndarray,
              new_shape=(416, 416),
              color=(114, 114, 114),
              auto=True,
              scale_fill=False,
              scale_up=True):


    shape = img.shape[:2]  # [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)


    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up: 
        r = min(r, 1.0)

    # compute padding
    ratio = r, r  
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1] 
    if auto:  
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  
    elif scale_fill:  
        dw, dh = 0, 0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  

    dw /= 2  
    dh /= 2

    # shape:[h, w]  new_unpad:[w, h]
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1)) 

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def clip_coords(boxes: np.ndarray, img_shape: tuple):

    boxes[:, 0].clip(0, img_shape[1])  # x1
    boxes[:, 1].clip(0, img_shape[0])  # y1
    boxes[:, 2].clip(0, img_shape[1])  # x2
    boxes[:, 3].clip(0, img_shape[0])  # y2


def turn_back_coords(img1_shape, coords, img0_shape, ratio_pad=None):

    if ratio_pad is None:  
        gain = max(img1_shape) / max(img0_shape)  
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  
    coords[:, [1, 3]] -= pad[1]  
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def xywh2xyxy(x: np.ndarray):
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  
    y[:, 1] = x[:, 1] - x[:, 3] / 2  
    y[:, 2] = x[:, 0] + x[:, 2] / 2  
    y[:, 3] = x[:, 1] + x[:, 3] / 2  
    return y


def bboxes_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def nms(bboxes: np.ndarray, iou_threshold=0.5, soft_threshold=0.3, sigma=0.5, method="nms", ) -> np.ndarray:

    assert method in ["nms", "soft-nms"]

    bboxes = np.concatenate([bboxes, np.arange(bboxes.shape[0]).reshape(-1, 1)], axis=1)

    best_bboxes_index = []
    while len(bboxes) > 0:
        max_ind = np.argmax(bboxes[:, 4])  
        best_bbox = bboxes[max_ind]
        best_bboxes_index.append(best_bbox[5])
        bboxes = np.concatenate([bboxes[:max_ind], bboxes[max_ind + 1:]])  
        ious = bboxes_iou(best_bbox[np.newaxis, :4], bboxes[:, :4])

        if method == "nms":
            iou_mask = np.less(ious, iou_threshold)  
        else:  # soft-nms
            weight = np.exp(-(np.square(ious) / sigma))
            bboxes[:, 4] = bboxes[:, 4] * weight
            iou_mask = np.greater(bboxes[:, 4], soft_threshold) 

        bboxes = bboxes[iou_mask]

    return np.array(best_bboxes_index, dtype=np.int8)


def post_process(pred: np.ndarray, multi_label=False, conf_thres=0.3):
    """
    输入的xywh都是归一化后的值
    :param pred: [num_obj, [x1, y1, x2, y2, objectness, cls1, cls1...]]
    :param img_size:
    :param multi_label:
    :param conf_thres:
    :return:
    """
    min_wh, max_wh = 2, 4096
    pred = pred[pred[:, 4] > conf_thres]  
    pred = pred[((pred[:, 2:4] > min_wh) & (pred[:, 2:4] < max_wh)).all(1)]  

    if pred.shape[0] == 0:
        return np.empty((0, 6))  

    box = xywh2xyxy(pred[:, :4])

    if multi_label:  
        pass
    else:  
        objectness = pred[:, 5:]
        class_index = np.argmax(objectness, axis=1)
        conf = objectness[(np.arange(pred.shape[0]), class_index)]
       
        pred = np.concatenate((box,
                               np.expand_dims(conf, axis=1),
                               np.expand_dims(class_index, axis=1)), 1)[conf > conf_thres]

    n = pred.shape[0]  
    if n == 0:
        return np.empty((0, 6))  

    cls = pred[:, 5]  
    boxes, scores = pred[:, :4] + cls.reshape(-1, 1) * max_wh, pred[:, 4:5]
    t1 = time.time()
    indexes = nms(np.concatenate([boxes, scores], axis=1))
    print("NMS time is {}".format(time.time() - t1))
    pred = pred[indexes]

    return pred


def main():
    img_size = 512
    save_path = "yolov3spp.onnx"
    img_path = "test.jpg"
    input_size = (img_size, img_size) 


    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(save_path)

    img_o = cv2.imread(img_path)
    assert img_o is not None, "Image Not Found " + img_path


    img, ratio, pad = scale_img(img_o, new_shape=input_size, auto=False, color=(0, 0, 0))

    img = img[:, :, ::-1].transpose(2, 0, 1)  
    img = np.ascontiguousarray(img).astype(np.float32)

    img /= 255.0  
    img = np.expand_dims(img, axis=0)  


    ort_inputs = {"images": img}

    t1 = time.time()

    pred = ort_session.run(None, ort_inputs)[0]
    t2 = time.time()
    print(t2 - t1)

    pred[:, [0, 2]] *= input_size[1]
    pred[:, [1, 3]] *= input_size[0]
    pred = post_process(pred)


    p_boxes = turn_back_coords(img1_shape=img.shape[2:],
                               coords=pred[:, :4],
                               img0_shape=img_o.shape,
                               ratio_pad=[ratio, pad]).round()


    bboxes = p_boxes
    scores = pred[:, 4]
    classes = pred[:, 5].astype(np.int) + 1

    category_index = dict([(i + 1, str(i + 1)) for i in range(90)])
    img_o = draw_box(img_o[:, :, ::-1], bboxes, classes, scores, category_index)
    plt.imshow(img_o)
    plt.show()


if __name__ == '__main__':
    main()
