import json

from models_co_attention import *
from build_utils.datasets import *
from build_utils.utils import *
from train_utils import get_coco_api_from_dataset, CocoEvaluator


def summarize(self, catId=None):


    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:

            s = self.eval['precision']

            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:

            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 12, [""] * 12
    stats[0], print_list[0] = _summarize(1)
    stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
    stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
    stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
    stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
    stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[1])
    stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[2])
    stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
    stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run accumulate() first')

    return stats, print_info


def main(parser_data):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))


    label_json_path = './data/my_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {v: k for k, v in class_dict.items()}

    data_dict = parse_data_cfg(parser_data.data)
    test_path = "/media/Harddisk/jmy/SPG-Det/battery_dataset_101/"


    batch_size = parser_data.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]) 
    print('Using %g dataloader workers' % nw)


    val_dataset = LoadImagesAndLabels(test_path,  parser_data.img_size, batch_size,
                                      hyp=parser_data.hyp,
                                      rect=True,  
                                      split='val')
    val_dataset_loader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=batch_size,
                                                    num_workers=nw,
                                                    pin_memory=True,
                                                    collate_fn=val_dataset.collate_fn)


    model = SPG_Det(parser_data.cfg)
    model.load_state_dict(torch.load(parser_data.weights, map_location='cpu')["model"],strict=False)
    model.to(device)


    coco = get_coco_api_from_dataset(val_dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)
    cpu_device = torch.device("cpu")

    model.eval()
    with torch.no_grad():
        for imgs, targets, paths, shapes, img_index,normal, mask in tqdm(val_dataset_loader, desc="validation..."):
            mask = mask.to(device).float() / 255.0
            imgs = imgs.to(device).float() / 255.0
            pred = model(imgs, mask)[0]  

            pred = non_max_suppression(pred, conf_thres=0.01, iou_thres=0.6, multi_label=False)

            outputs = []
            for index, p in enumerate(pred):
                if p is None:
                    p = torch.empty((0, 6), device=cpu_device)
                    boxes = torch.empty((0, 4), device=cpu_device)
                else:

                    boxes = p[:, :4]

                    boxes = scale_coords(imgs[index].shape[2:], boxes, shapes[index][0]).round()


                info = {"boxes": boxes.to(cpu_device),
                        "labels": p[:, 5].to(device=cpu_device, dtype=torch.int64),
                        "scores": p[:, 4].to(cpu_device)}
                outputs.append(info)

            res = {img_id: output for img_id, output in zip(img_index, outputs)}

            coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()


    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    coco_eval = coco_evaluator.coco_eval["bbox"]

    coco_stats, print_coco = summarize(coco_eval)


    voc_map_info_list = []
    for i in range(len(category_index)):
        stats, _ = summarize(coco_eval, catId=i)
        voc_map_info_list.append(" {:15}: {}".format(category_index[i + 1], stats[1]))

    print_voc = "\n".join(voc_map_info_list)
    print(print_voc)


    with open("record_mAP.txt", "w") as f:
        record_lines = ["COCO results:",
                        print_coco,
                        "",
                        "mAP(IoU=0.5) for each category:",
                        print_voc]
        f.write("\n".join(record_lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)


    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--num-classes', type=int, default='6', help='number of classes')
    parser.add_argument('--cfg', type=str, default='cfg/my_yolov3.cfg', help="*.cfg path")
    parser.add_argument('--data', type=str, default='data/my_data.data', help='*.data path')
    parser.add_argument('--hyp', type=str, default='cfg/hyp.yaml', help='hyperparameters path')
    parser.add_argument('--img-size', type=int, default=512, help='test size')
    parser.add_argument('--weights', default="", type=str, help='training weights')
    parser.add_argument('--batch_size', default=1, type=int, metavar='N',
                        help='batch size when validation.')

    args = parser.parse_args()

    main(args)
