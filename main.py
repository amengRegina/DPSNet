import datetime
import argparse

import yaml
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from models_co_attention import *
from build_utils.datasets import *
from build_utils.utils import *
from train_utils import train_eval_utils as train_util
from train_utils import get_coco_api_from_dataset


def train(hyp):
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))
    
    w="weights"
    wdir = w + os.sep  
    best = wdir + "figure1.pt"
    results_file =wdir+ "results"+"{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+w+".txt"

    
    cfg = opt.cfg
    data = opt.data
    epochs = opt.epochs
    batch_size = opt.batch_size
    accumulate = max(round(64 / batch_size), 1)  
    weights = opt.weights 
    imgsz_train = opt.img_size
    imgsz_test = opt.img_size 
    multi_scale = opt.multi_scale


    gs = 32  
    assert math.fmod(imgsz_test, gs) == 0, "--img-size %g must be a %g-multiple" % (imgsz_test, gs)
    grid_min, grid_max = imgsz_test // gs, imgsz_test // gs
    if multi_scale:
        imgsz_min = opt.img_size // 1.5
        imgsz_max = opt.img_size // 0.667


        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
        imgsz_min, imgsz_max = int(grid_min * gs), int(grid_max * gs)
        imgsz_train = imgsz_max  
        print("Using multi_scale training, image range[{}, {}]".format(imgsz_min, imgsz_max))

    data_dict = parse_data_cfg(data)
    train_path = "/media/Harddisk/jmy/SPG-Det/battery_dataset_101/"
    test_path = "/media/Harddisk/jmy/SPG-Det/battery_dataset_101/"
    nc = 1 if opt.single_cls else int(data_dict["classes"])  
    hyp["cls"] *= nc / 80  
    hyp["obj"] *= imgsz_test / 320


    for f in glob.glob(results_file):
        os.remove(f)


    model = SPG_Det(cfg).to(device)

    for k in model.parameters():
        print(' {}'.format(k.requires_grad))


    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=hyp["lr0"], momentum=hyp["momentum"],
                          weight_decay=hyp["weight_decay"], nesterov=True)

    scaler = torch.cuda.amp.GradScaler() if opt.amp else None

    start_epoch = 0
    best_map = 0.0
    if weights.endswith(".pt") or weights.endswith(".pth"):
        ckpt = torch.load(weights, map_location=device)


        try:
            ckpt["model"] = {k: v for k, v in ckpt["model"].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt["model"], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e


        if ckpt["optimizer"] is not None:
            if "best_map" in ckpt.keys():
                best_map = ckpt["best_map"]


        if ckpt.get("training_results") is not None:
            with open(results_file, "w") as file:
                file.write(ckpt["training_results"])  # write results.txt


        start_epoch = ckpt["epoch"] + 1
        if epochs < start_epoch:
            print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                  (opt.weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  

        if opt.amp and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])

        del ckpt


    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp["lrf"]) + hyp["lrf"]  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch  

    train_dataset = LoadImagesAndLabels(train_path, imgsz_train, batch_size,
                                        augment=True,
                                        hyp=hyp,  
                                        rect=opt.rect,  
                                        cache_images=opt.cache_images,
                                        single_cls=opt.single_cls,
                                        split='train')


    val_dataset = LoadImagesAndLabels(test_path, imgsz_test, batch_size,
                                      hyp=hyp,
                                      rect=True,  
                                      cache_images=opt.cache_images,
                                      single_cls=opt.single_cls,
                                      split='val')

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=nw,
                                                   shuffle=not opt.rect,
                                                   pin_memory=True,
                                                   collate_fn=train_dataset.collate_fn)

    val_datasetloader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=batch_size,
                                                    num_workers=nw,
                                                    pin_memory=True,
                                                    collate_fn=val_dataset.collate_fn)

    
    model.darknet.nc = nc  
    model.darknet.hyp = hyp  
    model.darknet.gr = 1.0  

    coco = get_coco_api_from_dataset(val_dataset)
    criterion = train_util.Criterion(opt)

    print("starting traning for %g epochs..." % epochs)
    print('Using %g dataloader workers' % nw)
    for epoch in range(start_epoch, epochs):
        mloss, lr = train_util.train_one_epoch(model, optimizer, train_dataloader, criterion,
                                               device, epoch,
                                               accumulate=accumulate, 
                                               img_size=imgsz_train,  
                                               multi_scale=multi_scale,
                                               grid_min=grid_min, 
                                               grid_max=grid_max, 
                                               gs=gs,  
                                               print_freq=50,  
                                               warmup=True,
                                               scaler=scaler)

        scheduler.step()

        if opt.notest is False or epoch == epochs - 1:

            result_info = train_util.evaluate(model, val_datasetloader,
                                              coco=coco, device=device)

            coco_mAP = result_info[0]
            voc_mAP = result_info[1]
            coco_mAR = result_info[8]


            if tb_writer:
                tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss', 'train/loss', "learning_rate",
                        "mAP@[IoU=0.50:0.95]", "mAP@[IoU=0.5]", "mAR@[IoU=0.50:0.95]"]

                for x, tag in zip(mloss.tolist() + [lr, coco_mAP, voc_mAP, coco_mAR], tags):
                    tb_writer.add_scalar(tag, x, epoch)


            with open(results_file, "a") as f:
                result_info = [str(round(i, 4)) for i in result_info + [mloss.tolist()[-1]]] + [str(round(lr, 6))]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")


            if voc_mAP > best_map:
                best_map = voc_mAP

            if opt.savebest is False:

                with open(results_file, 'r') as f:
                    save_files = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'training_results': f.read(),
                        'epoch': epoch,
                        'best_map': best_map}
                    if opt.amp:
                        save_files["scaler"] = scaler.state_dict()
                    torch.save(save_files, "./weights/yolov3spp-{}.pt".format(epoch))
            else:

                if best_map == voc_mAP:
                    with open(results_file, 'r') as f:
                        save_files = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'training_results': f.read(),
                            'epoch': epoch,
                            'best_map': best_map}
                        if opt.amp:
                            save_files["scaler"] = scaler.state_dict()
                        torch.save(save_files, best.format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=True,  action='store_false')
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--cfg', type=str, default="/media/Harddisk/jmy/SPG-Det/yolov3_spp/cfg/my_yolov3.cfg", help="*.cfg path")
    parser.add_argument('--data', type=str, default='data/my_data.data', help='*.data path')
    parser.add_argument('--hyp', type=str, default='cfg/hyp.yaml', help='hyperparameters path')
    parser.add_argument('--multi-scale', type=bool, default=True,
                        help='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--img-size', type=int, default=512, help='test size')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--savebest', type=bool, default=True, help='only save best checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default="",
                        help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--freeze-layers', type=bool, default=False, help='Freeze non-output layers')
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument('--normal_loss',default='cos', help='cos|mse')
    parser.add_argument('--normal_w',   default=1)
    opt = parser.parse_args()


    opt.cfg = check_file(opt.cfg)
    opt.data = check_file(opt.data)
    opt.hyp = check_file(opt.hyp)
    print(opt)

    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter(comment=opt.name)
    train(hyp)
