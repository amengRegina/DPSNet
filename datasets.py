from hashlib import new
import math
import os
import random
import shutil
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import xyxy2xywh, xywh2xyxy

help_url = 'https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data'
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']


# get orientation in exif tag
# 找到图像exif信息中对应旋转信息的key值
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


def exif_size(img):
    """
    获取图像的原始img size
    通过exif的orientation信息判断图像是否有旋转，如果有旋转则返回旋转前的size
    :param img: PIL图片
    :return: 原始图像的size
    """
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270  顺时针翻转90度
            s = (s[1], s[0])
        elif rotation == 8:  # ratation 90  逆时针翻转90度
            s = (s[1], s[0])
    except:
        # 如果图像的exif信息中没有旋转信息，则跳过
        pass

    return s


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self,
                 path1,   # 指向data/my_train_data.txt路径或data/my_val_data.txt路径
                 # 这里设置的是预处理后输出的图片尺寸
                 # 当为训练集时，设置的是训练过程中(开启多尺度)的最大尺寸
                 # 当为验证集时，设置的是最终使用的网络大小
                 img_size=416,
                 batch_size=16,
                 augment=False,  # 训练集设置为True(augment_hsv)，验证集设置为False
                 hyp=None,  # 超参数字典，其中包含图像增强会使用到的超参数
                 rect=False,  # 是否使用rectangular training
                 cache_images=False,  # 是否缓存图片到内存中
                 single_cls=False, pad=0.0, rank=-1,split='train'):
# ===========================self.img_files 样本文件夹名称==================================
        try:
            path = os.path.join(path1, split + '_mtrl.txt')
            # parent = str(Path(path).parent) + os.sep
            self.path1 = path1
            if os.path.isfile(path):  # file
                # 读取对应my_train/val_data.txt文件，读取每一行的图片路劲信息
                with open(path, "r") as f:
                    f = f.read().splitlines()
            else:
                raise Exception("%s does not exist" % path)

            # 检查每张图片后缀格式是否在支持的列表中，保存支持的图像路径
            # img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
            self.img_files = [x for x in f]
        except Exception as e:
            raise FileNotFoundError("Error loading data from {}. {}".format(path, e))
# ===========================self.img_files 样本文件夹名称路径 list==================================

        # 如果图片列表中没有图片，则报错
        n = len(self.img_files)
        assert n > 0, "No images found in %s. See %s" % (path, help_url)

        # batch index
        # 将数据划分到一个个batch中
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)
        # 记录数据集划分后的总batch数
        nb = bi[-1] + 1  # number of batches

        self.n = n  # number of images 图像总数目
        self.batch = bi  # batch index of image 记录哪些图片属于哪个batch
        self.img_size = img_size  # 这里设置的是预处理后输出的图片尺寸
        self.augment = augment  # 是否启用augment_hsv
        self.hyp = hyp  # 超参数字典，其中包含图像增强会使用到的超参数
        self.rect = rect  # 是否使用rectangular training
        # 注意: 开启rect后，mosaic就默认关闭
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)

        # Define labels
        # 遍历设置图像对应的label路径
        # (./my_yolo_dataset/train/images/2009_004012.jpg) -> (./my_yolo_dataset/train/labels/2009_004012.txt)

# ===========================self.label_files 样本标签label路径 list==================================
        self.label_files = [os.path.join(path1, 'Images', x , x+'.txt')
                            for x in self.img_files]
# ===========================self.label_files 样本标签label路径 list==================================
        #print(self.label_files)

# ===========================self.normal_file 光度立体图路径 list==================================
        self.normal_file = [os.path.join(path1, 'Images', x , x[:-14]+'normal.jpg')
                            for x in self.img_files]
# ===========================self.normal_file 光度立体图路径 list==================================

# ===========================self.mask_file 掩模路径 list==================================
        self.mask_file = [os.path.join(path1, 'Images', x , x[:-14]+'albedo_mask.jpg')
                            for x in self.img_files]
# ===========================self.mask_file 样本标签label list==================================

        # Read image shapes (wh)
        # 查看data文件下是否缓存有对应数据集的.shapes文件，里面存储了每张图像的width, height
        sp = path.replace(".txt", ".shapes")  # shapefile path
        try:
            with open(sp, "r") as f:  # read existing shapefile
                s = [x.split() for x in f.read().splitlines()]
                # 判断现有的shape文件中的行数(图像个数)是否与当前数据集中图像个数相等
                # 如果不相等则认为是不同的数据集，故重新生成shape文件
                assert len(s) == n, "shapefile out of aync"
        except Exception as e:
            # print("read {} failed [{}], rebuild {}.".format(sp, e, sp))
            # tqdm库会显示处理的进度
            # 读取每张图片的size信息
            if rank in [-1, 0]:
                image_files = tqdm(self.img_files, desc="Reading image shapes")
            else:
                image_files = self.img_files
            s = [exif_size(Image.open(f)) for f in image_files]
            # 将所有图片的shape信息保存在.shape文件中
            np.savetxt(sp, s, fmt="%g")  # overwrite existing (if any)


        # 记录每张图像的原始尺寸
# ===========================self.shapes 对应图片尺寸 ==================================
        self.shapes = np.array(s, dtype=np.int64)
# ===========================self.shapes 对应图片尺寸 ==================================
        
        # Rectangular Training https://github.com/ultralytics/yolov3/issues/232
        # 如果为ture，训练网络时，会使用类似原图像比例的矩形(让最长边为img_size)，而不是img_size x img_size
        # 注意: 开启rect后，mosaic就默认关闭
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            # 计算每个图片的高/宽比
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            # argsort函数返回的是数组值从小到大的索引值
            # 按照高宽比例进行排序，这样后面划分的每个batch中的图像就拥有类似的高宽比
            irect = ar.argsort()
            # 根据排序后的顺序重新设置图像顺序、标签顺序以及shape顺序
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.normal_file = [self.normal_file[i] for i in irect]
            self.mask_file = [self.mask_file[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # set training image shapes
            # 计算每个batch采用的统一尺度
            shapes = [[1, 1]] * nb  # nb: number of batches
            for i in range(nb):
                ari = ar[bi == i]  # bi: batch index
                # 获取第i个batch中，最小和最大高宽比
                mini, maxi = ari.min(), ari.max()

                # 如果高/宽小于1(w > h)，将w设为img_size
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                # 如果高/宽大于1(w < h)，将h设置为img_size
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]
            # 计算每个batch输入网络的shape值(向上设置为32的整数倍)
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / 32. + pad).astype(np.int) * 32

        # cache labels
        self.imgs = [None] * n  # n为图像总数
        # label: [class, x, y, w, h] 其中的xywh都为相对值
        self.labels = [np.zeros((0, 5), dtype=np.float32)] * n
        extract_bounding_boxes, labels_loaded = False, False
        nm, nf, ne, nd = 0, 0, 0, 0  # number mission, found, empty, duplicate
        # 这里分别命名是为了防止出现rect为False/True时混用导致计算的mAP错误
        # 当rect为True时会对self.images和self.labels进行从新排序
        if rect is True:
            np_labels_path = str(Path(self.label_files[0]).parent) + ".rect.npy"  # saved labels in *.npy file
        else:
            np_labels_path = str(Path(self.label_files[0]).parent) + ".norect.npy"

        if os.path.isfile(np_labels_path):
            x = np.load(np_labels_path, allow_pickle=True)
            if len(x) == n:
                # 如果载入的缓存标签个数与当前计算的图像数目相同则认为是同一数据集，直接读缓存
                self.labels = x
                labels_loaded = True

        # 处理进度条只在第一个进程中显示
        if rank in [-1, 0]:
            pbar = tqdm(self.label_files)
        else:
            pbar = self.label_files

        pbar = tqdm(self.label_files)
        # 遍历载入标签文件
        for i, file in enumerate(pbar):
            if labels_loaded is True:
                # 如果存在缓存直接从缓存读取
                l = self.labels[i]
            else:
                # 从文件读取标签信息
                try:
                    with open(file, "r") as f:
                        # 读取每一行label，并按空格划分数据
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                except Exception as e:
                    print("An error occurred while loading the file {}: {}".format(file, e))
                    nm += 1  # file missing
                    continue

            # 如果标注信息不为空的话
            if l.shape[0]:
                # 标签信息每行必须是五个值[class, x, y, w, h]
                assert l.shape[1] == 5, "> 5 label columns: %s" % file
                assert (l >= 0).all(), "negative labels: %s" % file
                assert (l[:, 1:] <= 1).all(), "non-normalized or out of bounds coordinate labels: %s" % file

                # 检查每一行，看是否有重复信息
                if np.unique(l, axis=0).shape[0] < l.shape[0]:  # duplicate rows
                    nd += 1
                if single_cls:
                    l[:, 0] = 0  # force dataset into single-class mode

                self.labels[i] = l
                nf += 1  # file found

                # # Extract object detection boxes for a second stage classifier
                if extract_bounding_boxes:
                    p = Path(self.img_files[i])
                    img = cv2.imread(str(p))
                    h, w = img.shape[:2]
                    for j, x in enumerate(l):
                        f = "%s%sclassifier%s%g_%g_%s" % (p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                        if not os.path.exists(Path(f).parent):
                            os.makedirs(Path(f).parent)  # make new output folder

                        # 将相对坐标转为绝对坐标
                        # b: x, y, w, h
                        b = x[1:] * [w, h, w, h]  # box
                        # 将宽和高设置为宽和高中的最大值
                        b[2:] = b[2:].max()  # rectangle to square
                        # 放大裁剪目标的宽高
                        b[2:] = b[2:] * 1.3 + 30  # pad
                        # 将坐标格式从 x,y,w,h -> xmin,ymin,xmax,ymax
                        b = xywh2xyxy(b.reshape(-1, 4)).revel().astype(np.int)

                        # 裁剪bbox坐标到图片内
                        b[[0, 2]] = np.clip[b[[0, 2]], 0, w]
                        b[[1, 3]] = np.clip[b[[1, 3]], 0, h]
                        assert cv2.imwrite(f, img[b[1]:b[3], b[0]:b[2]]), "Failure extracting classifier boxes"
            else:
                ne += 1  # file empty

            # 处理进度条只在第一个进程中显示
            if rank in [-1, 0]:
                # 更新进度条描述信息
                pbar.desc = "Caching labels (%g found, %g missing, %g empty, %g duplicate, for %g images)" % (
                    nf, nm, ne, nd, n)
        assert nf > 0, "No labels found in %s." % os.path.dirname(self.label_files[0]) + os.sep

        # # 如果标签信息没有被保存成numpy的格式，且训练样本数大于1000则将标签信息保存成numpy的格式
        if not labels_loaded and n > 1000:
            print("Saving labels to %s for faster future loading" % np_labels_path)
            np.save(np_labels_path, self.labels)  # save for next time

        # Cache images into memory for faster training (Warning: large datasets may exceed system RAM)
        if cache_images:  # if training
            gb = 0  # Gigabytes of cached images 用于记录缓存图像占用RAM大小
            if rank in [-1, 0]:
                pbar = tqdm(range(len(self.img_files)), desc="Caching images")
            else:
                pbar = range(len(self.img_files))

            self.img_hw0, self.img_hw = [None] * n, [None] * n
            for i in pbar:  # max 10k images
                self.imgs[i], self.img_hw0[i], self.img_hw[i], self.normal,self.mask = load_image(self, i)  # img, hw_original, hw_resized
                gb += self.imgs[i].nbytes  # 用于记录缓存图像占用RAM大小
                if rank in [-1, 0]:
                    pbar.desc = "Caching images (%.1fGB)" % (gb / 1E9)

        # Detect corrupted images https://medium.com/joelthchao/programmatically-detect-corrupted-image-8c1b2006c3d3
        detect_corrupted_images = False
        if detect_corrupted_images:
            from skimage import io  # conda install -c conda-forge scikit-image
            for file in tqdm(self.img_files, desc="Detecting corrupted images"):
                try:
                    _ = io.imread(file)
                except Exception as e:
                    print("Corrupted image detected: {}, {}".format(file, e))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        hyp = self.hyp
        if self.mosaic:
            # load mosaic
            img, normal, mask, labels = load_mosaic(self, index)
            shapes = None
        else:
            # load image
            img, (h0, w0), (h, w), normal, mask = load_image(self, index) # img(4, 703, 960, 3) normal(703, 960, 3) mask(703, 960)
 
            # letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, normal, mask, ratio, pad = letterbox(img, normal, mask, shape, auto=False, scale_up=self.augment) # [1, 4, 960, 960, 3] (960, 960, 3) (960, 960)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
            

            # load labels
            labels = []
            x = self.labels[index] # cx cy w h

            if x.size > 0:
                # Normalized xywh to pixel xyxy format cx,cy->xyxy
                labels = x.copy()  # label: class, x, y, w, h
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        if self.augment:
            # Augment imagespace
            if not self.mosaic:
                img, normal, mask, labels = random_affine(img, normal, mask, labels,
                                            degrees=hyp["degrees"],
                                            translate=hyp["translate"],
                                            scale=hyp["scale"],
                                            shear=hyp["shear"])

            # Augment colorspace
            img = np.array(img)
            # augment_hsv(img, h_gain=hyp["hsv_h"], s_gain=hyp["hsv_s"], v_gain=hyp["hsv_v"])

        nL = len(labels)  # number of labels
        if nL:
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0-1
            labels[:, [2, 4]] /= img.shape[1]  # height
            labels[:, [1, 3]] /= img.shape[2]  # width

        
        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                iimg=np.zeros((img.shape[0], img.shape[1], img.shape[2], img.shape[3]))
                for i in range(img.shape[0]):
                    iimg[i, :, :, :] = np.fliplr(img[i, :, :, :])
                img = iimg
                normal = np.fliplr(normal)
                mask = np.fliplr(mask)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]  # 1 - x_center

            # random up-down flip
            ud_flip = True
            if ud_flip and random.random() < 0.5:
                iimg=np.zeros((img.shape[0], img.shape[1], img.shape[2], img.shape[3]))
                for i in range(img.shape[0]):
                    iimg[i, :, :, :] = np.flipud(img[i, :, :, :])
                img = iimg
                normal = np.flipud(normal)
                mask = np.flipud(mask)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]  # 1 - y_center
        
        labels_out = torch.zeros((nL, 6))  # nL: number of labels
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)
        
        # Convert BGR to RGB, and HWC to CHW(3x512x512)
        img = np.array(img)
        img = img.transpose(0, 3, 1, 2) # b c w h
        img = np.ascontiguousarray(img)

        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        mask = mask.repeat(3, 2)
        

        normal = normal.transpose(2, 0, 1).copy()
        mask = mask.transpose(2, 0, 1).copy()
       
        # cv2.imwrite('./1img.jpg',np.asarray(img[0]).transpose(1, 2, 0))
        # cv2.imwrite('./1normal.jpg',np.asarray(normal).transpose(1, 2, 0))
        # cv2.imwrite('./1masks.jpg',np.asarray(mask).transpose(1, 2, 0))

        item = {'img': torch.from_numpy(img), 'target': labels_out, 'path':self.img_files[index],'shape':shapes,'index':index,'normal':torch.from_numpy(normal),'mask':torch.from_numpy(mask)}
        return item
        
    def coco_index(self, index):
        """该方法是专门为cocotools统计标签信息准备，不对图像和标签作任何处理"""
        o_shapes = self.shapes[index][::-1]  # wh to hw

        # load labels
        labels = []
        x = self.labels[index]
        if x.size > 0:
            labels = x.copy()  # label: class, x, y, w, h
        return torch.from_numpy(labels), o_shapes
    
    @staticmethod
    def collate_fn(batches):
        inputs = []
        targets = []
        normals = []
        masks = []
        paths = []
        shapes = []
        indexs = [] 
        for b_item in batches:
            input, target, path ,shape ,index ,normal, mask= b_item['img'], b_item['target'], b_item['path'] ,b_item['shape'], b_item['index'] ,b_item['normal'],b_item['mask']
            inputs.append(input)
            targets.append(target)
            normals.append(normal)
            masks.append(mask)
            paths.append(path)
            shapes.append(shape)
            indexs.append(index)
        for i, l in enumerate(targets):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(inputs, 0), torch.cat(targets, 0), paths, shapes, indexs, torch.stack(normals, 0), torch.stack(masks, 0), 
    

def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    # img = self.imgs[index]    
    # if img is None:  # not cached
    path = self.img_files[index]
    data1 = os.path.join(self.path1,'Images',path,'filenames.txt')
    normal_path = self.normal_file[index]
    mask_path = self.mask_file[index]
    normal = cv2.imread(normal_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
       
    with open(data1, "r") as f:  # read existing shapefile for pictures
        img_list = [x for x in f.read().splitlines()]
        imgs   =  []
        
        for i in img_list:
            img = cv2.imread(os.path.join(self.path1,'Images',path,i))
            assert img is not None, "Image Not Found " + path + i
            # img = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
            h0, w0 = img.shape[:2]  # orig hw 3000 4096

# =========================== 图片缩放 ==================================
            # img_size 设置的是预处理后输出的图片尺寸
            r = self.img_size / max(h0, w0)  # resize image to img_size 960 / max(3000 4096)
            if r != 1:  # always resize down, only resize up if training with augmentation
                interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
                img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
# =========================== 图片缩放 ==================================
    
            imgs.append(img)
        img = np.stack(imgs, axis=0)
        # mask = mask.reshape(mask.shape[0], mask.shape[1], 1) 
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            normal = cv2.resize(normal, (int(w0 * r), int(h0 * r)), interpolation=interp)
            mask = cv2.resize(mask, (int(w0 * r), int(h0 * r)), interpolation=interp)
    assert img is not None, "Image Not Found " + data1
    # print(img.shape, normal.shape, mask.shape) #  (4, 703, 960, 3) (703, 960, 3)(703, 960)
    return img, (h0, w0), img.shape[1:3], normal, mask # img, hw_original, hw_resized

    # else:
    #     return self.imgs[index], self.img_hw0[index], self.img_hw[index], self.normal,self.mask  # img, hw_original, hw_resized


def load_mosaic(self, index):
    """
    将四张图片拼接在一张马赛克图像中
    :param self:
    :param index: 需要获取的图像索引
    :return:
    """
    # loads images in a mosaic

    labels4 = []  # 拼接图像的label信息
    s = self.img_size
    # 随机初始化拼接图像的中心点坐标
    xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
    # 从dataset中随机寻找三张图像进行拼接
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  # 3 additional image indices
    # 遍历四张图像进行拼接
    for i, index in enumerate(indices):
        # load image
        img, _, (h, w), normal, mask= load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            # 创建马赛克图像
            img4 = np.full((img.shape[0],s * 2, s * 2, img.shape[3]), 114, dtype=np.uint8)  # base image with 4 tiles
            img1 = np.full((s * 2, s * 2, img.shape[3]), 114, dtype=np.uint8)  # base image with 4 tiles
            img2 = np.full((s * 2, s * 2), 114, dtype=np.uint8)  # base image with 4 tiles
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            # 计算截取的图像区域信息(以xc,yc为第一张图像的右下角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            # 计算截取的图像区域信息(以xc,yc为第二张图像的左下角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            # 计算截取的图像区域信息(以xc,yc为第三张图像的右上角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            # 计算截取的图像区域信息(以xc,yc为第四张图像的左上角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        # 将截取的图像区域填充到马赛克图像的相应位置
        img4[:,y1a:y2a, x1a:x2a] = img[:,y1b:y2b, x1b:x2b]  # img4[:,ymin:ymax, xmin:xmax]
        img1[y1a:y2a, x1a:x2a] = normal[y1b:y2b, x1b:x2b]
        img2[y1a:y2a, x1a:x2a] = mask[y1b:y2b, x1b:x2b]
        # 计算pad(图像边界与马赛克边界的距离，越界的情况为负值)
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels 获取对应拼接图像的labels信息
        x = self.labels[index]
        labels = x.copy()  # 深拷贝，防止修改原数据
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            # 计算标注数据在马赛克图像中的
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

    # Augment
    # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
    img4, normal, mask, labels4 = random_affine(img4, img1, img2,labels4,
                                  degrees=self.hyp['degrees'],
                                  translate=self.hyp['translate'],
                                  scale=self.hyp['scale'],
                                  shear=self.hyp['shear'],
                                  border=-s // 2)  # border to remove

    return img4, normal, mask, labels4


def random_affine(img, normal, mask, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    # targets = [cls, xyxy]

    # 给定的输入图像的尺寸(416/512/640)，等于img4.shape / 2
    height = img.shape[1] + border * 2
    width = img.shape[2] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[2] / 2, img.shape[1] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[2] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    imgs   =  []
    k = []
    for i in range(img.shape[0]):
        if (border != 0) or (M != np.eye(3)).any():  # image changed
            k = cv2.warpAffine(img[i], M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
            imgs.append(k)
    if k!=[]:
        img = imgs
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        normal = cv2.warpAffine(normal, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
        mask = cv2.warpAffine(mask, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        # [4*n, 3] -> [n, 8]
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]  # [n, 4]
        y = xy[:, [1, 3, 5, 7]]  # [n, 4]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T  # [n, 4]

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        # 对坐标进行裁剪，防止越界
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]

        # 计算调整后的每个box的面积
        area = w * h
        # 计算调整前的每个box的面积
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        # 计算每个box的比例
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        # 选取长宽大于4个像素，且调整前后面积比例大于0.2，且比例小于10的box
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, normal, mask, targets


def augment_hsv(img, h_gain=0.5, s_gain=0.5, v_gain=0.5):
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1  # random gains  
    for i in range(img.shape[0]):
        dtype = img[i].dtype  # uint8
        hue, sat, val = cv2.split(cv2.cvtColor(img[i], cv2.COLOR_BGR2HSV))
        

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img[i])  # no return needed

    # Histogram equalization
    # if random.random() < 0.2:
    #     for i in range(3):
    #         img[:, :, i] = cv2.equalizeHist(img[:, :, i])


def letterbox(img: np.ndarray,
              normal: np.ndarray,
              mask: np.ndarray,
              new_shape=(416, 416),
              color=(0, 0, 0),
              auto=True,
              scale_fill=False,
              scale_up=True):
    """
    将图片缩放调整到指定大小
    :param img:
    :param new_shape:
    :param color:
    :param auto:
    :param scale_fill:
    :param scale_up:
    :return:
    """

    shape = img.shape[1:3]  # [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:  # only scale down, do not scale up (for better test mAP) 对于大于指定输入大小的图片进行缩放,小于的不变
        r = min(r, 1.0)

    # compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimun rectangle 保证原图比例不变，将图像最大边缩放到指定大小
        # 这里的取余操作可以保证padding后的图片是32的整数倍
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scale_fill:  # stretch 简单粗暴的将图片缩放到指定尺寸
        dw, dh = 0, 0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # wh ratios

    dw /= 2  # divide padding into 2 sides 将padding分到上下，左右两侧
    dh /= 2

    # shape:[h, w]  new_unpad:[w, h]
    # print(shape[::-1],new_unpad )
    # BUG: 这儿可能报错
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        normal = cv2.resize(normal, new_unpad, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 计算上下两侧的padding
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 计算左右两侧的padding
    #print(new_shape)
    iimg=np.zeros((img.shape[0], new_shape[0], new_shape[1], img.shape[-1]))
    for i in range(img.shape[0]):
        iimg[i, :, :, :]= cv2.copyMakeBorder(img[i, :, :, :], top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    normal = cv2.copyMakeBorder(normal, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return iimg, normal, mask, ratio, (dw, dh)


def create_folder(path="./new_folder"):
    # Create floder
    if os.path.exists(path):
        shutil.rmtree(path)  # dalete output folder
    os.makedirs(path)  # make new output folder


def normalToMask(normal, thres=0.01):
    """
    Due to the numerical precision of uint8, [0, 0, 0] will save as [127, 127, 127] in gt normal,
    When we load the data and rescale normal by N / 255 * 2 - 1, [127, 127, 127] becomes 
    [-0.003927, -0.003927, -0.003927]
    """
    mask = (np.square(normal).sum(2, keepdims=True) > thres).astype(np.float32)
    return mask

