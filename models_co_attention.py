from build_utils.layers import *
from build_utils.parse_config import *
from torch.nn.init import kaiming_normal_
from train_utils import model_utils
import torchvision.utils as vutils
ONNX_EXPORT = False

class Regressor(nn.Module):
    def __init__(self, batchNorm=True): 
        super(Regressor, self).__init__()
        self.deconv1 = model_utils.conv(batchNorm, 128, 128,  k=3, stride=1, pad=1)
        self.deconv2 = model_utils.conv(batchNorm, 128, 128,  k=3, stride=1, pad=1)
        self.deconv5 = model_utils.deconv(128, 128)
        self.deconv3 = model_utils.conv(batchNorm, 128, 64,  k=3, stride=1, pad=1)
        self.deconv4 = model_utils.deconv(64, 32)

        self.est_normal= self._make_output(32, 3, k=3, stride=1, pad=1)


    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
               nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

    def forward(self, x):
        out    = self.deconv1(x)
        out    = self.deconv2(out)
        out    = self.deconv5(out)
        out    = self.deconv3(out)
        out    = self.deconv4(out)
        normal = self.est_normal(out)
        # normal = normal * mask
        # cv2.imwrite('./normal_pred.jpg',np.asarray(((normal.permute(0, 2, 3, 1).contiguous() + 1)*255.0/2)[3].cpu().detach()))
        normal = torch.nn.functional.normalize(normal, 2, 1)
        return normal
        
class SPG_Det(nn.Module):
    def __init__(self, cfg, fuse_type='max', batchNorm=False, c_in=3, other={}, img_size=(416, 416)):
        super(SPG_Det, self).__init__()
        self.extractor = Darknet(cfg, img_size, only_front=True)
        self.regressor = Regressor(batchNorm)
        self.darknet = Darknet(cfg, img_size)
        self.c_in      = c_in
        self.fuse_type = fuse_type
        self.other = other

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x ,mask):
       
        b, ib, c, h, w = x.shape
        extractor_in = x.reshape(ib*b, c, h, w)
        feat = self.extractor(extractor_in)
        _, c_f, h_f, w_f = feat.shape
        extractor_out = feat.reshape(b, ib , c_f, h_f, w_f)
        cv2.imwrite('./input_feature1.jpg',np.asarray(((extractor_out[0].permute(0, 2, 3, 1).contiguous() + 1)*255.0/2)[0][:,:,:3].cpu().detach()))
        cv2.imwrite('./input_feature2.jpg',np.asarray(((extractor_out[0].permute(0, 2, 3, 1).contiguous() + 1)*255.0/2)[1][:,:,:3].cpu().detach()))
        cv2.imwrite('./input_feature3.jpg',np.asarray(((extractor_out[0].permute(0, 2, 3, 1).contiguous() + 1)*255.0/2)[2][:,:,:3].cpu().detach()))
        cv2.imwrite('./input_feature4.jpg',np.asarray(((extractor_out[0].permute(0, 2, 3, 1).contiguous() + 1)*255.0/2)[3][:,:,:3].cpu().detach()))
        if self.fuse_type == 'mean':
            feat_fused = torch.mean(extractor_out, dim=1, keepdim=False)
        elif self.fuse_type == 'max':
            feat_fused= torch.max(extractor_out, dim=1, keepdim=False)
        normal = self.regressor(feat_fused.values)
        normal_GL = (normal + 1) / 2 
        normal_GL = normal_GL * mask
        # vutils.save_image(normal_GL, './normal_pred_xunlian.jpg')  
        if self.training:
            # print(normal.shape) #[1, 3, 960, 960]
            x = self.darknet(normal_GL, extractor_out)
            return x, normal 
        else:
            x, p =self.darknet(normal_GL, extractor_out)
            return x, p, normal

def create_modules(modules_defs: list, img_size):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    :param modules_defs: 通过.cfg文件解析得到的每个层结构的列表
    :param img_size:
    :return:
    """

    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size
    # 删除解析cfg列表中的第一个配置(对应[net]的配置)
    modules_defs.pop(0)  # cfg training hyperparams (unused)
    output_filters = [3]  # input channels
    module_list = nn.ModuleList()
    # 统计哪些特征层的输出会被后续的层使用到(可能是特征融合，也可能是拼接)
    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1

    # 遍历搭建每个层结构
    for i, mdef in enumerate(modules_defs):
        modules = nn.Sequential()

        if mdef["type"] == "convolutional":
            bn = mdef["batch_normalize"]  # 1 or 0 / use or not
            filters = mdef["filters"]
            k = mdef["size"]  # kernel size
            stride = mdef["stride"] if "stride" in mdef else (mdef['stride_y'], mdef["stride_x"])
            if isinstance(k, int):
                modules.add_module("Conv2d", nn.Conv2d(in_channels=output_filters[-1],
                                                       out_channels=filters,
                                                       kernel_size=k,
                                                       stride=stride,
                                                       padding=k // 2 if mdef["pad"] else 0,
                                                       bias=not bn))
            else:
                raise TypeError("conv2d filter size must be int type.")

            if bn:
                modules.add_module("BatchNorm2d", nn.BatchNorm2d(filters))
            else:
                # 如果该卷积操作没有bn层，意味着该层为yolo的predictor
                routs.append(i)  # detection output (goes into yolo layer)

            if mdef["activation"] == "leaky":
                modules.add_module("activation", nn.LeakyReLU(0.1, inplace=True))
            else:
                pass

        elif mdef["type"] == "BatchNorm2d":
            pass

        elif mdef["type"] == "maxpool":
            k = mdef["size"]  # kernel size
            stride = mdef["stride"]
            modules = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)

        elif mdef["type"] == "upsample":
            if ONNX_EXPORT:  # explicitly state size, avoid scale_factor
                g = (yolo_index + 1) * 2 / 32  # gain
                modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))
            else:
                modules = nn.Upsample(scale_factor=mdef["stride"])

        elif mdef["type"] == "route":  # [-2],  [-1,-3,-5,-6], [-1, 61]
            layers = mdef["layers"]
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat(layers=layers)

        elif mdef["type"] == "shortcut":
            layers = mdef["from"]
            filters = output_filters[-1]
            # routs.extend([i + l if l < 0 else l for l in layers])
            routs.append(i + layers[0])
            modules = WeightedFeatureFusion(layers=layers, weight="weights_type" in mdef)

        elif mdef["type"] == "yolo":
            yolo_index += 1  # 记录是第几个yolo_layer [0, 1, 2]
            stride = [32, 16, 8]  # 预测特征层对应原图的缩放比例

            modules = YOLOLayer(anchors=mdef["anchors"][mdef["mask"]],  # anchor list
                                nc=mdef["classes"],  # number of classes
                                img_size=img_size,
                                stride=stride[yolo_index])

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            try:
                j = -1
                # bias: shape(255,) 索引0对应Sequential中的Conv2d
                # view: shape(3, 85)
                b = module_list[j][0].bias.view(modules.na, -1)
                b.data[:, 4] += -4.5  # obj
                b.data[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
                module_list[j][0].bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            except Exception as e:
                print('WARNING: smart bias initialization failure.', e)
        else:
            print("Warning: Unrecognized Layer Type: " + mdef["type"])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    routs_binary = [False] * len(modules_defs)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary


class YOLOLayer(nn.Module):
    """
    对YOLO的输出进行处理
    """
    def __init__(self, anchors, nc, img_size, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.stride = stride  # layer stride 特征图上一步对应原图上的步距 [32, 16, 8]
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85: x, y, w, h, obj, cls1, ...)
        self.nx, self.ny, self.ng = 0, 0, (0, 0)  # initialize number of x, y gridpoints
        # 将anchors大小缩放到grid尺度
        self.anchor_vec = self.anchors / self.stride
        # batch_size, na, grid_h, grid_w, wh,
        # 值为1的维度对应的值不是固定值，后续操作可根据broadcast广播机制自动扩充
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)
        self.grid = None

        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))  # number x, y grid points

    def create_grids(self, ng=(13, 13), device="cpu"):
        """
        更新grids信息并生成新的grids参数
        :param ng: 特征图大小
        :param device:
        :return:
        """
        self.nx, self.ny = ng
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets 构建每个cell处的anchor的xy偏移量(在feature map上的)
        if not self.training:  # 训练模式不需要回归到最终预测boxes
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device),
                                     torch.arange(self.nx, device=device)])
            # batch_size, na, grid_h, grid_w, wh
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p):
        if ONNX_EXPORT:
            bs = 1  # batch size
        else:
            bs, _, ny, nx = p.shape  # batch_size, predict_param(255), grid(13), grid(13)
            if (self.nx, self.ny) != (nx, ny) or self.grid is None:  # fix no grid bug
                self.create_grids((nx, ny), p.device)

        # view: (batch_size, 255, 13, 13) -> (batch_size, 3, 85, 13, 13)
        # permute: (batch_size, 3, 85, 13, 13) -> (batch_size, 3, 13, 13, 85)
        # [bs, anchor, grid, grid, xywh + obj + classes]
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p
        elif ONNX_EXPORT:
            # Avoid broadcasting for ANE operations
            m = self.na * self.nx * self.ny  # 3*
            ng = 1. / self.ng.repeat(m, 1)
            grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng

            p = p.view(m, self.no)
            # xy = torch.sigmoid(p[:, 0:2]) + grid  # x, y
            # wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            # p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else \
            #     torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])  # conf
            p[:, :2] = (torch.sigmoid(p[:, 0:2]) + grid) * ng  # x, y
            p[:, 2:4] = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            p[:, 4:] = torch.sigmoid(p[:, 4:])
            p[:, 5:] = p[:, 5:self.no] * p[:, 4:5]
            return p
        else:  # inference
            # [bs, anchor, grid, grid, xywh + obj + classes]
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy 计算在feature map上的xy坐标
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method 计算在feature map上的wh
            io[..., :4] *= self.stride  # 换算映射回原图尺度
            torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]


class Darknet(nn.Module):
    """
    YOLOv3 spp object detection model
    """
    def __init__(self, cfg, img_size=(416, 416), verbose=False, only_front = False ):
        super(Darknet, self).__init__()
        # 这里传入的img_size只在导出ONNX模型时起作用
        self.input_size = [img_size] * 2 if isinstance(img_size, int) else img_size
        # 解析网络对应的.cfg文件
        self.module_defs = parse_model_cfg(cfg)

        self.con_attention1 = model_utils.conv(batchNorm=True, cin= 128, cout= 128,  k=1, stride=1, pad=0)
        self.con_attention2 = model_utils.conv(batchNorm=True, cin= 128, cout= 128,  k=1, stride=1, pad=0)
        self.con_attention3 = model_utils.conv(batchNorm=True, cin= 128, cout= 128,  k=1, stride=1, pad=0)
        self.con_attention4 = model_utils.conv(batchNorm=True, cin= 128, cout= 128,  k=1, stride=1, pad=0)
        
        # 根据解析的网络结构一层一层去搭建
        self.module_list, self.routs = create_modules(self.module_defs, img_size)
        # 获取所有YOLOLayer层的索引
        self.yolo_layers = get_yolo_layers(self)

        self.only_front = only_front
        self.verbose = verbose
    


        # 打印下模型的信息，如果verbose为True则打印详细信息
        self.info() if not ONNX_EXPORT else None  # print model description

    def forward(self, x, I=None):
        return self.forward_once(x, I)

    def forward_once(self, x, I, verbose=False):
        # yolo_out收集每个yolo_layer层的输出
        # out收集每个模块的输出
        yolo_out, out = [], []
        if verbose:
            print('0', x.shape)
            str = ""

        if self.only_front:
            self.module_list = self.module_list[:6]

        xq = []
        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if i == 6:
                cv2.imwrite('./normal_feature.jpg',np.asarray(((x.permute(0, 2, 3, 1).contiguous() + 1)*255.0/2)[0][:,:,:3].cpu().detach()))
                xq.append(self.con_attention1(I[:,0,:,:,:]))
                xq.append(self.con_attention2(I[:,1,:,:,:]))
                xq.append(self.con_attention3(I[:,2,:,:,:]))
                xq.append(self.con_attention4(I[:,3,:,:,:]))
                xq = torch.stack(xq, dim=1)
                x = torch.add(x, torch.mean(xq, dim=1, keepdim=False))
                # x = torch.add(x, torch.max(xq, dim=1, keepdim=False).values)
                cv2.imwrite('./modified_feature1.jpg',np.asarray(((self.con_attention1(I[:,0,:,:,:]).permute(0, 2, 3, 1).contiguous() + 1)*255.0/2)[0][:,:,:3].cpu().detach()))
                cv2.imwrite('./modified_feature2.jpg',np.asarray(((self.con_attention1(I[:,1,:,:,:]).permute(0, 2, 3, 1).contiguous() + 1)*255.0/2)[0][:,:,:3].cpu().detach()))
                cv2.imwrite('./modified_feature3.jpg',np.asarray(((self.con_attention1(I[:,2,:,:,:]).permute(0, 2, 3, 1).contiguous() + 1)*255.0/2)[0][:,:,:3].cpu().detach()))
                cv2.imwrite('./modified_feature4.jpg',np.asarray(((self.con_attention1(I[:,3,:,:,:]).permute(0, 2, 3, 1).contiguous() + 1)*255.0/2)[0][:,:,:3].cpu().detach()))
                cv2.imwrite('./fused_feature.jpg',np.asarray(((x.permute(0, 2, 3, 1).contiguous() + 1)*255.0/2)[0][:,:,:3].cpu().detach()))
            if name in ["WeightedFeatureFusion", "FeatureConcat"]:  # sum, concat
                if verbose:
                    l = [i - 1] + module.layers  # layers
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
                    str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, sh)])
                x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
            elif name == "YOLOLayer":
                yolo_out.append(module(x))
            else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
                x = module(x)

            out.append(x if self.routs[i] else [])
            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), name), list(x.shape), str)
                str = ''
            



        if self.training:  # train
            if self.only_front:
                return x
            return yolo_out
        elif ONNX_EXPORT:  # export
            # x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            # return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
            p = torch.cat(yolo_out, dim=0)

            # # 根据objectness虑除低概率目标
            # mask = torch.nonzero(torch.gt(p[:, 4], 0.1), as_tuple=False).squeeze(1)
            # # onnx不支持超过一维的索引（pytorch太灵活了）
            # # p = p[mask]
            # p = torch.index_select(p, dim=0, index=mask)
            #
            # # 虑除小面积目标，w > 2 and h > 2 pixel
            # # ONNX暂不支持bitwise_and和all操作
            # mask_s = torch.gt(p[:, 2], 2./self.input_size[0]) & torch.gt(p[:, 3], 2./self.input_size[1])
            # mask_s = torch.nonzero(mask_s, as_tuple=False).squeeze(1)
            # p = torch.index_select(p, dim=0, index=mask_s)  # width-height 虑除小目标
            #
            # if mask_s.numel() == 0:
            #     return torch.empty([0, 85])

            return p
        else:  # inference or test
            if self.only_front:
                return x
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs

            return x, p

    def info(self, verbose=False):
        """
        打印模型的信息
        :param verbose:
        :return:
        """
        torch_utils.model_info(self, verbose)


def get_yolo_layers(self):
    """
    获取网络中三个"YOLOLayer"模块对应的索引
    :param self:
    :return:
    """
    return [i for i, m in enumerate(self.module_list) if m.__class__.__name__ == 'YOLOLayer']  # [89, 101, 113]


