# -*- coding: utf-8 -*-
import collections
import math
import random
import re
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from IPython import display
from torch import nn
from torch.utils import data
from torchvision import datasets, transforms

d2l = sys.modules[__name__]


# -------------------- data  -------------------


def load_array(tensors, batch_size, is_train=True):
    """使用tensors创建data iter"""
    dataset = data.TensorDataset(*tensors)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def load_data_fashion_mnist(batch_size, resize=None):
    """加载FashionMNIST."""
    # 定义transforms，肯定要ToTensor，Resize is optional
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    # 下载数据
    train_set = datasets.FashionMNIST(root='../data', train=True, transform=trans, download=True)
    test_set = datasets.FashionMNIST(root='../data', train=False, transform=trans, download=True)
    # dataset to data_iter
    return (data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4),
            data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4))


# -------------------- utils  -------------------


def try_gpu():
    """尽量使用gpu"""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# defined in 03.training pipeline
class Accumulator:
    """累计n个数据"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# -------------------- plot  -------------------


def use_svg_display():
    """使用svg格式"""
    display.set_matplotlib_formats('svg')


# defined in 03.training pipeline
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置坐标轴"""
    # 设置坐标标签
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    # 设置比例尺，{`linear`, `log`, ...}
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    # 设置x轴和y轴的显示范围
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    # 加上图例、网格
    if legend:
        axes.legend(legend)
    axes.grid()


# defined in 03.training pipeline
class Animator:
    """动态画折线图"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5)):
        """参数都是 matplotlib 画图的参数"""
        # 使用svg格式
        d2l.use_svg_display()
        # 获得画布和坐标轴
        self.fig, self.axes = plt.subplots(figsize=figsize)
        # config_axes() 即 d2l.set_axes(self.axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.config_axes = lambda: d2l.set_axes(self.axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        """Add multiple data points into the figure"""
        if not hasattr(y, "__len__"):
            y = [y]
        # Total n curves
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        # initialization
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        # 添加数据
        for i, (a, b) in enumerate(zip(x, y)):
            self.X[i].append(a)
            self.Y[i].append(b)
        self.axes.cla()  # 清除子图目前状态，防止重叠
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes.plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        # 不是多图而是动态
        display.clear_output(wait=True)


def plot(X, Y, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """
    画折线图
    参数都是matplotlib画图的参数
    """
    d2l.use_svg_display()
    plt.rcParams['figure.figsize'] = figsize
    # 可自定义坐标轴
    axes = axes if axes else plt.gca()

    # Return True if `Z` (tensor or list) has 1 axis
    def has_one_axis(Z):
        return (hasattr(Z, "ndim") and Z.ndim == 1 or
                isinstance(Z, list) and not hasattr(Z[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if has_one_axis(Y):
        Y = [Y]
    if len(X) == 1:
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x, y, fmt)
    d2l.set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


def annotate(text, xy, xytext):
    """
    画箭头做标注
    :param text: 文本
    :param xy: 要指向的位置
    :param xytext: 文本的位置
    """
    plt.gca().annotate(text, xy=xy, xytext=xytext,
                       arrowprops=dict(arrowstyle='->'))


# -------------------- computation  -------------------


def correct_predictions(y_hat, y):
    """
    :param y_hat: (n_samples, n_categories)
    :param y: (n_samples, )
    :return: 正确预测的个数
    """
    y_hat = y_hat.argmax(axis=1)  # across columns
    is_correct = y_hat.type(y.dtype) == y
    return float(is_correct.type(y.dtype).sum())


# defined in 03.training pipeline
def accuracy(net, data_iter, device):
    """
    :param net: 模型
    :param data_iter: 图像分类数据集
    :param device: 尽量使用GPU
    :return: 模型的准确率，这里使用了Accumulator和correct_predictions
    """
    net.eval()  # Set the model to evaluation mode
    metric = d2l.Accumulator(2)  # No. of correct predictions, no. of predictions
    # 预测时需no_grad
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            # y.numel()表示y中的数据数
            metric.add(d2l.correct_predictions(net(X), y), y.numel())
    return metric[0] / metric[1]


# -------------------- training  -------------------


def train_image_classifier(net, train_iter, test_iter, learning_rate, num_epochs):
    """
    训练图像分类器，记录数据并打印
    e.g. training FashionMNIST
    """
    device = d2l.try_gpu()
    # 需模型和数据均转向device
    net.to(device=device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # 记录误差和、正确预测样本数、总样本数
    metric = d2l.Accumulator(3)
    # 画训练误差、训练准确率、测试准确率
    animator = d2l.Animator(xlabel="epoch", xlim=[1, num_epochs], ylim=[0, 1],
                            legend=["train_loss", "train_acc", "test_acc"])
    for epoch in range(num_epochs):
        net.train()  # 因为计算accuracy会使net转向eval模式
        metric.reset()
        for x, y in train_iter:
            # Compute prediction error
            x, y = x.to(device), y.to(device)
            y_hat = net(x)
            loss = loss_fn(y_hat, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 记录数据
            metric.add(float(loss) * len(y), d2l.correct_predictions(y_hat, y), y.numel())
        # 画图
        animator.add(epoch + 1,
                     (metric[0] / metric[2], metric[1] / metric[2], d2l.accuracy(net, test_iter, device)))
    # 打印最终的数据
    print(f"loss {animator.Y[0][-1]:.3f}, "
          f"train acc {animator.Y[1][-1]:3f}, "
          f"test acc {animator.Y[2][-1]: 3f}")


def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes


def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes


def bbox_to_rect(bbox, color):
    """
    将边界框的(左上x, 左上y, 右下x, 右下y)格式转换成matplotlib的((左上x, 左上y), 宽, 高)格式
    """
    return plt.Rectangle(xy=(bbox[0], bbox[1]),
                         width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
                         fill=False, edgecolor=color, linewidth=2)


def generate_anchor_boxes(images, sizes, ratios):
    """
    生成以每个像素为中心具有不同形状的锚框
    :param images: shape is (batch_size, num_channels, h, w)
    :param sizes: 一系列相对比例，以高为基准
    :param ratios: 一系列宽高比
    :return: shape is (1, h * w * (len(sizes) + len(ratios) -1), 4)
    """
    in_height, in_width = images.shape[-2:]
    # 为了将锚点移动到像素的中心，需要设置偏移量，因为一个像素高为1宽为1，需各自偏移0.5
    offset_h, offset_w = 0.5, 0.5
    # 中心点的x和y
    center_h = (torch.arange(in_height) + offset_h) / in_height
    center_w = (torch.arange(in_width) + offset_w) / in_width
    shift_y, shift_x = torch.meshgrid(center_h, center_w)
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)
    # 每个中心点都将有“boxes_per_pixel”个锚框
    boxes_per_pixel = len(sizes) + len(ratios) - 1
    # 生成所有锚框的中心点，一个锚框以(xmin, ymin, xmax, ymax)表示
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                           dim=1).repeat_interleave(boxes_per_pixel, dim=0)

    size_tensor = torch.tensor(sizes)
    ratio_tensor = torch.tensor(ratios)
    # 生成锚框的高和宽，以高为准，宽需乘相应比例
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:]))) * in_height / in_width
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # 除以2来获得半高和半宽
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
        in_height * in_width, 1) / 2

    # 中心点 + 锚框偏移
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)


def show_bboxes(axes, bboxes, labels=None):
    """显示所有边界框"""
    colors = ['b', 'g', 'r', 'm', 'c']
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        # 添加边界框
        rect = d2l.bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        # 添加label
        if labels:
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color='k',
                      bbox=dict(facecolor=color, lw=0))


def box_iou(boxes1, boxes2):
    """
    计算两组边界框的交并比
    :param boxes1: shape (num_boxes1, 4)
    :param boxes2: shape (num_boxes2, 4)
    :return: shape (num_boxes1, num_boxes2) 的交并比
    """

    def box_area(boxes):
        """各个边界框的面积"""
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # 左上点的右下，右下点的左上
    # shape (num_boxes1, num_boxes2, 2)
    inter_upper_lefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lower_rights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    # 交的长和宽，将负值转化为0
    inters = (inter_lower_rights - inter_upper_lefts).clamp(min=0)
    # 交的面积
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    # 并的面积
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas


def assign_anchor_to_bbox(ground_truth, anchors, iou_threshold=0.5):
    """
    获得每个锚框所分配的真实边界框的idx，若无满足要求的边界框则idx为-1
    """
    # 位于第i行和第j列的元素 x_ij 是锚框i和真实边界框j的IoU
    jaccard = box_iou(anchors, ground_truth)
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # idx先预设为-1
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long)
    # 阈值保底
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    # 遍历各列进行分配和丢弃
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        # 获得当前最大值的行和列
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        # 分配
        anchors_bbox_map[anc_idx] = box_idx
        # 丢弃
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map


def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """
    为已分配边界框的锚框计算偏移
    :return: shape (num_anchors, 4)
    """
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset


def multi_box_target(anchors, labels):
    """
    使用真实边界框标记锚框
    :param anchors: shape (1, num_anchors, 4)
    :param labels: 边界框的类别和位置, shape (batch_size, num_bboxes, 5)
    :return: 偏移 shape (batch_size, num_anchors * 4)，
             掩码 shape (batch_size, num_anchors * 4)，
             类别 shape (batch_size, num_anchors)
    """
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    # 偏移，掩码，类别
    batch_offset, batch_mask, batch_class_labels = [], [], []
    num_anchors = anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        # 分配边界框
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors)
        # 将类标签和分配的边界框坐标初始化为零
        class_labels = torch.zeros(num_anchors, dtype=torch.long)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32)
        # 使用真实边界框来标记锚框的类别, 如果一个锚框没有被分配，我们标记其为背景（值为零）
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # 区分背景和非背景，shape (num_anchors, 4)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)
        # 计算偏移量，忽略背景
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return bbox_offset, bbox_mask, class_labels


# -------------------- NLP  -------------------


def read_time_machine():
    # 读取《The Time Machine》by H. G. Wells
    lines = open("../data/timemachine.txt").readlines()
    # 非字母都转换成空格、大写字母转小写
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines if lines]


def tokenize(lines, token_type='char'):
    # 把每行分裂成一个个字符或是一个个单词
    if token_type == 'word':
        return [line.split() for line in lines]
    elif token_type == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token_type)


class Vocab:
    """tokens的词汇表"""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        # 特殊的tokens，如<pad>等
        if reserved_tokens is None:
            reserved_tokens = []
        # 统计各个token的出现次数
        counter = collections.Counter([token for line in tokens for token in line])
        # 按出现次数排序
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The index for the unknown token is 0
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        # 词汇的出现次数需大于等于min_freq
        uniq_tokens += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in uniq_tokens]
        # 索引如何转token、token如何转索引
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """tokens转化成索引"""
        if not isinstance(tokens, (list, tuple)):
            # 可以直接转
            return self.token_to_idx.get(tokens, self.unk)
        # 递归转
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """索引转化成tokens"""
        if not isinstance(indices, (list, tuple)):
            # 索引不能越界，不然会报错
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


def load_corpus_time_machine():
    """获得timemachine语料库与词汇表"""
    # tokenize
    tokens = tokenize(read_time_machine())
    # 建立词汇表
    vocab = Vocab(tokens)
    # 转化为List[int]
    corpus = [vocab[token] for line in tokens for token in line if vocab[token] != 0]
    return corpus, vocab


class TimeMachineDataLoader:
    """生成timemachine数据集"""

    def __init__(self, batch_size, num_steps):
        # 读取上一步的结果
        self.corpus, self.vocab = load_corpus_time_machine()
        # batch_size: 每个batch的样本数
        # num_steps: 每个样本的token数，也是索引数
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        # 加点随机性，从offset开始读
        offset = random.randint(0, self.num_steps - 1)
        num_tokens = ((len(self.corpus) - offset - 1) // self.batch_size) * self.batch_size
        # shape: (batch_size, -1)
        Xs = torch.tensor(self.corpus[offset: offset + num_tokens]
                          ).reshape(self.batch_size, -1)
        Ys = torch.tensor(self.corpus[offset + 1: offset + 1 + num_tokens]
                          ).reshape(self.batch_size, -1)
        # 计算batch数
        num_batches = Xs.shape[1] // self.num_steps
        for i in range(0, self.num_steps * num_batches, self.num_steps):
            # 相应列的内容
            X = Xs[:, i: i + self.num_steps]
            Y = Ys[:, i: i + self.num_steps]
            yield X, Y


def load_data_time_machine(batch_size, num_steps):
    """读取timemachine数据集和词汇表"""
    data_iter = TimeMachineDataLoader(batch_size, num_steps)
    return data_iter, data_iter.vocab


class RNNModel(nn.Module):
    """RNN模型"""

    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        # 比如说nn.RNN()
        self.rnn = rnn_layer
        # 词汇量的大小
        self.vocab_size = vocab_size
        # 输入vocab_size -> 隐藏状态num_hiddens -> 输出vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 是否双向
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        # shape of inputs: (`batch_size`, `num_steps`)
        # shape of X: (`num_steps`, `batch_size`, `vocab_size`)
        # 将输入的int转为one_hot表示
        X = F.one_hot(inputs.T.long(), self.vocab_size).type(torch.float32)
        # shape of Y: (`num_steps`, `batch_size`, `num_directions` * `num_hiddens`)
        # shape of state: (`num_layers` * `num_directions`, `batch_size`, `num_hiddens`)
        # state是最终的隐藏状态
        Y, state = self.rnn(X, state)
        # shape of output: (`num_steps` * `batch_size`, `vocab_size`)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, batch_size=1, device=d2l.try_gpu()):
        if not isinstance(self.rnn, nn.LSTM):
            # `nn.RNN` and `nn.GRU` takes a tensor as hidden state
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens), device=device)
        else:
            # `nn.LSTM` takes a tuple of hidden states
            return (torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens), device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens), device=device))


def predict_language_model(prefix, num_preds, net, vocab):
    """在`prefix`之后生成新的字符"""
    device = d2l.try_gpu()
    # 初始化state
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    # 获取outputs[-1]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    # 得到有意义的state
    for y in prefix[1:]:
        # batch_size和num_steps均为1
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    # 预测`num_preds`步
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        # 得到概率最大的索引
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    # 索引转换成tokens
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def train_language_model(net, train_iter, vocab, lr, num_epochs):
    """训练语言模型"""
    device = try_gpu()
    net = net.to(device=device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr)
    # 记录误差和，tokens总数
    metric = Accumulator(2)
    # 画困惑度曲线
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        metric.reset()
        state = None
        for X, Y in train_iter:
            if state is None:
                # 初始化state
                state = net.begin_state(batch_size=X.shape[0])
            else:
                # state将不会被自动求导，这里分LSTM和非LSTM两种情况讨论
                if isinstance(state, tuple):
                    for s in state:
                        s.detach_()
                else:
                    state.detach_()
            # 前向传播
            y_hat, state = net(X, state)
            # (`batch_size`, `num_steps`)变为(`num_steps` * `batch_size`,)
            y = Y.T.reshape(-1)
            # 计算损失
            loss = loss_fn(y_hat, y.long()).mean()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 记录数据
            metric.add(loss * y.numel(), y.numel())
        # 画困惑度曲线
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, [math.exp(metric[0] / metric[1])])
    # predict
    print(predict_language_model("time traveller", 50, net, vocab))


def read_data_nmt():
    """读取英-法数据集"""
    # 读取整个文件
    text = open("../data/fra.txt").read()
    # 使用空格替换不间断空格、大写字母转小写
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 把标点符号和单词分开
    out = [' ' + char if i > 0 and (char in set(',.!?') and text[i - 1] != ' ') else char
           for i, char in enumerate(text)]
    return ''.join(out)


def tokenize_nmt(text, num_examples=None):
    """Tokenize英-法数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        # 保留前num_examples个句子对，可以作为测试集
        if num_examples and i > num_examples:
            break
        # 以\t分隔同一行中的英语和法语
        parts = line.split('\t')
        if len(parts) == 2:
            # tokenize为一个个单词，得到List[List[str]]
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


def truncate_pad(line, num_steps, padding_token):
    # 通过截断或填充使得每个句子的长度都是num_steps
    if len(line) >= num_steps:
        return line[: num_steps]
    return line + [padding_token] * (num_steps - len(line))


def build_array_nmt(lines, vocab, num_steps):
    """tokens转化为数字索引，且通过截断或填充使每个句子长度都一样"""
    # <eos>标明句子结束
    lines = [vocab[l] + [vocab['<eos>']] for l in lines]
    # 截断或填充
    array = torch.tensor([truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])
    # 标明哪些token是<pad>
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


def load_data_nmt(batch_size, num_steps, num_examples=600):
    """读取英-法数据集、英语词汇表、法语词汇表"""
    # 读取数据并tokenize
    text = read_data_nmt()
    source, target = tokenize_nmt(text, num_examples)
    # 分别建立英、法词汇表
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # 得到array和valid_len
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    # 使用load_array建立pytorch-dataset
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


class Encoder(nn.Module):
    """编码器的基类"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


class Decoder(nn.Module):
    """解码器的基类"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    """编码器-解码器结构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不想关的项"""
    # `X` shape: (`batch_size`, `num_steps`)
    # [None, :] makes (`num_steps`,) to (1, `num_steps`)
    # [:, None] makes (`batch_size`) to (`batch_size`, 1)
    mask = torch.arange((X.size(1)), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带屏蔽的softmax交叉熵损失函数"""
    # shape of pred: (`batch_size`, `num_steps`, `vocab_size`)
    # shape of label: (`batch_size`, `num_steps`)
    # shape of valid_len: (`batch_size`,)
    def forward(self, pred, label, valid_len):
        # 非pad为1，pad为0
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        # 'none': no reduction will be applied, 'mean': the weighted mean of the output is taken
        self.reduction = 'none'
        # nn.CrossEntropyLoss((`batch_size`, `vocab_size`, `num_steps`), (`batch_size`, `num_steps`))
        unweighted_loss = super().forward(pred.permute(0, 2, 1), label)
        # 得到带屏蔽的损失
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


def train_nmt(net, data_iter, lr, num_epochs, tgt_vocab):
    """训练机器翻译模型"""
    device = d2l.try_gpu()
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = MaskedSoftmaxCELoss()
    net.train()  # 用了Dropout，必须明示
    # 画带屏蔽的交叉熵损失
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        # 损失和，tokens总数
        metric = d2l.Accumulator(2)
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            # 解码器的输入是<bos>+真实输出序列
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)
            # 模型需是Encoder-Decoder结构
            Y_hat, _ = net(X, dec_input, X_valid_len)

            # Backpropagation
            optimizer.zero_grad()
            loss = loss_fn(Y_hat, Y, Y_valid_len)
            loss.sum().backward()
            optimizer.step()
            # 记录数据
            with torch.no_grad():
                metric.add(loss.sum(), Y_valid_len.sum())
        # 画图
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))


def predict_nmt(net, src_sentence, src_vocab, tgt_vocab, num_steps, device=d2l.try_gpu()):
    """机器翻译模型做预测"""
    net.eval()
    # 处理src_sentence
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    # 解码器初始state及初始输入
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    dec_X = torch.unsqueeze(
        torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    # 一步一步来
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # We use the token with the highest prediction likelihood as the input
        # of the decoder at the next time step
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # Once the end-of-sequence token is predicted, the generation of the
        # output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq))


def bleu(pred_seq, label_seq, k):
    """计算 BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    # 计算n元语法的精确度
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        # 统计标签序列中各n元语法的数量
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i:i + n])] += 1
        # 计算匹配
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_tokens[i:i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i:i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


def masked_softmax(X, valid_lens):
    """实现带遮蔽的softmax"""
    # shape of X: (`batch_size`, no. of queries, no. of key-value pairs)
    # shape of valid_lens: either (`batch_size`,) or (`batch_size`, no. of queries)
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        # 将valid_lens转化为(`batch_size` * no. of queries)
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 在最后的轴上，遮蔽的元素被替换成一个非常大的负值，其指数约为0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout):
        super(AdditiveAttention, self).__init__()
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        # shape of queries: (`batch_size`, no. of queries, `query_size`)
        # shape of keys: (`batch_size`, no. of key-value pairs, `key_size`)
        # shape of values: (`batch_size`, no. of key-value pairs, `value_size`)
        # shape of valid_lens: either (`batch_size`,) or (`batch_size`, no. of queries)
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion,
        # shape of queries: (`batch_size`, no. of queries, 1, `num_hiddens`)
        # shape of keys: (`batch_size`, 1, no. of key-value pairs, `num_hiddens`).
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # Shape of `scores`: (`batch_size`, no. of queries, no. of key-value pairs)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Output shape: (`batch_size`, no. of queries, `value_size`)
        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        # Shape of queries: (`batch_size`, no. of queries, `d`)
        # Shape of keys: (`batch_size`, no. of key-value pairs, `d`)
        # Shape of values: (`batch_size`, no. of key-value pairs, `value_size`)
        # Shape of valid_lens: (`batch_size`,) or (`batch_size`, no. of queries)
        d = queries.shape[-1]
        # Shape of `scores`: (`batch_size`, no. of queries, no. of key-value pairs)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Output shape: (`batch_size`, no. of queries, `value_size`)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        # `num_heads`个线性变换拼接起来，所以`num_hiddens`应可以整除`num_heads`
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # Shape of `queries`, `keys`, or `values`:
        # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)
        # Shape of `valid_lens`:
        # (`batch_size`,) or (`batch_size`, no. of queries)
        # After transposing, shape of output `queries`, `keys`, or `values`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs, `num_hiddens` / `num_heads`)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens,
                                                 repeats=self.num_heads, dim=0)

        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries, `num_hiddens` / `num_heads`)
        output = self.attention(queries, keys, values, valid_lens)

        # Shape of `output_concat`: (`batch_size`, no. of queries, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


def transpose_qkv(X, num_heads):
    """改变X的shape"""
    # Shape of input `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    # Shape of output `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
    # `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # Shape of output `X`:
    # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3)

    # Shape of `output`:
    # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转`transpose_qkv`的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的 `P`
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        # `num_hideens`必须为偶数，不然shape对不上
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_input)

    def forward(self, X):
        # X shape: (`batch_size`, `num_steps`, `ffn_num_input`)
        # 输入和输出的形状一样
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    """残差连接和层归一化"""
    def __init__(self, normalized_shape, dropout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # normalized_shape指定均值和方差计算的维度，需是后几个维度
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        # 先残差连接，再层归一化
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False):
        super(EncoderBlock, self).__init__()
        # key_size=query_size=value_size in Transformer
        # 多头注意力
        # num_hiddens应能整除num_heads，每个头的宽度为 num_hiddens//num_heads，W_o: num_hiddens -> num_hiddens
        self.attention = d2l.MultiHeadAttention(key_size, query_size,
                                                value_size, num_hiddens,
                                                num_heads, dropout, use_bias)
        # 第一个add&norm
        self.addnorm1 = d2l.AddNorm(norm_shape, dropout)
        # positionwiseFFN
        self.ffn = d2l.PositionWiseFFN(num_hiddens, ffn_num_hiddens)
        # 第二个add&norm
        self.addnorm2 = d2l.AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        # `X` shape: (`batch_size`, `num_steps`, `num_hiddens`)
        # `valid_lens` shape: None or (`batch_size`,) or (`batch_size`, `num_steps`)
        # 第一个子层
        # 在attention后被mask的的位置正常计算，但除了layerNorm外都是每个位置独立计算
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        # 第二个子层，形状不变
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(d2l.Encoder):
    """Transformer的编码器"""
    def __init__(self, vocab_size, num_hiddens,
                 norm_shape, ffn_num_hiddens, num_heads,
                 num_layers, dropout, use_bias=False):
        super(TransformerEncoder, self).__init__()
        self.num_hiddens = num_hiddens
        # Embedding将输入从`vocab_size`变为`num_hiddens`
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        # 位置编码
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        # 各个EncoderBlock
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block" + str(i),
                EncoderBlock(num_hiddens, num_hiddens, num_hiddens, num_hiddens,
                             norm_shape, ffn_num_hiddens, num_heads,
                             dropout, use_bias))

    def forward(self, X, valid_lens):
        # X shape: (`batch_size`, `num_steps`, `vocab_size`)
        # 因为位置编码值在-1到1之间，因此需要进行平方根缩放，保持它们在一个量级
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        # valid_lens在每个block都生效
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X


class DecoderBlock(nn.Module):
    """解码器中的第i个块"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_hiddens, num_heads,
                 dropout, i):
        super(DecoderBlock, self).__init__()
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(key_size, query_size,
                                                 value_size, num_hiddens,
                                                 num_heads, dropout)
        self.addnorm1 = d2l.AddNorm(norm_shape, dropout)
        self.attention2 = d2l.MultiHeadAttention(key_size, query_size,
                                                 value_size, num_hiddens,
                                                 num_heads, dropout)
        self.addnorm2 = d2l.AddNorm(norm_shape, dropout)
        self.ffn = d2l.PositionWiseFFN(num_hiddens, ffn_num_hiddens)
        self.addnorm3 = d2l.AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        # 训练阶段 `X` shape: (`batch_size`, `num_steps`, `num_hiddens`)
        # 预测阶段 `X` shape: (`batch_size`, 1, `num_hiddens`)
        # enc_outputs来自编码器（即其最后一个编码器block的输出）shape (`batch_size`, `num_steps`, `num_hiddens`)
        # enc_valid_lens也来编码器
        enc_outputs, enc_valid_lens = state[0], state[1]

        # `state[2][self.i]` 用于预测阶段，初始化为None，它存储截止目前时间步的的输出序列
        # 训练和第一个token的预测
        if state[2][self.i] is None:
            key_values = X
        # 后续预测
        else:
            # 跟RNN-seq2seq不一样，Transformer预测要用到截止目前的输出序列，而不只是上一时间步的输出
            # key_values shape: (`batch_size`, `cur_steps`, `num_hiddens`)
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values

        if self.training:
            batch_size, num_steps, _ = X.shape
            # 防作弊
            # shape of dec_valid_lens: (`batch_size`, `num_steps`)
            # 其中每一行是 [1, 2, ..., `num_steps`]
            dec_valid_lens = torch.arange(1, num_steps + 1,
                                          device=X.device).repeat(batch_size, 1)
        else:
            # 预测时token by token就不用了
            dec_valid_lens = None

        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # Encoder-decoder attention
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


class TransformerDecoder(d2l.Decoder):
    """Transformer解码器"""
    def __init__(self, vocab_size, num_hiddens,
                 norm_shape, ffn_num_hiddens, num_heads,
                 num_layers, dropout):
        super(TransformerDecoder, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        # 各个DecoderBlock
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block" + str(i),
                DecoderBlock(num_hiddens, num_hiddens, num_hiddens, num_hiddens,
                             norm_shape, ffn_num_hiddens, num_heads,
                             dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # 给state[2]留位置
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        # 常规操作
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        for i, blk in enumerate(self.blks):
            # state[0]和state[1]存储编码器的信息
            # state[2]用于预测，用来存储截止目前时间步各个block的输出序列
            X, state = blk(X, state)
        return self.dense(X), state


