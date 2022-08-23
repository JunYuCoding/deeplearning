import torch
import torch.nn as nn

import numpy as np
from datetime import datetime
import datetime

import torch.optim as optim
import matplotlib.pyplot as plt

from fcn.voc_seg_data import VOC_SEG
from fcn8s_model import *
# <---------------------------------------------->
# 下面开始训练网络

# 在训练网络前定义函数用于计算Acc 和 mIou
# 计算混淆矩阵
def _fast_hist(label_true, label_pred, n_class):
    # 寻找真实标签中为目标的像素索引， 【0，1，2，4，5】 若0-4，则0，1，2为mask
    mask = (label_true >= 0) & (label_true < n_class)
    # np.bincount 返回列表中，0-x数字出现的次数，按位置，即hist长度一定为x+1,例x=9
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

# 根据混淆矩阵计算Acc和mIou
def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    # np.diag 对角线元素 预测正确的
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1) # 行求和
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
                hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    return acc, acc_cls, mean_iu


def main():
    # 1. load dataset
    root = "../../dataset/VOC2012"
    batch_size = 32
    height = 224
    width = 224
    voc_train = VOC_SEG(root, width, height, train=True)
    voc_test = VOC_SEG(root, width, height, train=False)
    train_dataloader = DataLoader(voc_train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(voc_test, batch_size=batch_size, shuffle=True)

    # 2. load model
    num_class = 21
    vgg_model = VGGNet(requires_grad=True, show_params=False)
    model = FCN8s(pretrained_net=vgg_model,num_classes=num_class)
    # model = FCN8s(,num_classes=num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 3. prepare super parameters
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.7)
    epoch = 50

    # 4. train
    val_acc_list = []
    out_dir = "./checkpoints/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    for epoch in range(0, epoch):
        print('\nEpoch: %d' % (epoch + 1))
        model.train()
        sum_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            length = len(train_dataloader)
            print(images.shape,"===========train_data_len",length)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)  # torch.size([batch_size, num_class, width, height])
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            predicted = torch.argmax(outputs.data, 1)

            label_pred = predicted.data.cpu().numpy()
            label_true = labels.data.cpu().numpy()
            acc, acc_cls, mean_iu = label_accuracy_score(label_true, label_pred, num_class)

            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% | Acc_cls: %.03f%% |Mean_iu: %.3f'
                  % (epoch + 1, (batch_idx + 1 + epoch * length), sum_loss / (batch_idx + 1),
                     100. * acc, 100. * acc_cls, mean_iu))
            # val_info = ('global correct: {:.1f}\n'\
            #             'average row correct: {}\n'\
            #             'IoU: {}\n'\
            #             'mean IoU: {:.1f}').format(\
            #             acc * 100,
            #             ['{:.1f}'.format(i) for i in (acc_cls * 100).tolist()],
            #             ['{:.1f}'.format(i) for i in (mean_iu * 100).tolist()],
            #             mean_iu.mean().item() * 100)
            # with open(results_file, "a") as f:
            #     # 记录每个epoch对应的train_loss、lr以及验证集各指标
            #     train_info = f"[epoch: {epoch+1}]\n" \
            #                  f"train_loss: {sum_loss / (batch_idx + 1):.4f}\n"
            #     f.write(train_info + val_info + "\n\n")



        # get the ac with testdataset in each epoch
        print('Waiting Val...')
        mean_iu_epoch = 0.0
        mean_acc = 0.0
        mean_acc_cls = 0.0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_dataloader):
                model.eval()
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predicted = torch.argmax(outputs.data, 1) # 预测值最大的那个，沿着深度方向，每个像素预测值最大的索引，argmax(1)

                label_pred = predicted.data.cpu().numpy()
                label_true = labels.data.cpu().numpy()
                acc, acc_cls, mean_iu = label_accuracy_score(label_true, label_pred, num_class)

                # total += labels.size(0)
                # iou = torch.sum((predicted == labels.data), (1,2)) / float(width*height)
                # iou = torch.sum(iou)
                # correct += iou
                mean_iu_epoch += mean_iu
                mean_acc += acc
                mean_acc_cls += acc_cls

            print('Acc_epoch: %.3f%% | Acc_cls_epoch: %.03f%% |Mean_iu_epoch: %.3f'
                  % ((100. * mean_acc / len(val_dataloader)), (100. * mean_acc_cls / len(val_dataloader)),
                     mean_iu_epoch / len(val_dataloader)))

            val_acc_list.append(mean_iu_epoch / len(val_dataloader))

        torch.save(model.state_dict(), out_dir + "last.pt")
        if mean_iu_epoch / len(val_dataloader) == max(val_acc_list):
            torch.save(model.state_dict(), out_dir + "best.pt")
            print("save epoch {} model".format(epoch))



if __name__ == "__main__":
    main()
    # train(epo_num=100, show_vgg_params=True)
