from pathlib import Path
import requests
import pickle
import gzip
from matplotlib import pyplot
import numpy as np
import torch
import math
from torch import optim
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, roc_auc_score
import xlwt

print(torch.cuda.is_available())

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
# dev = torch.device("cpu")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

DATA_PATH = Path("data")
MODEL_PATH = Path("models")
HISTORY_PATH = Path("history")
PATH = DATA_PATH / "DCE"
# model_PATH = "D:\\data\\pytorchmodel\\DCE\\DCE181_bs1000_lr0001_shuffle0_random8888_data0422"
# model_PATH = "C:\\Users\\HP\\PycharmProjects\pytorch\models\DCE52r";
# history_PATH = HISTORY_PATH / "DCE"
history_PATH = '/home/ps/PycharmProjects/history'
# model_PATH = MODEL_PATH / "DCE60am"
# model_PATH = "I:\\models\\DCE60aq"
# model_PATH = MODEL_PATH / "DCE60ba"
model_PATH = '/home/ps/PycharmProjects/models/DCE110'
PATH.mkdir(parents=True, exist_ok=True)

FILENAME = 'post_train_original_n165_0001_3601_bin36.pkl'
with open((PATH / FILENAME).as_posix(), "rb") as f:
    # (x_train_original, y_train_original, ID_train_original) = pickle.load(f, encoding="latin-1")
    (x_train, y_train, ID_train) = pickle.load(f, encoding="latin-1")
FILENAME = 'post_train_augment_n165_0001_3601_bin36.pkl'
with open((PATH / FILENAME).as_posix(), "rb") as f:
    (x_train_augment, y_train_augment, ID_train_augment) = pickle.load(f, encoding="latin-1")
FILENAME = 'post_validation_original_n34_0001_3601_bin36.pkl'
with open((PATH / FILENAME).as_posix(), "rb") as f:
    (x_valid, y_valid, ID_valid) = pickle.load(f, encoding="latin-1")
FILENAME = 'post_test_original_n132_0001_3601_bin36.pkl'
with open((PATH / FILENAME).as_posix(), "rb") as f:
    (x_test, y_test, ID_test) = pickle.load(f, encoding="latin-1")

# y_train = np.concatenate([y_train_original, y_train_augment], axis=0)
# x_train = np.concatenate([x_train_original, x_train_augment], axis=0)
# ID_train = np.concatenate([ID_train_original, ID_train_augment], axis=0)
# np.random.seed(123)

pyplot.imshow(x_train[0].reshape((6, 36)), cmap="gray")
pyplot.show()
print(x_train.shape)
x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('int64')
y_valid = y_valid.astype('int64')
y_test = y_test.astype('int64')
# PyTorch uses torch.tensor, rather than numpy arrays, so we need to convert our data.
x_train0, y_train0, x_valid, y_valid, x_test, y_test = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid, x_test, y_test)
)
# n, c = x_train.shape
# x_train, x_train.shape, y_train.min(), y_train.max()
# print(x_train, y_train)
# print(x_train.shape)
# print(y_train.min(), y_train.max())
a = [i for i in range(x_train.shape[0])]

# np.random.seed(7777)
# np.random.shuffle(a)

x_train = x_train0[a]
y_train = y_train0[a]
# x_train = x_train0
# y_train = y_train0

bs = x_train.shape[0]   # batch size

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

lr = 0.0001  # learning rate
# epoch = 83408 # optimal epoch
# epoch = 5258  # optimal epoch
epoch = 8010  # optimal epoch

loss_func = F.cross_entropy
# loss_func = F.binary_cross_entropy
train_ds = TensorDataset(x_train, y_train)
# train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
# valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

test_ds = TensorDataset(x_test, y_test)

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def evaluate(epoch, model, loss_func, opt, train_dl, valid_dl, test_dl, train_dl_all, valid_dl_all, test_dl_all, history_PATH):
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('history')

    save_PATH = os.path.join(model_PATH, 'parameters_' + str(epoch) + '.pt')
    checkpoint = torch.load(save_PATH)

    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()

    with torch.no_grad():
        losses_train, nums_train = zip(
            *[loss_batch(model, loss_func, xb, yb) for xb, yb in train_dl]
        )
        losses_valid, nums_valid = zip(
            *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
        )
        losses_test, nums_test = zip(
            *[loss_batch(model, loss_func, xb, yb) for xb, yb in test_dl]
        )

    train_loss = np.sum(np.multiply(losses_train, nums_train)) / np.sum(nums_train)
    accuracies_train = [accuracy(model(xb), yb).data.cpu().numpy() for xb, yb in train_dl]
    train_accuracy = np.sum(np.multiply(accuracies_train, nums_train)) / np.sum(nums_train)

    valid_loss = np.sum(np.multiply(losses_valid, nums_valid)) / np.sum(nums_valid)
    accuracies_valid = [accuracy(model(xb), yb).data.cpu().numpy() for xb, yb in valid_dl]
    valid_accuracy = np.sum(np.multiply(accuracies_valid, nums_valid)) / np.sum(nums_valid)

    test_loss = np.sum(np.multiply(losses_test, nums_test)) / np.sum(nums_test)
    accuracies_test = [accuracy(model(xb), yb).data.cpu().numpy() for xb, yb in test_dl]
    test_accuracy = np.sum(np.multiply(accuracies_test, nums_test)) / np.sum(nums_test)

    START = 0
    for xb, yb in train_dl_all:
        # print([torch.argmax(model(xb), dim=1), yb])
        # print(accuracy(model(xb), yb))
        # predict_score = torch.argmax(model(xb), dim=1).cpu().numpy()
        ground_truth = yb.cpu().numpy()
        # fpr, tpr, _ = roc_curve(ground_truth, predict_score)
        PROB0 = model(xb)[:,0].cpu().detach().numpy()
        PROB1 = model(xb)[:, 1].cpu().detach().numpy()
        fpr, tpr, _ = roc_curve(ground_truth, PROB1/(PROB0+PROB1+0.001))
        AUC_train = auc(fpr, tpr)
        predict_score = model(xb).cpu().detach().numpy()
        ground_truth_one_hot = F.one_hot(yb).cpu().numpy()
        AUC_train00 = roc_auc_score(ground_truth_one_hot, predict_score, average=None)
        for ii in range(len(ground_truth)):
            worksheet.write(START, 0, int(ground_truth[ii]))
            worksheet.write(START, 1, float(PROB0[ii]))
            worksheet.write(START, 2, float(PROB1[ii]))
            worksheet.write(START, 3, float(PROB1[ii]/(PROB0[ii]+PROB1[ii]+0.001)))
            worksheet.write(START, 4, float(predict_score[ii, 0]))
            worksheet.write(START, 5, float(predict_score[ii, 1]))
            START = START+1

    print(START)
    for xb, yb in valid_dl_all:
        # print([torch.argmax(model(xb), dim=1), yb])
        # print(accuracy(model(xb), yb))
        # predict_score = torch.argmax(model(xb), dim=1).cpu().numpy()
        ground_truth = yb.cpu().numpy()
        # fpr, tpr, _ = roc_curve(ground_truth, predict_score)
        PROB0 = model(xb)[:, 0].cpu().detach().numpy()
        PROB1 = model(xb)[:, 1].cpu().detach().numpy()
        fpr, tpr, _ = roc_curve(ground_truth, PROB1 / (PROB0 + PROB1+0.001))
        AUC_valid = auc(fpr, tpr)
        predict_score = model(xb).cpu().detach().numpy()
        ground_truth_one_hot = F.one_hot(yb).cpu().numpy()
        AUC_valid00 = roc_auc_score(ground_truth_one_hot, predict_score, average=None)
        for ii in range(len(ground_truth)):
            worksheet.write(START, 0, int(ground_truth[ii]))
            worksheet.write(START, 1, float(PROB0[ii]))
            worksheet.write(START, 2, float(PROB1[ii]))
            worksheet.write(START, 3, float(PROB1[ii] / (PROB0[ii] + PROB1[ii] + 0.001)))
            worksheet.write(START, 4, float(predict_score[ii, 0]))
            worksheet.write(START, 5, float(predict_score[ii, 1]))
            START = START+1

    print(START)
    for xb, yb in test_dl_all:
        # print([torch.argmax(model(xb), dim=1), yb])
        # print(accuracy(model(xb), yb))
        # predict_score = torch.argmax(model(xb), dim=1).cpu().numpy()
        ground_truth = yb.cpu().numpy()
        # fpr, tpr, _ = roc_curve(ground_truth, predict_score)
        PROB0 = model(xb)[:, 0].cpu().detach().numpy()
        PROB1 = model(xb)[:, 1].cpu().detach().numpy()
        fpr, tpr, _ = roc_curve(ground_truth, PROB1 / (PROB0 + PROB1+0.001))
        AUC_test = auc(fpr, tpr)
        predict_score = model(xb).cpu().detach().numpy()
        ground_truth_one_hot = F.one_hot(yb).cpu().numpy()
        AUC_test00 = roc_auc_score(ground_truth_one_hot, predict_score, average=None)
        for ii in range(len(ground_truth)):
            worksheet.write(START, 0, int(ground_truth[ii]))
            worksheet.write(START, 1, float(PROB0[ii]))
            worksheet.write(START, 2, float(PROB1[ii]))
            worksheet.write(START, 3, float(PROB1[ii] / (PROB0[ii] + PROB1[ii] + 0.001)))
            worksheet.write(START, 4, float(predict_score[ii, 0]))
            worksheet.write(START, 5, float(predict_score[ii, 1]))
            START = START+1

    print(START)

    print(epoch, train_loss, train_accuracy, AUC_train, valid_loss, valid_accuracy, AUC_valid, test_loss, test_accuracy, AUC_test, AUC_train00[0],  AUC_train00[1], AUC_valid00[0],  AUC_valid00[1], AUC_test00[0],  AUC_test00[1])

    # workbook.save(os.path.join(history_PATH, 'prediction_breast3_52z_noshuffle.xls'))
    # workbook.save(os.path.join(history_PATH, 'prediction_aq_ep5258_r77775555_noshuffle3.xls'))
    workbook.save(os.path.join(history_PATH, 'prediction_model110_ep8010_r6666.xls'))

def get_data(train_ds, valid_ds, test_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=False),
        DataLoader(valid_ds, batch_size=bs),
        DataLoader(test_ds, batch_size=bs),
    )

def get_data_test(train_ds, valid_ds, test_ds):
    return (
        DataLoader(train_ds, batch_size=x_train.shape[0], shuffle=False),
        DataLoader(valid_ds, batch_size=x_valid.shape[0]),
        DataLoader(test_ds, batch_size=x_test.shape[0]),
    )

# nn.Sequential

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


# Our CNN is fairly concise, but it only works with MNIST, because:
# It assumes the input is a 28*28 long vector
# It assumes that the final CNN grid size is 4*4 (since that’s the average
# pooling kernel size we used)

# Let’s get rid of these two assumptions, so our model works with any 2d single channel image.
# First, we can remove the initial Lambda layer by moving the data preprocessing into a generator:

def preprocess(x, y):
    return x.view(-1, 1, 6, 36).to(dev), y.to(dev)

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

train_dl, valid_dl, test_dl = get_data(train_ds, valid_ds, test_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)
test_dl = WrappedDataLoader(test_dl, preprocess)

train_dl_all, valid_dl_all, test_dl_all = get_data_test(train_ds, valid_ds, test_ds)
train_dl_all = WrappedDataLoader(train_dl_all, preprocess)
valid_dl_all = WrappedDataLoader(valid_dl_all, preprocess)
test_dl_all = WrappedDataLoader(test_dl_all, preprocess)
CHANNEL = 16
model = nn.Sequential(
    nn.Conv2d(1, CHANNEL, kernel_size=3, stride=[1, 3], padding=[0, 0]),#[4, 12]
    nn.BatchNorm2d(CHANNEL),
    nn.ReLU(),
    nn.Conv2d(CHANNEL, CHANNEL, kernel_size=3, stride=[1, 3], padding=[0, 0]),#[2, 4]
    nn.BatchNorm2d(CHANNEL),
    nn.ReLU(),
    nn.MaxPool2d(2),#[3, 3]
    # Lambda(lambda x: x.view( -1, 2)),
    # nn.Softmax2d(),
    Lambda(lambda x: x.view(x.size(0), -1)),
    nn.Linear(CHANNEL*2, CHANNEL),
    nn.ReLU(),
    # nn.Dropout(0.5),
    nn.Linear(CHANNEL, 2),
    nn.Softmax(dim=1),

)
model.to(dev)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0)

evaluate(epoch, model, loss_func, opt, train_dl, valid_dl, test_dl, train_dl_all, valid_dl_all, test_dl_all, history_PATH)