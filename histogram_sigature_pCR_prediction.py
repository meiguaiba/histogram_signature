from pathlib import Path
import pickle
from matplotlib import pyplot
import numpy as np
import torch
from torch import optim
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, roc_auc_score
import xlwt

print(torch.cuda.is_available())

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

DATA_PATH = Path("data")
MODEL_PATH = Path("models")
HISTORY_PATH = Path("history")
model_PATH = MODEL_PATH / "001"


FILENAME = 'histogram_signature_box20.pkl'
with open((DATA_PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = pickle.load(f, encoding="latin-1")

pyplot.imshow(x_train[0].reshape((6, 50)), cmap="gray")
pyplot.show()
print(x_train.shape)
x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('int64')
y_valid = y_valid.astype('int64')
y_test = y_test.astype('int64')
x_train0, y_train0, x_valid, y_valid, x_test, y_test = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid, x_test, y_test)
)

a = [i for i in range(x_train.shape[0])]


x_train = x_train0[a]
y_train = y_train0[a]

bs = x_train.shape[0]   # batch size

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

lr = 0.0001  # learning rate
epoch = 5258  # optimal epoch

loss_func = F.cross_entropy
train_ds = TensorDataset(x_train, y_train)

valid_ds = TensorDataset(x_valid, y_valid)

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
        ground_truth = yb.cpu().numpy()
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
        ground_truth = yb.cpu().numpy()
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
        ground_truth = yb.cpu().numpy()
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

    workbook.save(os.path.join(history_PATH, 'prediction_bd_ep9436_r77775555_noshuffle3.xls'))

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


def preprocess(x, y):
    return x.view(-1, 1, 6, 50).to(dev), y.to(dev)

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
CHANNEL = 64
model = nn.Sequential(
    nn.Conv2d(1, CHANNEL, kernel_size=3, stride=[1, 1], padding=[0, 0]),  # [4, 48]
    nn.ReLU(),
    nn.Conv2d(CHANNEL, CHANNEL, kernel_size=3, stride=[1, 1], padding=[0, 0]),  # [2, 46]
    nn.ReLU(),
    nn.Conv2d(CHANNEL, CHANNEL, kernel_size=2, stride=[1, 1], padding=[0, 0]),  # [1, 45]
    nn.ReLU(),
    nn.Conv2d(CHANNEL, CHANNEL, kernel_size=[1, 3], stride=[1, 3], padding=0),  # [1, 15]
    nn.ReLU(),
    nn.Conv2d(CHANNEL, CHANNEL, kernel_size=[1, 3], stride=[1, 3], padding=0),  # [1, 5]
    nn.ReLU(),
    nn.Conv2d(CHANNEL, CHANNEL, kernel_size=[1, 3], stride=1, padding=0),
    nn.ReLU(),
    nn.Conv2d(CHANNEL, CHANNEL, kernel_size=[1, 3], stride=1, padding=0),
    nn.ReLU(),
    Lambda(lambda x: x.view(x.size(0), -1)),
    nn.Linear(CHANNEL, 16),
    nn.Linear(16, 2),
    nn.Softmax(dim=1),

)
model.to(dev)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0)

evaluate(epoch, model, loss_func, opt, train_dl, valid_dl, test_dl, train_dl_all, valid_dl_all, test_dl_all, HISTORY_PATH)