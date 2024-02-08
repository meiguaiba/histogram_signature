import pickle
import numpy as np
from sklearn.model_selection import train_test_split
NUM_aug = 100
LEN = 300
random_div = 7777
augment_datafile = 'C:\\Users\\HP\\PycharmProjects\\SKlearn\\data\\breast_post_augmentation100_n331_0100_2600_bin50_box20.pkl'
with open (augment_datafile, 'rb') as pickle_file:
    (Xdata_aug, Ydata_aug)=pickle.load(pickle_file,encoding="latin-1")

original_datafile = 'C:\\Users\\HP\\PycharmProjects\\SKlearn\\data\\breast_post_original_n331_0100_2600_bin50_box20.pkl'
with open (original_datafile, 'rb') as pickle_file:
    (Xdata, Ydata, ID)=pickle.load(pickle_file,encoding="latin-1")


ODER = np.arange(0, 331)

TRAIN_VALID, TEST = train_test_split(ODER, test_size=0.4, random_state=random_div, stratify=Ydata)
TRAIN, VALID = train_test_split(TRAIN_VALID, test_size=0.16667, random_state=random_div, stratify=Ydata[TRAIN_VALID])
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Y_train = Ydata[TRAIN]
NUM_pos = Y_train.sum()
NUM_neg = Y_train.shape[0]-NUM_pos

NUM_aug_pos = NUM_aug
NUM_aug_neg = np.round(NUM_aug * NUM_pos / NUM_neg)
NUM_aug_neg = NUM_aug_neg.astype('int64')


NUM_total = NUM_aug_pos * NUM_pos + NUM_aug_neg * NUM_neg
NUM_total = NUM_total.astype('int64')

X_train = np.zeros([NUM_total, LEN])
Y_train = np.zeros(NUM_total)

LINE = 0
for ii in TRAIN:
    if Ydata[ii] == 1:
        for jj in range(NUM_aug_pos):
            X_train[LINE,:] = Xdata_aug[ii,:,jj]
            Y_train[LINE] = Ydata[ii]
            LINE += 1
    else:
        for jj in range(NUM_aug_neg):
            X_train[LINE,:] = Xdata_aug[ii,:,jj]
            Y_train[LINE] = Ydata[ii]
            LINE += 1
print('Train:')
print(NUM_total)

X_train = np.concatenate((X_train, Xdata[TRAIN,:]),axis=0)
Y_train = np.concatenate((Y_train, Ydata[TRAIN]),axis=0)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

X_valid = Xdata[VALID, :]
Y_valid = Ydata[VALID]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
X_test = Xdata[TEST,:]
Y_test = Ydata[TEST]

with open ('C:\\Users\\HP\\PycharmProjects\\SKlearn\\data\\histogram_signature_box20_augmentation.pkl', 'wb') as pickle_file:
    pickle.dump(((X_train, Y_train),(X_valid, Y_valid),(X_test, Y_test)), pickle_file)