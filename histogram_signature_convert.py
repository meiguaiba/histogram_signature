import numpy as np
import json, os
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pickle
import xlwt
import pandas as pd
workbook = xlwt.Workbook(encoding='utf-8')
worksheet = workbook.add_sheet('preNCRT_ID')

src_root = 'G:\\breastdata\\afterCRTphase'
src_root_vec = src_root.split(os.path.sep)
APEN_len = 0

BINS = 50
BOX = 20

Xdata = np.zeros([331, BINS*6])
Patient_num = 0

Ydata = np.zeros([331,])
ID_list = np.zeros([331,])

for patient_root, subdirs1, files in os.walk(src_root):
    patient_root_vec = patient_root.split(os.path.sep)
    if len(patient_root_vec) != len(src_root_vec)+1:
        continue
    ID = patient_root_vec[-1]


    if APEN_len==0:
        ID=ID
    else:
        ID = ID[:-APEN_len]
    nii_number = 0
    ID_list[Patient_num] = ID

    MaskImage = sitk.ReadImage(os.path.join(patient_root,'lesion_patch_1','mask.nii'))
    MaskArray = sitk.GetArrayFromImage(MaskImage)

    Mask0 = np.sum(np.sum(MaskArray, axis=2), axis=1)
    Mask1 = np.sum(np.sum(MaskArray, axis=2), axis=0)
    Mask2 = np.sum(np.sum(MaskArray, axis=0), axis=0)

    pos0 = np.squeeze(np.argwhere(Mask0 > 0))
    pos1 = np.squeeze(np.argwhere(Mask1 > 0))
    pos2 = np.squeeze(np.argwhere(Mask2 > 0))

    MaskArray0 = np.zeros(MaskArray.shape)
    MaskArray0[int(np.median(pos0)) - BOX:int(np.median(pos0)) + BOX+1, int(np.median(pos1)) - BOX:int(np.median(pos1)) + BOX+1, int(np.median(pos2)) - 1:int(np.median(pos2)) + 2] = 1
    with open(os.path.join(patient_root,'lesion_patch_1', 'labels.json'), 'r', encoding='utf-8') as f_json:
        label_config = json.load(f_json)
    ypCR = label_config['Label']['ypCR']
    if 'non_pcr' in ypCR:
        Ydata[Patient_num] = 0
    elif 'pcr' in ypCR :
        Ydata[Patient_num] = 1
    else:
        Ydata[Patient_num] = 2
    print(ID)
    print(Ydata[Patient_num])
    START = 0
    MAX = 0
    MIN = 10000
    MEA = 0
    STD = 0
    for sudir in subdirs1:
        Image = sitk.ReadImage(os.path.join(patient_root, sudir, 'main.nii'))
        Array = sitk.GetArrayFromImage(Image)
        Array = Array*MaskArray0
        arr = Array.flatten()
        arr = arr[arr>0]
        if MAX<arr.max():
            MAX=arr.max()
        if MIN > arr.min():
            MIN = arr.min()
        if MEA < np.mean(arr):
            MEA = np.mean(arr)
        if STD < np.std(arr):
            STD = np.std(arr)

        nii_number += 1
        n,bins = np.histogram(arr, bins=BINS, normed=0, density=0, range=(100, 2600))
        Xdata[Patient_num,START:START+BINS] = n

        START += BINS
        print(sudir)
    worksheet.write(Patient_num, 0, float(ID))
    worksheet.write(Patient_num, 1, Ydata[Patient_num])
    worksheet.write(Patient_num, 2, float(np.min(pos0)))
    worksheet.write(Patient_num, 3, float(np.max(pos0)))
    worksheet.write(Patient_num, 4, float(np.min(pos1)))
    worksheet.write(Patient_num, 5, float(np.max(pos1)))
    worksheet.write(Patient_num, 6, float(np.min(pos2)))
    worksheet.write(Patient_num, 7, float(np.max(pos2)))
    worksheet.write(Patient_num, 8, float(MAX))
    worksheet.write(Patient_num, 9, float(MIN))
    worksheet.write(Patient_num, 10, float(MEA))
    worksheet.write(Patient_num, 11, float(STD))
    Patient_num += 1
print(MAX)
print(MIN)

with open ('C:\\Users\\HP\\PycharmProjects\\SKlearn\\data\\breast_post_original_n331_0100_2600_bin50_box20.pkl', 'wb') as pickle_file:
    pickle.dump((Xdata, Ydata, ID_list), pickle_file)
workbook.save('C:\\Users\\HP\\PycharmProjects\\SKlearn\\data\\postNCRT_patch_ID_n331_0100_2600_bin50_box20.xls')