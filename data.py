from operator import index
import numpy as np
import torch
import glob
import os.path
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import cv2
import torch.nn as nn

from seed import setup_seed
from data_func import *
setup_seed(42)

import numpy as np

# 3D sice coefficient
def cal_subject_level_dice(prediction, target, class_num=2):# class_num是你分割的目标的类别个数
    eps = 1e-10
    empty_value = -1.0
    dscs = empty_value * np.ones((class_num), dtype=np.float32)
    for i in range(0, class_num):
        if i not in target and i not in prediction:
            continue
        target_per_class = np.where(target == i, 1, 0).astype(np.float32)
        prediction_per_class = np.where(prediction == i, 1, 0).astype(np.float32)

        tp = np.sum(prediction_per_class * target_per_class)
        fp = np.sum(prediction_per_class) - tp
        fn = np.sum(target_per_class) - tp
        dsc = 2 * tp / (2 * tp + fp + fn + eps)
        dscs[i] = dsc
    # dscs = np.where(dscs == -1.0, np.nan, dscs)
    subject_level_dice = np.nanmean(dscs[1:])
    return subject_level_dice

class Data:
    def __init__(self, X_train, Y_train, X_val, Y_val, X_test, Y_test, handler):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler

        self.n_pool = len(X_train)
        self.n_test = len(X_test)

        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        # self.unlabeled_idxs = np.zeros(self.n_pool, dtype=bool)

    def supervised_training_labels(self):
        # used for supervised learning baseline, put all data labeled
        tmp_idxs = np.arange(self.n_pool)
        self.labeled_idxs[tmp_idxs[:]] = True

    def initialize_labels_random(self, num):
        # generate initial labeled pool
        # use idx to distinguish labeled and unlabeled data取1000张有target的sample
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        count = 0
        for i in tmp_idxs:
            if np.sum(self.Y_train[i])!=0:
                self.labeled_idxs[i] = True
                count+=1
                if count == num:
                    break

    def get_labeled_data(self):
        # get labeled data for training
        labeled_idxs = np.arange(self.n_pool, dtype=int)[self.labeled_idxs]
        # print("labeled data", labeled_idxs.shape)
        # print("labeled_idxs ", labeled_idxs)
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs],mode="train")

    # used for pseudo label filter remove blank patches
    def delete_black_patch(self, unlabeled_idxs, label):
        index = []
        for i in range(label.shape[0]):#24537
            if torch.sum(label[i])==0:
                index.append(unlabeled_idxs[i])
        return index

    def get_unlabeled_data(self, index=None):
        # get unlabeled data for active learning selection process
        unlabeled_idxs = np.arange(self.n_pool, dtype=int)[~self.labeled_idxs]#24537
        # print("unlabeled_idxs",unlabeled_idxs.shape)

        if index!=None:
            self.labeled_idxs[index] = True #5486
            unlabeled_idxs = np.arange(self.n_pool, dtype=int)[~self.labeled_idxs]#19051 19255
            self.labeled_idxs[index] = False
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs],mode="val")

    def get_val_data(self):
        # get validation dataset if exist
        return self.handler(self.X_val, self.Y_val,mode="val")

    def get_test_data(self):
        # get test dataset if exist
        return self.handler(self.X_test, self.Y_test,mode="val")

    def cal_test_acc(self, logits, targets):
        # calculate accuracy for test dataset
        dscs = []
        for prediction, target in zip(logits, targets):
            dsc = cal_subject_level_dice(prediction, target, class_num=2)
            dscs.append(dsc)
            dice = np.mean(dscs)
        return dice

    def cal_train_acc(self, preds):
        # calculate accuracy for train dataset for early stopping
        return 1.0 * (self.Y_train == preds).sum().item() / self.n_pool

    def add_labeled_data(self, data, label):
        # used for generated adversarial image expansion. Adding generated adversarial image with label to training dataset
        data = torch.reshape(data, (len(data),128,128))
        # data = torch.unsqueeze(data, 1)
        self.X_train = torch.tensor(self.X_train)#([25537, 128, 128])
        self.Y_train = torch.tensor(self.Y_train)
        self.X_train = torch.cat((self.X_train, data), 0)#([26037, 128, 128])
        self.Y_train = torch.cat((self.Y_train, label), 0)
        # print("labeled_idxs",self.labeled_idxs.shape)
        array = np.ones(len(data),dtype=bool)
        self.labeled_idxs = np.append(self.labeled_idxs, array)
        self.n_pool += len(data)
        return np.array(self.X_train)

    def get_label(self, idx):
        # Get the real label (share lable) for adversarial samples
        self.Y_train = np.array(self.Y_train)
        label = torch.tensor(self.Y_train[idx])
        return label

    def cal_target(self):
        target_num = []
        for i in range(len(self.Y_train)):
            target_num.append(np.sum(self.Y_train[i]))
        return target_num


def get_MSSEG(handler,supervised = False):
    #both 2d and 3d 
    # train_dir_name = "../MSSEG/Training/"
    # test_dir_name = "../MSSEG/Testing/"

    # # ps=get_path(train_dir_name)
    # # train_images_path = np.stack([name for name in [
    # # #     [os.path.join(train_dir_name + patient + '/Raw_Data/FLAIR.nii.gz') for patient in ps],
    # #     [os.path.join(train_dir_name + patient + '/Preprocessed_Data/FLAIR_preprocessed.nii.gz') for patient in ps],
    # # #     [os.path.join(train_dir_name + patient + '/Preprocessed_Data/DP_preprocessed.nii.gz') for patient in ps],
    # # #     [os.path.join(train_dir_name + patient + '/Preprocessed_Data/T2_preprocessed.nii.gz') for patient in ps],
    # # #     [os.path.join(train_dir_name + patient + '/Preprocessed_Data/T1_preprocessed.nii.gz') for patient in ps]
    # # ] if name is not None], axis=1)

    # # train_masks_path = np.stack([name for name in [
    # #     [os.path.join(train_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # # #     [os.path.join(train_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # # #     [os.path.join(train_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # # #     [os.path.join(train_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # # ] if name is not None], axis=1)

    # # train_brain_masks_path = np.stack([name for name in [
    # #     [os.path.join(train_dir_name + patient + '/Masks/Brain_Mask.nii.gz') for patient in ps],
    # # #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # # #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # # #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # # ] if name is not None], axis=1)

    # ps=get_path(test_dir_name)
    # test_images_path = np.stack([name for name in [
    # #     [os.path.join(test_dir_name + patient + '/Raw_Data/FLAIR.nii.gz') for patient in ps],
    #     [os.path.join(test_dir_name + patient + '/Preprocessed_Data/FLAIR_preprocessed.nii.gz') for patient in ps],
    # #     [os.path.join(test_dir_name + patient + '/Preprocessed_Data/T2_preprocessed.nii.gz') for patient in ps],
    # #     [os.path.join(test_dir_name + patient + '/Preprocessed_Data/T1_preprocessed.nii.gz') for patient in ps]
    # ] if name is not None], axis=1)

    # test_masks_path = np.stack([name for name in [
    #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # ] if name is not None], axis=1)

    # test_brain_masks_path = np.stack([name for name in [
    #     [os.path.join(test_dir_name + patient + '/Masks/Brain_Mask.nii.gz') for patient in ps],
    # #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # #     [os.path.join(test_dir_name + patient + '/Masks/Consensus.nii.gz') for patient in ps],
    # ] if name is not None], axis=1)
   
   
    # # train_images = get_image(train_images_path,label=False)
    # # train_masks = get_image(train_masks_path,label=True) 
    # # train_brain_masks = get_image(train_brain_masks_path,label=True)

    # test_images = get_image(test_images_path,label=False)  
    # test_masks = get_image(test_masks_path,label=True)
    # test_brain_area_masks = get_image(test_brain_masks_path,label=True)

    # # print(np.array(train_images).shape)
    # # print(np.array(train_masks).shape)
    # # print(np.array(train_brain_masks).shape)

    # print(np.array(test_images).shape)
    # print(np.array(test_masks).shape)
    # print(np.array(test_brain_area_masks).shape)

    # # train_brain_images,train_brain_masks, train_brain_area_masks = get_brain_area(train_images,train_brain_masks,train_masks)
    # # print(train_brain_images.shape)
    # # print(train_brain_masks.shape)

    # test_brain_images,test_brain_masks,test_brain_area_masks = get_brain_area(test_images,test_brain_area_masks,test_masks)
    # print(test_brain_images.shape)
    # print(test_brain_masks.shape)
    # print(test_brain_area_masks.shape)

    # # #2d
    # # #切分2d slice
    # # x_train_slice = get_2d_slice(train_brain_images,train_brain_masks,restrict=True)
    # # y_train_slice = get_2d_slice(train_brain_masks,train_brain_masks,restrict=True)
    # # print(x_train_slice.shape,y_train_slice.shape)

    # # # x_val_slice = get_2d_slice(val_x,val_y,restrict=False)
    # # # y_val_slice = get_2d_slice(val_y,val_y,restrict=False)
    # # # print(x_val.shape,y_val.shape)

    # x_test_slice = get_2d_slice(test_brain_images,test_brain_masks,restrict=False)
    # y_test_slice = get_2d_slice(test_brain_masks,test_brain_masks,restrict=False)
    # print(x_test_slice.shape,y_test_slice.shape)

    # # #为切分2d patch 防止有的无法整除
    # # full_train_imgs_list = paint_border_overlap(x_train_slice,stride=32)
    # # print(np.array(full_train_imgs_list).shape)
    # # full_train_masks_list = paint_border_overlap(y_train_slice,stride=32)
    # # print(np.array(full_train_masks_list).shape)

    # # # full_val_imgs_list = paint_border_overlap(x_val_slice,stride=64)
    # # # print(np.array(full_val_imgs_list).shape)
    # # # full_val_masks_list = paint_border_overlap(y_val_slice,stride=64)
    # # # print(np.array(full_val_masks_list).shape)

    # full_test_imgs_list = paint_border_overlap(x_test_slice,stride=96)
    # print(np.array(full_test_imgs_list).shape)
    # full_test_masks_list = paint_border_overlap(y_test_slice,stride=96)
    # print(np.array(full_test_masks_list).shape)

    # # #得到64*64 2d patch
    # # x_train,y_train = extract_ordered_overlap(np.array(full_train_imgs_list),label=full_train_masks_list,stride=32,train=True)
    # # print(np.array(x_train).shape,np.array(y_train).shape)

    # x_test,y_test = extract_ordered_overlap(np.array(full_test_imgs_list),label=full_test_masks_list,stride=96,train=False)
    # print(np.array(x_test).shape,np.array(y_test).shape)

    # train_x,val_x,train_y,val_y = train_test_split(x_train,y_train,test_size=0.2,random_state=42)
    # print(np.array(train_x).shape)
    # print(np.array(val_x).shape)

    # full_train_imgs_list = paint_border_overlap_3d(train_x,stride=16)
    # print(np.array(full_train_imgs_list).shape)
    # full_train_masks_list = paint_border_overlap_3d(train_y,stride=16)
    # print(np.array(full_train_masks_list).shape)

    # full_val_imgs_list = paint_border_overlap_3d(val_x,stride=48)
    # print(np.array(full_val_imgs_list).shape)
    # full_val_masks_list = paint_border_overlap_3d(val_y,stride=48)
    # print(np.array(full_val_masks_list).shape)

    # full_imgs_list = paint_border_overlap_3d(test_images,stride=48)
    # print(np.array(full_imgs_list).shape)
    # full_masks_list = paint_border_overlap_3d(test_masks,stride=48)
    # print(np.array(full_masks_list).shape)

    # # x_train = extract_ordered_overlap_3d(np.array(full_train_imgs_list),label=full_train_masks_list,stride=16,train=True)
    # # print(np.array(x_train).shape)
    # # y_train = extract_ordered_overlap_3d(np.array(full_train_masks_list),label=full_train_masks_list,stride=16,train=True)
    # # print(np.array(y_train).shape)

    # # x_val = extract_ordered_overlap_3d(np.array(full_val_imgs_list),label=full_val_masks_list,stride=48,train=False)
    # # print(np.array(x_val).shape)
    # # y_val = extract_ordered_overlap_3d(np.array(full_val_masks_list),label=full_val_masks_list,stride=48,train=False)
    # # print(np.array(y_val).shape)

    # # x_test = extract_ordered_overlap_3d(np.array(full_imgs_list),stride=48,train=False)
    # # print(np.array(x_test).shape)
    # # y_test = extract_ordered_overlap_3d(np.array(full_masks_list),stride=48,train=False)
    # # print(np.array(y_test).shape)

    # x_train = train_x
    # y_train = train_y
    # x_val = val_x
    # y_val = val_y


    # x_train = torch.load('../MSSEG/x_train_2d.pt')
    # y_train = torch.load('../MSSEG/y_train_2d.pt')
    # x_val = torch.load('../MSSEG/x_val_2d.pt')
    # y_val = torch.load('../MSSEG/y_val_2d.pt')
    # x_test = torch.load('../MSSEG/x_test_2d.pt')
    # y_test = torch.load('../MSSEG/x_test_2d.pt')

    x_test_slice = np.load("../MSSEG/x_test_slice.npy", allow_pickle=True)
    full_test_imgs_list = np.load("../MSSEG/full_test_imgs_list.npy", allow_pickle=True)
    test_brain_images = np.load("../MSSEG/test_brain_images.npy", allow_pickle=True)
    test_brain_masks = np.load("../MSSEG/test_brain_masks.npy", allow_pickle=True)

    x_test = np.load("../MSSEG/x_test_2d_patch.npy", allow_pickle=True)
    y_test = np.load("../MSSEG/y_test_2d_patch.npy", allow_pickle=True)

    if supervised == True:
        x_train = np.load("../MSSEG/x_train_2d_patch.npy", allow_pickle=True)
        y_train = np.load("../MSSEG/y_train_2d_patch.npy", allow_pickle=True)
        x_val = np.load("../MSSEG/x_val_2d_patch.npy", allow_pickle=True)
        y_val = np.load("../MSSEG/y_val_2d_patch.npy", allow_pickle=True)

    else:
        x_train = np.load("../MSSEG/x_train_2d_patch_full.npy", allow_pickle=True)
        y_train = np.load("../MSSEG/y_train_2d_patch_full.npy", allow_pickle=True)
        x_val = np.load("../MSSEG/x_val_2d_patch_full.npy", allow_pickle=True)
        y_val = np.load("../MSSEG/y_val_2d_patch_full.npy", allow_pickle=True)

    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    print(x_test.shape, y_test.shape)

    return x_train, y_train, x_val, y_val, x_test, y_test, handler, full_test_imgs_list, x_test_slice, test_brain_images, test_brain_masks
