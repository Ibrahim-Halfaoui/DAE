import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import torchvision
import os
from model import ResNetAE #autoencoder
from dloader import DataloaderDocs
import cv2
dir_ = os.getcwd()

import cv2
import numpy as np
import torch
from skimage.filters.rank import mean_bilateral
from skimage import morphology
from PIL import Image
from PIL import ImageEnhance


def padCropImg(img):
    H = img.shape[0]
    W = img.shape[1]

    patchRes = 128
    pH = patchRes
    pW = patchRes
    ovlp = int(patchRes * 0.125)  # 32

    padH = (int((H - patchRes) / (patchRes - ovlp) + 1) * (patchRes - ovlp) + patchRes) - H
    padW = (int((W - patchRes) / (patchRes - ovlp) + 1) * (patchRes - ovlp) + patchRes) - W

    padImg = cv2.copyMakeBorder(img, 0, padH, 0, padW, cv2.BORDER_REPLICATE)

    ynum = int((padImg.shape[0] - pH) / (pH - ovlp)) + 1
    xnum = int((padImg.shape[1] - pW) / (pW - ovlp)) + 1

    totalPatch = np.zeros((ynum, xnum, patchRes, patchRes, 3), dtype=np.uint8)

    for j in range(0, ynum):
        for i in range(0, xnum):
            x = int(i * (pW - ovlp))
            y = int(j * (pH - ovlp))

            if j == (ynum - 1) and i == (xnum - 1):
                totalPatch[j, i] = img[-patchRes:, -patchRes:]
            elif j == (ynum - 1):
                totalPatch[j, i] = img[-patchRes:, x:int(x + patchRes)]
            elif i == (xnum - 1):
                totalPatch[j, i] = img[y:int(y + patchRes), -patchRes:]
            else:
                totalPatch[j, i] = padImg[y:int(y + patchRes), x:int(x + patchRes)]

    return totalPatch, padH, padW


def illCorrection(model, totalPatch):
    totalPatch = totalPatch.astype(np.float32) / 255.0

    ynum = totalPatch.shape[0]
    xnum = totalPatch.shape[1]

    totalResults = np.zeros((ynum, xnum, 128, 128, 3), dtype=np.float32)

    for j in range(0, ynum):
        for i in range(0, xnum):
            patchImg = torch.from_numpy(totalPatch[j, i]).permute(2, 0, 1)
            patchImg = patchImg.cuda().view(1, 3, 128, 128)

            output = model(patchImg)
            output = output.permute(0, 2, 3, 1).data.cpu().numpy()[0]

            output = output * 255.0
            output = output.astype(np.uint8)

            totalResults[j, i] = output

    return totalResults


def composePatch(totalResults, padH, padW, img):
    ynum = totalResults.shape[0]
    xnum = totalResults.shape[1]
    patchRes = totalResults.shape[2]

    ovlp = int(patchRes * 0.125)
    step = patchRes - ovlp

    resImg = np.zeros((patchRes + (ynum - 1) * step, patchRes + (xnum - 1) * step, 3), np.uint8)
    resImg = np.zeros_like(img).astype('uint8')

    for j in range(0, ynum):
        for i in range(0, xnum):
            sy = int(j * step)
            sx = int(i * step)

            if j == 0 and i != (xnum - 1):
                resImg[sy:(sy + patchRes), sx:(sx + patchRes)] = totalResults[j, i]
            elif i == 0 and j != (ynum - 1):
                resImg[sy + 10:(sy + patchRes), sx:(sx + patchRes)] = totalResults[j, i, 10:]
            elif j == (ynum - 1) and i == (xnum - 1):
                resImg[-patchRes + 10:, -patchRes + 10:] = totalResults[j, i, 10:, 10:]
            elif j == (ynum - 1) and i == 0:
                resImg[-patchRes + 10:, sx:(sx + patchRes)] = totalResults[j, i, 10:]
            elif j == (ynum - 1) and i != 0:
                resImg[-patchRes + 10:, sx + 10:(sx + patchRes)] = totalResults[j, i, 10:, 10:]
            elif i == (xnum - 1) and j == 0:
                resImg[sy:(sy + patchRes), -patchRes + 10:] = totalResults[j, i, :, 10:]
            elif i == (xnum - 1) and j != 0:
                resImg[sy + 10:(sy + patchRes), -patchRes + 10:] = totalResults[j, i, 10:, 10:]
            else:
                resImg[sy + 10:(sy + patchRes), sx + 10:(sx + patchRes)] = totalResults[j, i, 10:, 10:]

    resImg[0, :, :] = 255

    return resImg


def test_model_att(model, args, experiment_name, train_loader, val_loader, device):
    print('Start inference...\n')
    print(model)
    model.to(device)
    model.eval()



    for X_batch, Y_batch in val_loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        X_batch_vis = X_batch.squeeze(1)
        X_batch_vis = X_batch_vis.permute(0, 2, 3, 1)
        Y_batch_vis = Y_batch.squeeze(1)

        X_batch_np = X_batch_vis.detach().cpu().numpy()
        Y_batch_np = Y_batch_vis.detach().cpu().numpy()



        img = ((X_batch_np[0])).astype(np.uint8)
        totalPatch, padH, padW = padCropImg(img)
        totalResults = illCorrection(model, totalPatch)
        resImg = composePatch(totalResults, padH, padW, img)




        # Y_pred_np = Y_pred_vis.detach().cpu().numpy()

        for i0 in range(X_batch_np.shape[0]):
            # im = resImg
            im = np.concatenate((cv2.cvtColor(X_batch_np[i0], cv2.COLOR_BGR2RGB),
                                 cv2.cvtColor(Y_batch_np[i0].transpose(1, 2, 0), cv2.COLOR_BGR2RGB)  , resImg), axis=1)
            cv2.imshow('Augmented_GT_Prediction', im )
            cv2.imwrite('/home/ibrahim.halfaoui/data/uniqa_prep_poc/test/out.png', im)
            cv2.waitKey()



## Test or function
def test_model(model, args, experiment_name, train_loader, val_loader, device):
    print('Start inference...\n')
    print(model)
    model.to(device)
    model.eval()

    for X_batch, Y_batch in val_loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        y_pred = model(X_batch)


        # X_batch_vis = X_batch.squeeze(0)
        # Y_batch_vis = Y_batch.squeeze(0)
        # Y_pred_vis = y_pred.squeeze(0)

        X_batch_np = X_batch.squeeze(0).detach().cpu().numpy()
        Y_batch_np = Y_batch.squeeze(0).detach().cpu().numpy()
        Y_pred_np = y_pred.squeeze(0).detach().cpu().numpy()

        for i0 in range(X_batch_np.shape[0]):
            # th, imth = cv2.threshold(Y_pred_np[i0], 50, 255, cv2.THRESH_BINARY)
            im = np.concatenate((X_batch_np[i0], Y_batch_np[i0], Y_pred_np[i0] ), axis=1)
            cv2.imshow('Augmented_GT_Prediction', im)
            cv2.waitKey()



## Main
if __name__ == "__main__":
    # parse Arguments
    parser = argparse.ArgumentParser(description='Model for pre-processing scanned documents.')
    parser.add_argument('--mode', type=str, help='train or test', default='test')
    parser.add_argument('--datadir', type=str, help='path to the data folder',
                        default='C:/Projects/Uniqa_BUL/111/images')
    parser.add_argument('--batch_size', type=int, help='batch size',
                        default=1)
    parser.add_argument('--epochs', type=int, help='number of epochs', default=5000)
    parser.add_argument('--lr_rate', type=float, help='initial learning rate', default=1e-4)
    # parser.add_argument('--log_dir', default='C:/Pycharm_projects/ml_git/ml-projects/uniqa-health/pre_processing/log',
    #                     type=str, help='directory to save checkpoints and summaries')
    parser.add_argument('--log_dir', default='/home/ibrahim.halfaoui/projects/ml_git/ml-projects/uniqa-health/pre_processing/log',
                        type=str, help='directory to save checkpoints and summaries')
    parser.add_argument('--state', type=str, help='path to a specific checkpoint to load', default='')
    args = parser.parse_args()
    experiment_name = '/pre_processer_' + str(args.lr_rate) + '_' + str(args.batch_size) + '_' + str(args.epochs)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Used device: ', device)

    # Prepare Data
    train_data = DataloaderDocs(args.datadir + '/train/')
    test_data = DataloaderDocs(args.datadir + '/test/')
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=test_data, batch_size=args.batch_size)

    # Run training
    model = ResNetAE()
    if args.state != '':
        try:
            model.load_state_dict(torch.load(args.state))
        except AssertionError:
            model.load_state_dict(torch.load(args.state,
                 map_location=lambda storage, loc: storage))
    # if args.state != '':
    #     try:
    #         # model.load_state_dict(torch.load(args.state))
    #         model_dict = model.state_dict()
    #         pretrained_dict = torch.load(args.state, map_location='cuda:0')
    #         print(len(pretrained_dict.keys()))
    #         pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    #         print(len(pretrained_dict.keys()))
    #         model_dict.update(pretrained_dict)
    #         model.load_state_dict(model_dict)
    #
    #     except AssertionError:
    #         model.load_state_dict(torch.load(args.state,
    #                                          map_location=lambda storage, loc: storage))

    if args.mode == 'test':
        test_model(model, args, experiment_name, train_loader, val_loader, device)
        # test_model_att(model, args, experiment_name, train_loader, val_loader, device)
