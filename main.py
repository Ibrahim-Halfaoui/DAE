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

def unfold_batch(X_batch, patch_s):
    # c = X_batch.shape[1]
    x = torch.nn.functional.unfold(X_batch, kernel_size=patch_s, stride=int(patch_s // 4))
    x = x.view(-1, X_batch.shape[1], patch_s, patch_s)
    return x

def fold_batch(X_batch, patch_s, H, W):
    x = X_batch.view(-1, X_batch.shape[1] * patch_s * patch_s, X_batch.shape[0])
    ou_size = (H, W)
    x = torch.nn.functional.fold(x, ou_size , kernel_size=patch_s, stride=int(patch_s // 4))
    # norm_map = F.fold(F.unfold(torch.ones(I.shape).type(dtype), kernel_size, stride=stride), I.shape[-2:], kernel_size,
    #                   stride=stride)
    # I_f /= norm_map
    return x

## Training function
def train_model(model, args, experiment_name, train_loader, val_loader, device):
    print('Start training...\n')
    print(model)
    model.to(device)
    criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr_rate)
    if not os.path.isdir(args.log_dir + experiment_name):
        os.mkdir(args.log_dir + experiment_name)
    writer = SummaryWriter(args.log_dir + experiment_name)
    model.train()
    Patch_s = 128

    # Train loop
    for e in range(1, args.epochs + 1):
        epoch_loss = 0
        k = 0
        best_loss = 0
        # optimizer.zero_grad()
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            #X_batch_mod = unfold_batch(X_batch, Patch_s)
            X_batch_mod = X_batch

            # X_batch_vis = X_batch #.squeeze(0)
            # Y_batch_vis = Y_batch #.squeeze(0)
            # X_batch_np = X_batch_vis.detach().cpu().numpy()
            # Y_batch_np = Y_batch_vis.detach().cpu().numpy()
            # for i0 in range(X_batch_np.shape[0]):
            #     # im = X_batch_np[i0].transpose(1,2,0)
            #     im = np.concatenate((X_batch_np[i0].transpose(1,2,0), Y_batch_np[i0].transpose(1,2,0)), axis=1)
            #     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            #     cv2.imshow('Numpy Horizontal Concat', im  * 255)
            #     cv2.waitKey()
            optimizer.zero_grad()
            y_pred = model(X_batch_mod)
            #y_pred = fold_batch(y_pred, Patch_s, Y_batch.shape[2], Y_batch.shape[3])
            loss = criterion(y_pred, Y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            grid_img = torchvision.utils.make_grid(y_pred , nrow=args.batch_size)
            grid_img_gt = torchvision.utils.make_grid(Y_batch , nrow=args.batch_size)
            k+= 1
            if k == 1:
                best_loss = epoch_loss
            writer.add_image('output_train', grid_img, k + e * len(train_loader))
            writer.add_image('GT_train', grid_img_gt, k + e * len(train_loader))
            # data = Y_batch[
            #     0].detach().cpu().numpy()  # cv2.cvtColor(Y_batch[0].detach().cpu().numpy(), cv2.COLOR_BGR2RGB)
            # writer.add_image('data', data, k + e * len(train_loader))
            # data2 = y_pred[
            #     0].detach().cpu().numpy()  # cv2.cvtColor(Y_batch[0].detach().cpu().numpy(), cv2.COLOR_BGR2RGB)
            # writer.add_image('data2', data2, k + e * len(train_loader))

        print(
            f'Epoch {e + 0:03}: | TrainLoss: {epoch_loss / len(train_loader):.5f}')
        writer.add_scalar('TrainLoss', epoch_loss / len(train_loader), e)


        if e % 10 == 0:
            model.eval()
            # Validation loop
            epoch_loss_t = 0
            k1 = 0
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                #X_batch_mod = unfold_batch(X_batch, Patch_s)

                y_pred = model(X_batch)
                #y_pred = fold_batch(y_pred, Patch_s, Y_batch.shape[2], Y_batch.shape[3])
                loss = criterion(y_pred, Y_batch)


                epoch_loss_t += loss.item()
                grid_img = torchvision.utils.make_grid(y_pred * 255, nrow=args.batch_size)
                grid_img_gt = torchvision.utils.make_grid(Y_batch * 255, nrow=args.batch_size)
                k1 += 1
                writer.add_image('output_val', grid_img, k1 + e * len(val_loader))
                writer.add_image('GT_val', grid_img_gt, k + e * len(val_loader))

            print(
                f'Epoch {e + 0:03}: | ValLoss: {epoch_loss_t / len(val_loader):.5f}')
            writer.add_scalar('ValLoss', epoch_loss_t / len(val_loader), e)


        # Saving snapshots
        if epoch_loss <= best_loss:
            # torch.save(model.state_dict(), args.log_dir + experiment_name + '/epoch_{}.pt'.format(e))
            torch.save(model.state_dict(), args.log_dir + experiment_name + '/best_model.pt')


## Main
if __name__ == "__main__":
    # parse Arguments
    parser = argparse.ArgumentParser(description='Model for pre-processing scanned documents.')
    parser.add_argument('--mode', type=str, help='train or test', default='train')
    parser.add_argument('--datadir', type=str, help='path to the data folder',
                        default='C:/Projects/Uniqa_BUL/111/images')
    parser.add_argument('--batch_size', type=int, help='batch size',
                        default=8)
    parser.add_argument('--epochs', type=int, help='number of epochs', default=1000000)
    parser.add_argument('--lr_rate', type=float, help='initial learning rate', default=0.003)
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
    #     #try:
    #         # model.load_state_dict(torch.load(args.state))
    #         model_dict = model.state_dict()
    #         pretrained_dict = torch.load(args.state, map_location='cuda:0')
    #         print(len(pretrained_dict.keys()))
    #         pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    #         print(len(pretrained_dict.keys()))
    #         model_dict.update(pretrained_dict)
    #         model.load_state_dict(model_dict)
    #
    #   #  except AssertionError:
    #    #     model.load_state_dict(torch.load(args.state,
    #     #         map_location=lambda storage, loc: storage))

    if args.mode == 'train':
        train_model(model, args, experiment_name, train_loader, val_loader, device)
    # else:
    #     test_model(model, device, dir_ + experiment_name, val_loader)
