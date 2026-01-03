import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from scipy.io import loadmat,savemat
from data_processing import *
from Parameter import *
from load_dataset import *
import pandas as pd
import random
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from Model.HCLNN_SCAF_6 import *
from torch import nn
import time
from pytorchtools import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Evaluation import *


def set_seed(seed):
    random.seed(seed)  # random
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现

def train_main():

    paInputTrainNorm, paOutputTrainNorm, _, _ = load_and_prepare_data(0)

    X_train, Y_train = process_data(paInputTrainNorm, paOutputTrainNorm)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=False,drop_last=True)

    device = torch.device('cuda')

    net = HCLNNmodel_scaf_6()
    net = net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_loss = float('inf')

    # early_stopping = EarlyStopping(patience=patience, verbose=False, path=f'model_save/{name}')
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10)

    loss_train_csv = []

    for epoch in range(ep):
        net.train()
        print('Epoch: {}'.format(epoch + 1))
        start_time = time.time()
        losses_train = 0.0
        for data in train_dataloader:
            features, targets = data
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            out = net(features)
            loss = loss_fn(out, targets)
            losses_train += loss.item()
            loss.backward()
            optimizer.step()
        loss_train = losses_train/len(train_dataloader)
        loss_train_csv.append(loss_train)


        if loss_train < best_loss:
            torch.save(net, f'model_save/{name}')
            end_time = time.time()
            print(f'val_loss imporved from {best_loss} to {loss_train}.\n')
            best_loss = loss_train
            epoch_time = end_time - start_time
            print(f'time = {epoch_time:.1f} s - train_loss = {loss_train}. \n')

        else:
            end_time = time.time()
            epoch_time = end_time - start_time
            print(f'val_loss did not improve from {best_loss}.\n')
            print(f'time = {epoch_time:.1f} s - train_loss = {loss_train}. \n')

        # early_stopping(loss_train, net)
        # # scheduler.step(loss_val)
        #
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

    df = pd.DataFrame({
        'Step': range(1, len(loss_train_csv) + 1),  # 步骤编号，从 1 开始
        'Loss_train': loss_train_csv
    })

    df.to_csv('loss_values_HCLNN_SCAF.csv', index=False)




def main():
    set_seed(2024)
    train_main()

    for Test_ID in range(5):
        dataset_list = ['DataTrain', 'DataTest1', 'DataTest2', 'DataTest3', 'DataTest4']
        dataset = dataset_list[Test_ID]

        print(dataset)

        _, _, paInputTestNorm, paOutputTestNorm = load_and_prepare_data(Test_ID)

        [no_samples, _] = np.shape(paInputTestNorm)

        X_test = process_data_test_model(paInputTestNorm)
        test_dataset = TensorDataset(torch.Tensor(X_test))
        test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False, drop_last=True)

        device = torch.device('cuda')

        net = torch.load(f'model_save/{name}')
        net = net.eval()
        with torch.no_grad():
            prediction = []
            for data in test_dataloader:
                features = data[0]
                features = features.to(device)
                outputs = net(features)
                prediction.append(outputs.cpu())

        prediction = torch.cat(prediction, dim=0)

        Y_out = np.zeros((len(prediction), 1), dtype=np.complex128)

        for i in range(len(prediction)):
            Y_out[i, 0] = complex(prediction[i, 0], prediction[i, 1])

        Y_pred = Y_out * normalizing_factor

        outPA_whole = paOutputTestNorm * normalizing_factor
        inpPA_whole = paInputTestNorm * normalizing_factor

        if Test_ID==0:
            out_ignore_first_mem = outPA_whole[seqLength - 1:no_samples-145]  # First M samples are neglected
            inp_ignore_first_mem = inpPA_whole[seqLength - 1:no_samples-145]

        if Test_ID==1 or Test_ID==2:
            out_ignore_first_mem = outPA_whole[seqLength - 1:no_samples-53]  # First M samples are neglected
            inp_ignore_first_mem = inpPA_whole[seqLength - 1:no_samples-53]

        if Test_ID==3 or Test_ID==4:
            out_ignore_first_mem = outPA_whole[seqLength - 1:no_samples-501]  # First M samples are neglected
            inp_ignore_first_mem = inpPA_whole[seqLength - 1:no_samples-501]

        error = Y_pred - out_ignore_first_mem

        NMSE(error, out_ignore_first_mem)
        ACEPR_cal(error, inp_ignore_first_mem)

        if Test_ID==1 or Test_ID==2:
            piece1_end = 20000
            piece1_pred = Y_pred[0:piece1_end -seqLength+1]
            piece1_out = outPA_whole[seqLength-1:piece1_end]
            piece1_in = inpPA_whole[seqLength-1:piece1_end]
            error = piece1_pred-piece1_out

            print('For piece1')

            NMSE(error, piece1_out)
            ACEPR_cal(error, piece1_in)

            piece2_pred = Y_pred[piece1_end:piece1_end*2-seqLength+1]
            piece2_out = outPA_whole[piece1_end+seqLength-1:piece1_end*2-53]

            piece2_in = inpPA_whole[piece1_end+seqLength-1:piece1_end*2-53]
            error = piece2_pred-piece2_out

            print('For piece2')

            NMSE(error, piece2_out)
            ACEPR_cal(error, piece2_in)

        # if Test_ID == 4:
        #
        #     Y_pred = np.concatenate([np.zeros_like(Y_pred[0:11]), Y_pred, np.zeros_like(Y_pred[0:501])], axis=0)
        #
        #     savemat(f'data_save/PA_HCLNN.mat', {f'PA_HCLNN': Y_pred})

        if Test_ID == 1:

            Y_pred = np.concatenate([np.zeros_like(Y_pred[0:11]), Y_pred, np.zeros_like(Y_pred[0:53])], axis=0)

            savemat(f'data_save/PA_HCLNN.mat', {f'PA_HCLNN': Y_pred})



if __name__ == '__main__':
    main()

