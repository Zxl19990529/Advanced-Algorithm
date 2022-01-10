import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode',type=str,default='test',help='test or train,ex: --mode train')
parser.add_argument('--log',type = str, default='log\\3_layer_tanh\\test.log')
args = parser.parse_args()

def read_trainlog(logpath):

    epoch_list =  []
    iter_list = []
    loss_list = []
    acc_list = []
    mae_list = []
    rmse_list = []
    mape_list = []
    batch_size = None
    for line in open(logpath,'r').readlines():
        line = line.strip()
        spt = line.split(' , ')
        epoch_list.append(int(spt[0].split(':')[-1]))
        iter_list.append(int(spt[1].split(':')[-1]))
        loss_list.append(float(spt[2].split(':')[-1]))
        acc_list.append(float(spt[3].split(':')[-1]))
        mae_list.append(float(spt[4].split(':')[-1]))
        rmse_list.append(float(spt[5].split(':')[-1]))
        mape_list.append(float(spt[6].split(':')[-1]))
        batch_size = int(spt[7].split(':')[-1])
    return epoch_list,iter_list,loss_list,acc_list,mae_list,rmse_list,mape_list,batch_size


def read_testlog(logpath):
    epoch_list =  []
    acc_list = []
    mae_list = []
    rmse_list = []
    mape_list = []
    batch_size = None
    for line in open(logpath,'r').readlines():
        line = line.strip()
        spt = line.split(' , ')
        epoch_list.append(int(spt[0].split(':')[-1]))
        acc_list.append(float(spt[1].split(':')[-1]))
        mae_list.append(float(spt[2].split(':')[-1]))
        rmse_list.append(float(spt[3].split(':')[-1]))
        mape_list.append(float(spt[4].split(':')[-1]))
        batch_size = int(spt[5].split(':')[-1])

    return epoch_list,acc_list,mae_list,rmse_list,mape_list,batch_size

if __name__ == '__main__':
    data = None
    if args.mode == 'train':
        epoch_list,iter_list,loss_list,acc_list,mae_list,rmse_list,mape_list,batch_size = read_trainlog(args.log)
        max_epoch = np.max(epoch_list)+1
        acc_epoch_array = np.zeros((max_epoch,1))
        loss_epoch_array = np.zeros((max_epoch,1))
        mae_epoch_array = np.zeros((max_epoch,1))
        rmse_epoch_array = np.zeros((max_epoch,1))
        mape_epoch_array = np.zeros((max_epoch,1))
        for i,epoch in enumerate(epoch_list):
            acc_epoch_array[epoch] += acc_list[i]
            loss_epoch_array[epoch] += loss_list[i]
            mae_epoch_array[epoch] += mae_list[i]
            rmse_epoch_array[epoch] += rmse_list[i]
            mape_epoch_array[epoch] += mape_list[i]
        iter_num = np.max(iter_list)
        iter_num = max(iter_num,1)
        acc_epoch_array /= iter_num
        loss_epoch_array /= iter_num
        mae_epoch_array /= iter_num
        rmse_epoch_array /= iter_num
        mape_epoch_array /= iter_num
        
        epoch_array = np.linspace(start=0,stop= max_epoch-1,num = max_epoch ).reshape(-1,1)
        plt.subplots_adjust(hspace=0.5) # 调整2个图的间距
        plt.subplot(2,1,1)
        acc, = plt.plot(epoch_array,acc_epoch_array,c='b')
        mae, = plt.plot(epoch_array,mae_epoch_array,c='g')
        rmse, = plt.plot(epoch_array,rmse_epoch_array,c='r')
        mape, = plt.plot(epoch_array,mape_epoch_array,c='m')
        plt.grid()
        plt.ylim(-1,2)
        plt.ylabel('Evaluation value')
        plt.xlabel('Epoch')
        plt.title('Trin Eval metircs')
        plt.legend(handles=[acc,mae,rmse,mape],labels=['acc','mae','rmse','mape'],loc='best')

        plt.subplot(2,1,2)
        loss, = plt.plot(epoch_array,loss_epoch_array,c='r')
        plt.grid()
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.title('Train loss')
        plt.legend(handles=[loss],labels=['loss'],loc='best')

        plt.show()
    elif args.mode == 'test':

        epoch_list,acc_list,mae_list,rmse_list,mape_list,batch_size = read_testlog(args.log)

        acc, = plt.plot(epoch_list,acc_list,c='b')
        mae, = plt.plot(epoch_list,mae_list,c='g')
        rmse, = plt.plot(epoch_list,rmse_list,c='r')
        mape, = plt.plot(epoch_list,mape_list,c='m')
        plt.legend(handles=[acc,mae,rmse,mape],labels=['acc','mae','rmse','mape'],loc='best')
        plt.ylim(0,2)
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel('Evaluation value')
        plt.title('Test Eval metrix')

        plt.show()