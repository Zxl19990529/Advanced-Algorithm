import os
import numpy as np
import argparse
from utils import read_csv,normalization,standardization,MAE,MAPE,RMSE,MSE_loss,cal_acc
from dataset import data_iterator
import yaml
from models import Network
from basic_layer import Linear,Tanh,Relu,Sigmoid
from sklearn.model_selection import train_test_split
parser = argparse.ArgumentParser()
parser.add_argument('--file',default='split\\Aotizhongxin\\2014_season1.csv',type=str)
parser.add_argument('--log_dir',type=str,default='log\\3_layer_tanh')
args = parser.parse_args()

if __name__ == '__main__':
    filepath = args.file
    newdata = read_csv(filepath)
    # PM25,TEMP,PRES,DEWP,RAIN,wd,WSPM = newdata[:,0],newdata[:,1],newdata[:,2],newdata[:,3],newdata[:,4],newdata[:,5],newdata[:,6]
    month,PM25,TEMP,PRES,DEWP,RAIN,WSPM = newdata[:,0],newdata[:,1],newdata[:,2],newdata[:,3],newdata[:,4],newdata[:,5],newdata[:,6]
    PM25 = PM25.astype(np.float16).reshape(-1,1)
    TEMP = TEMP.astype(np.float16).reshape(-1,1)
    PRES = PRES.astype(np.float16).reshape(-1,1)
    DEWP = DEWP.astype(np.float16).reshape(-1,1)
    RAIN = RAIN.astype(np.float16).reshape(-1,1)
    WSPM = WSPM.astype(np.float16).reshape(-1,1)

    # 数据归一化
    PM25 = normalization(PM25)
    TEMP = normalization(TEMP)
    PRES = normalization(PRES)
    DEWP = normalization(DEWP)
    RAIN = normalization(RAIN)
    WSPM = normalization(WSPM)

    # 划分数据及
    X = np.concatenate((TEMP,PRES,DEWP,RAIN,WSPM),1)
    Y = PM25.copy()
    x_train, x_test, y_train,y_test  = train_test_split(X,Y,test_size=0.2)
    

    # 定义网络
    config = yaml.load(open('config\config.yml','r'),Loader=yaml.SafeLoader)
    train_cfg = config['train']
    test_cfg = config['test']
    net_cfg = config['network']

    model = Network()
    model.add(Linear('fc1', 5, 256, 0.001))
    model.add(Tanh('tanh1'))
    model.add(Linear('fc2',256,128,0.001))
    model.add(Tanh('tanh2'))
    model.add(Linear('fc3',128,1,0.001))
    model.train()

    loss_func = MSE_loss(name='MSE_loss')
    loss_list = []
    total_iteration = 0

    # init log dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    train_log_file_path = os.path.join(args.log_dir,train_cfg['log'])
    test_log_file_path = os.path.join(args.log_dir,test_cfg['log'])

    for epoch in range(train_cfg['max_epoch']):
        model.train()
        batch_acc = []
        batch_loss = []

        batch_mae = []
        batch_rmse = []
        batch_mape = []
        for iteration ,(train_input, label) in enumerate(data_iterator(x_train, y_train, train_cfg['batch_size'])):
            batch_size = train_cfg['batch_size']

            # forward
            train_output = model.forward(train_input)

            # cal loss
            loss_val = loss_func.forward(train_output, label)

            # cal mae,rmse mape
            mae_val = MAE(train_output,label)
            rmse_val = RMSE(train_output,label)
            mape_val = MAPE(train_output,label)

            # cal accuracy
            acc_val = cal_acc(train_output, label)

            # backward
            grad = loss_func.backward(train_output,label)
            model.backward(grad)
            model.update(net_cfg)

            # log record
            batch_acc.append(acc_val)
            batch_loss.append(loss_val)

            batch_mae.append(mae_val)
            batch_rmse.append(rmse_val)
            batch_mape.append(mape_val)

            if iteration % train_cfg['record_freq'] == 0: # show evey 50 iterations
                batch_loss_mean = np.mean(batch_loss)
                batch_acc_mean = np.mean(batch_acc)
                batch_mae_mean = np.mean(batch_mae)
                batch_rmse_mean = np.mean(batch_rmse)
                batch_mape_mean = np.mean(batch_mape)

                context = '[Train] Epoch:%d , iter:%d , batch loss:%.6f , batch acc:%.6f , MAE:%.6f , RMSE:%.6f , MAPE:%.6f , bz:%d'%(epoch,iteration,batch_loss_mean,batch_acc_mean,batch_mae_mean,batch_rmse_mean,batch_mape_mean,batch_size)
                f = open(train_log_file_path,'a')
                f.writelines(context+'\n')
                f.close()
                print(context)
                batch_acc = []
                batch_loss = []
            total_iteration += 1
        if epoch == 10:
            tmp = 1
            for onelayer in model.layer_list:
                if 'fc' in onelayer.name:
                    onelayer.lr *= 0.9
                else:
                    continue
        
        # test model
        test_acc_list = []
        test_mae_list = []
        test_rmse_list = []
        test_mape_list = []

        if epoch % train_cfg['test_epoch'] == 0:
            # feeze the model
            model.eval()
            for test_input, label in data_iterator(x_test,y_test,test_cfg['batch_size']):
                batch_size = test_cfg['batch_size']
                test_output = model.forward(test_input)

                # cal accuracy
                test_acc = cal_acc(test_output,label)

                # cal mae,rmse mape
                test_mae_val = MAE(test_output,label)
                test_rmse_val = RMSE(test_output,label)
                test_mape_val = MAPE(test_output,label)


                # record log
                test_acc_list.append(test_acc)

                test_mae_list.append(test_mae_val)
                test_rmse_list.append(test_rmse_val)
                test_mape_list.append(test_mape_val)

            test_acc_mean = np.mean(test_acc_list)
            test_mae_mean = np.mean(test_mae_list)
            test_rmse_mean = np.mean(test_rmse_list)
            test_mape_mean = np.mean(test_mape_list)
            
            context = '[Test] Epoch:%d , epoch accay:%.6f , MAE:%.6f , RMSE:%.6f , MAPE:%.6f , bz:%d'%(epoch,test_acc_mean,test_mae_mean,test_rmse_mean,test_mape_mean,test_cfg['batch_size'])
            print(context)
            f = open(test_log_file_path,'a')
            f.writelines(context+'\n')
            f.close()