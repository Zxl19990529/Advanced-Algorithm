import os
import numpy as np 
import sklearn

def read_csv(file_path):
    data = None
    with open(file_path,'r') as f:
        data = np.loadtxt(f,str,delimiter=',',skiprows=1)
    new_data = []

    for index in range(data.shape[0]): # 清洗数据，NA值全删掉。
        month,PM25,TEMP,PRES,DEWP,RAIN,WSPM = data[index]
        # print (PM25.shape)
        new_data.append([month,PM25,TEMP,PRES,DEWP,RAIN,WSPM])
    new_data = np.array(new_data)
    return new_data

def normalization(data,std = 0.1):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range +std
 
 
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def MAE(y,y_hat):
    total_num = len(y)
    assert len(y) == len(y_hat)
    tmp1 = np.abs(y-y_hat)
    tmp2 = np.sum(tmp1)
    mae = tmp2/total_num
    return mae

def RMSE(y,y_hat):
    total_num = len(y)
    assert len(y) == len(y_hat)
    rmse = np.sqrt(np.sum((y-y_hat)**2)/total_num)
    return rmse

def MAPE(y,y_hat):
    total_num = len(y)
    assert len(y) == len(y_hat)

    mape = np.sum(np.abs((y-y_hat)/y))/total_num
    return mape

class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        return ((target - input) ** 2).mean(axis=0).sum() / 2.

    def backward(self, input, target):
        '''Your codes here'''
        return target - input

class MSE_loss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Mean square error'''

        # return ((target - input) ** 2).mean(axis=0).sum() / 2.
        loss = np.sum((target-input)**2).mean()
        return loss

    def backward(self, input, target):
        # the grad is y - y_hat
        return target - input

def cal_acc(output,label):
    r'''
    Calculate the mean accuracy of the regression result for all samples, whose number equals to batch_size.
    '''
    error = output - label
    error = np.sum(np.abs(error/np.abs(label)),axis = 0)/len(label)
    mean_correct = max(1 - error,0)
    return mean_correct