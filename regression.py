import os
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from utils import read_csv,normalization,standardization,MAE,MAPE,RMSE
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file',default='split\\Aotizhongxin\\2014_season1.csv',type=str)
parser.add_argument('--mode',default=0,type = int,help='0 or 1, 0 is linear and 1 is none linear,which uses SVR Regression')
args = parser.parse_args()

if __name__ == '__main__':
    filepath = args.file
    modenamelist = ['Linear','SVR']
    modename = modenamelist[args.mode]
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
    
    # 定义回归
    regmodel = None
    if args.mode == 0:
        regmodel = LinearRegression()
        regmodel.fit(x_train,y_train)
    elif args.mode == 1:
        regmodel = SVR()
        regmodel.fit(x_train,y_train)
    
    # 结果
    train_out = regmodel.predict(x_train).reshape(-1,1)
    test_out = regmodel.predict(x_test).reshape(-1,1)

    # 训练正确率
    train_mae = MAE(train_out,y_train)
    train_rmse = RMSE(train_out,y_train)
    train_mape = MAPE(train_out,y_train)

    # 测试正确率
    test_mae = MAE(test_out,y_test)
    test_rmse = RMSE(test_out,y_test)
    test_mape = MAPE(test_out,y_test)

    # 训练正确率
    print('Train MAE',train_mae)
    print('Train RMSE',train_rmse)
    print('Train MAPE',train_mape)

    # 测试正确率
    print('Test MAE',test_mae)
    print('Test RMSE',test_rmse)
    print('Test MAPE',test_mape)

    # 输出结果
    spt = args.file.split('\\')
    station = spt[1]
    season = spt[2].split('.')[0]
    resfolder = os.path.join('experiment',modename)
    if not os.path.exists(resfolder):
        os.makedirs(resfolder)
    if not os.path.exists(os.path.join(resfolder,'record.csv')):
        f = open(os.path.join('experiment',modename,'record.csv'),'a')
        headline = 'station,season,MAE,RMSE,MAPE\n'
        f.writelines(headline)
    f = open(os.path.join('experiment',modename,'record.csv'),'a')
    context = '%s,%s,%s,%s,%s\n'%(station,season,test_mae,test_rmse,test_mape)
    f.writelines(context)
    f.close()

    # 作图

    # train_digsx = np.linspace(1,len(y_train),len(y_train))
    # test_digsx = np.linspace(1,len(y_test),len(y_test))

    # l1, = plt.plot(test_digsx,test_out,c='r')
    # l2, = plt.plot(test_digsx,y_test,c='b')
    # plt.xlabel('sample id')
    # plt.ylabel('PM2.5')
    # plt.title('Linear regression of Aotizhongxin 2014_season1')
    # plt.legend(handles=[l1,l2],labels=['linear','gth'],loc='upper center')
    # plt.show()

