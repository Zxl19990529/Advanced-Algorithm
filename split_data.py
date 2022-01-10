import os,shutil
from typing_extensions import ParamSpecArgs
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file',default='PRSA2017_Data_20130301-20170228\PRSA_Data_Changping_20130301-20170228.csv',type=str)
args = parser.parse_args()

if __name__ == '__main__':
    # 提取14，15,16年的数据
    # 每个年度分成4个季度
    newdata = {'2013':[],'2014':[],'2015':[],'2016':[]}
    filename = args.file.split('\\')[-1]
    placename = filename.split('_')[2]

    outputfolder = os.path.join('split',placename)
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    with open(args.file,'r') as f:
        data = np.loadtxt(f,str,delimiter=',',skiprows=1)
    
    for index in range(data.shape[0]): # 清洗数据，NA值全删掉。
        No,year,month,day,hour,PM25,PM10,SO2,NO2,CO,O3,TEMP,PRES,DEWP,RAIN,wd,WSPM,station = data[index]
        # print (PM25.shape)
        if PM25 == 'NA' or TEMP == 'NA' or PRES == 'NA' or DEWP == 'NA' or RAIN == 'NA' or wd == 'NA' or WSPM == 'NA' :
            continue
        else:
            if 2013 <= int(year) <= 2016:
                newdata[year].append([month,PM25,TEMP,PRES,DEWP,RAIN,WSPM])
            else:
                continue
    
    for year in ['2013','2014','2015','2016']:
        oneyeardata = newdata[year]
        f1 = open(os.path.join(outputfolder,'%s_season1.csv'%year),'a')
        f2 = open(os.path.join(outputfolder,'%s_season2.csv'%year),'a')
        f3 = open(os.path.join(outputfolder,'%s_season3.csv'%year),'a')
        f4 = open(os.path.join(outputfolder,'%s_season4.csv'%year),'a')

        headline = 'month,PM25,TEMP,PRES,DEWP,RAIN,WSPM\n'
        f1.writelines(headline)
        f2.writelines(headline)
        f3.writelines(headline)
        f4.writelines(headline)

        for month,PM25,TEMP,PRES,DEWP,RAIN,WSPM in oneyeardata:
            if 1<=int(month)<=3:
                context = '%s,%s,%s,%s,%s,%s,%s\n'%(month,PM25,TEMP,PRES,DEWP,RAIN,WSPM)
                f1.writelines(context)
            elif 4<=int(month)<=6:
                context = '%s,%s,%s,%s,%s,%s,%s\n'%(month,PM25,TEMP,PRES,DEWP,RAIN,WSPM)
                f2.writelines(context)
            elif 7<=int(month)<=9:
                context = '%s,%s,%s,%s,%s,%s,%s\n'%(month,PM25,TEMP,PRES,DEWP,RAIN,WSPM)
                f3.writelines(context)
            elif 10<=int(month)<=12:
                context = '%s,%s,%s,%s,%s,%s,%s\n'%(month,PM25,TEMP,PRES,DEWP,RAIN,WSPM)
                f4.writelines(context)


