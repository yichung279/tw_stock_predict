import numpy as np
import re
import json
from glob import glob
from pprint import pprint

jsonlist = glob("stock_history/*")
jsonlist.sort()

# stock_codes = ['3481', '2408', '2454', '3231', '3008', '2330', '2451', '2388', '2379', '2356'] # tech
# normal =      [10,     100,    480,     30,    5000,   250,    110,    50,     130,    25]
# stock_codes = ['2801','2809', '2812', '2836', '2845', '2887', '2886', '2885', '2884', '2890'] # bank
# normal = [18, 40, 10, 10, 10, 15, 25, 15, 20, 13]
stock_codes = ['2002', '1402' ,'1314', '2023', '1102', '2014', '1301', '1326', '1312', '1710'] # tradition
normal = [25, 35, 15, 13, 45, 15, 110, 125, 30, 32]
stocks = np.zeros((10, 1000, 5))

for j, jsonname in enumerate(jsonlist[-1000:]) :
    with open(jsonname,'r') as jsonfile:
        js = json.loads(jsonfile.read())
    
        for i, stock_code in enumerate(stock_codes):
            stocks[i][j][0] = 0 if js[stock_code]['adj_close'] == "NULL" else js[stock_code]['adj_close']
            stocks[i][j][1] = 0 if js[stock_code]['close'] == "NULL" else js[stock_code]['close']
            stocks[i][j][2] = 0 if js[stock_code]['high'] == "NULL" else js[stock_code]['high']
            stocks[i][j][3] = 0 if js[stock_code]['low'] == "NULL" else js[stock_code]['low']
            stocks[i][j][4] = 0 if js[stock_code]['open'] == "NULL" else js[stock_code]['open']

data = np.zeros((9810, 20, 5))
for i in range(9810):
    for j in range(20):
        for k in range(5):
            st_code = i // 981
            day = (i % 981) + j
            data[i][j][k] = stocks[st_code][day][k] / normal[st_code]

np.save('training_data/tradition_data.npy', data)
