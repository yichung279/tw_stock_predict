import numpy as np
import re
import json
from glob import glob
from pprint import pprint

jsonlist = glob("stock_history/*")
jsonlist.sort()

stock_codes = ['3481','2408','2454','3231','3008','2330','2451','2388','2379','2890']
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
            data[i][j][k] = stocks[st_code][day][k]

np.save('stock_data.npy', data)
