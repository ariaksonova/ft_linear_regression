import pandas
import numpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas
import os

from train_model import *

def save(name='', fmt='png'):
    pwd = os.getcwd()
    iPath = './{}'.format(fmt)
    if not os.path.exists(iPath):
        os.mkdir(iPath)
    os.chdir(iPath)
    plt.savefig('{}.{}'.format(name, fmt), fmt='png')
    os.chdir(pwd)

def print_plot(predict_data, name):
    plt.plot(predict_data.iloc[:, 0], predict_data.iloc[:, 1], 'ro')
    save(name, fmt='png')
    plt.show()

def functional(wt0, wt1, predict_data):
    #датасет может быть пустой
    sum = 0
    errors = 0
    length = len(predict_data)
    for i in range(length):
        result = wt1 * predict_data.iloc[i][0]
        result += wt0
        result -= predict_data.iloc[i][1]
        result = result ** 2
        sum += result
    errors = int(sum / ( 2 * length))
    return errors

def change_wt(wt0, wt1, iter, data):
    length = len(data)
    sum0 = 0
    sum1 = 0
    iter = float(1 / (iter * iter))
    for i in range(length):
        sum0 += wt1 * data.iloc[i][0] + wt0 - data.iloc[i][1]
        sum1 += (wt1 * data.iloc[i][0] + wt0 - data.iloc[i][1]) * data.iloc[i][0]
    sum0 /= length
    sum1 /= length
    wt0 = float(wt0 - iter * sum0)
    wt1 = float(wt1 - iter * sum1)
    wt = numpy.array([wt0, wt1])
    return wt

def train_model(predict_data):
    predict_data = predict_data / 3650
    wt = numpy.array([0, 0])
    wt_plot = pandas.DataFrame()
    wt_plot = wt_plot.append({'errors': functional(0, 0, predict_data), 'iter': 0}, ignore_index=True)
    for i in range(1, 30000):
        wt = change_wt(wt[0], wt[1], i, predict_data)
        if i % 1000 == 0:
            print(i, functional(wt[0], wt[1], predict_data), wt[0] * 3650, wt[1] * 3650)
        #wt_plot = wt_plot.append({'errors': functional(wt[0], wt[1], predict_data), 'iter': i}, ignore_index=True)
    #print_plot(wt_plot, 'wt_plot')
    return wt