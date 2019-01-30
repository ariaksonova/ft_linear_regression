import pandas
import numpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas
import os

from math import *
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
    #датасет может быть пусто
    length = len(predict_data) - 1
    errors = sum([(((wt0 + wt1 * predict_data.loc[i][0]) - predict_data.loc[i][1]) ** 2) for i in range(len(predict_data))]) / length
#    for i in range(length):
#        result = wt1 * predict_data.iloc[i][0]
#        result += wt0
#        result -= predict_data.iloc[i][1]
#        result = result ** 2
#        sum += result
#    errors = sum / length
    return sqrt(errors)

def change_wt(wt0, wt1, iter, data):
    wt0_tmp = wt0
    wt1_tmp = wt1
    length = len(data)
    sum0 = 0
    sum1 = 0
    l_rate = 0.1
    #iter = float(1 / (iter * iter))
    for i in range(length):
        sum0 += (wt1 * data.loc[i][0] + wt0) - data.loc[i][1]
        sum1 += ((wt1 * data.loc[i][0] + wt0) - data.loc[i][1]) * data.loc[i][0]
    #sum0 /= length
    #sum1 /= length
    wt0 = wt0 - ((l_rate * sum0) / len(data))
    wt1 = wt1 - ((l_rate * sum1) / len(data))
    wt = numpy.array([wt0, wt1])
    return wt, wt0_tmp, wt1_tmp

def normalize_dataset(data):
	x_max = max(data['km'])
	x_min = min(data['km'])
	data['km'] = data['km'].astype('float')
	for i in range(len(data['km'])):
		data['km'][i] = (data['km'][i] - x_min) / (x_max - x_min)
	print(data)
	return data

def train_model(predict_data):
	predict_data = normalize_dataset(predict_data)
	print(predict_data)
	wt0_tmp, wt1_tmp = 0, 0
	wt = numpy.array([0, 0])
	wt_plot = pandas.DataFrame()
	wt_plot = wt_plot.append({'errors': functional(0, 0, predict_data), 'iter': 0}, ignore_index=True)
	i = 0
	errors = 683
	while errors > 682:
		i += 1
		wt, wt0_tmp, wt1_tmp = change_wt(wt[0], wt[1], i, predict_data)
		if wt0_tmp == wt[0] and wt1_tmp == wt[1]:
			break
		if i % 100 == 0:
			print("I'm alive. Wait...")
		errors = functional(wt[0], wt[1], predict_data)
		print(i, errors, wt[0], wt[1])
        #wt_plot = wt_plot.append({'errors': functional(wt[0], wt[1], predict_data), 'iter': i}, ignore_index=True)
    #print_plot(wt_plot, 'wt_plot')
	return wt
