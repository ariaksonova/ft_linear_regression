import numpy

import matplotlib
matplotlib.use('Agg')
from info import *
import pickle
import matplotlib.pyplot as plt
import pandas
import os
import sys

from math import *
from copy import deepcopy

def save(name='', fmt='png'):
	pwd = os.getcwd()
	iPath = './{}'.format(fmt)
	if not os.path.exists(iPath):
		os.mkdir(iPath)
	os.chdir(iPath)
	plt.savefig('{}.{}'.format(name, fmt), fmt='png')
	os.chdir(pwd)

def functional(wt0, wt1, predict_data):
	length = len(predict_data) - 1
	errors = sum([(((wt0 + wt1 * x[0]) - x[1]) ** 2) for x in predict_data]) / length
	return sqrt(errors)

def change_wt(wt0, wt1, data):
	wt0_tmp = wt0
	wt1_tmp = wt1
	sum0 = 0
	sum1 = 0
	l_rate = 0.1
	for x in data:
		sum0 += (wt1 * x[0] + wt0) - x[1]
		sum1 += ((wt1 * x[0] + wt0) - x[1]) * x[0]
	wt0 = wt0 - ((l_rate * sum0) / len(data))
	wt1 = wt1 - ((l_rate * sum1) / len(data))
	wt = numpy.array([wt0, wt1])
	return wt, wt0_tmp, wt1_tmp

def normalize_dataset(data):
	x_max = max([x[0] for x in data])
	x_min = min([x[0] for x in data])
	for i in range(len(data)):
		data[i][0] = (data[i][0] - x_min) / (x_max - x_min)
	return data, x_min, x_max

def convert_data(data):
	new_data = []
	for i in range(len(data)):
		tmp = [int(data['km'][i]), int(data['price'][i])]
		new_data.append(tmp)
	return new_data

def get_data(filename):
	predict_data = pandas.read_csv(filename, sep=',')
	return(predict_data)

def train_model():
	try:
		filename = sys.argv[1]
	except Exception:
		print("Usage: python3 train.py [csv file]")
		sys.exit(1)
	try:
		data = get_data(filename)
		data = convert_data(data)
		predict_data, x_min, x_max = normalize_dataset(deepcopy(data))
		wt = numpy.array([0, 0])
		i = 0
		while True:
			i += 1
			wt, wt0_tmp, wt1_tmp = change_wt(wt[0], wt[1], predict_data)
			if wt0_tmp == wt[0] and wt1_tmp == wt[1]:
				break
			if i % 1000 == 0:
				print("I'm alive. Wait...")
		inf = Info(wt[0], wt[1], x_min, x_max, functional(wt[0], wt[1], predict_data))
	except Exception:
		print("Invalid input data! Try again.")
		sys.exit(1)
	with open("/tmp/ft_linear_regression", "wb") as file:
		pickle.dump(inf, file)
	print("DONE")
	print("First weight:", wt[0])
	print("Second weight:", wt[1])
	print("Plot with data and line resulting from linear regression into 'png' directory.")
	x_min_norm = min([x[0] for x in predict_data])
	x_max_norm = max([x[0] for x in predict_data])
	res_y = [(x_min_norm * wt[1] + wt[0]), (x_max_norm * wt[1] + wt[0])]
	res_x = [x_min, x_max]
	plt.plot([x[0] for x in data], [x[1] for x in data], 'o')
	plt.plot(res_x, res_y, marker='o')
	save("data_plot")

if __name__ == "__main__":
	train_model()