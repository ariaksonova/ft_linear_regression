import matplotlib
matplotlib.use('Agg')
import numpy
import matplotlib.pyplot as plt
import pandas
import os

from train_model import *

from train_model import *

def save(name='', fmt='png'):
    pwd = os.getcwd()
    iPath = './{}'.format(fmt)
    if not os.path.exists(iPath):
        os.mkdir(iPath)
    os.chdir(iPath)
    plt.savefig('{}.{}'.format(name, fmt), fmt='png')
    os.chdir(pwd)


def get_data():
    predict_data = pandas.read_csv('data.csv', sep=',')
    return(predict_data)

def print_plot(wt_plot, predict_data, name):
    fig, ax = plt.subplots()
    x = numpy.linspace(100, 100, 100)
    y1 = wt_plot[0] * 3650 + wt_plot[1] * 3650 * x
    # ax.plot(x, y1, color="blue")
    ax.plot(predict_data.iloc[:, 0], predict_data.iloc[:, 1], 'ro')#, wt_plot.iloc[:, 0], wt_plot.iloc[:, 1], 'go')
    save(name, fmt='png')


if __name__ == "__main__":
    predict_data = get_data()
    wt = train_model(predict_data)
    print(wt[0], wt[1])
    print_plot(wt, predict_data, 'data_plot')