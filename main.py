import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas
import os

import train_model

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

def print_plot(predict_data):
    plt.plot(predict_data.iloc[:, 0], predict_data.iloc[:, 1], 'ro')
    save('data_plot', fmt='png')
    plt.show()

if __name__ == "__main__":
    predict_data = get_data()
#    print_plot(predict_data)
    print(functional(0, 0, predict_data))