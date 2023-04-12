from re import I
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import csv
import numpy as np
import pandas as pd
import argparse

pd.set_option('display.max_colwidth', None)

parser = argparse.ArgumentParser()

parser.add_argument('-p',"--plot", type=str, default='EMD', dest="whichPlot",
                    help="specify whether to plot FLOPS vs EMD or EMD error")
parser.add_argument('-f',"--filename", type=str, default='run-1', dest="filename",
                    help="specify what results file to make plot for")
parser.add_argument('-e',"--epoch", type=str, default='20', dest="epochs",
                    help="specify the number of epochs the each trial in the results is trained for")


baseline = {15: [6544, 1.3308, 0.393],
            20: [6544, 1.2385, 0.359],
            30: [6544, 1.2385, 0.359],
            50: [6544, 1.1703, 0.339],
            100: [6544, 1.138, 0.324]
            }

class make_plots():
    def __init__(self, args):
        self.df = pd.read_csv('./results/' + args.filename + '.csv')
        self.emd = self.df.loc[:,"EMD"]
        self.error = self.df.loc[:,"EMD_error"]
        self.ops = self.df.loc[:,"OPs"]
        self.info = self.df.loc[:,'info']
        self.bl = baseline[int(args.epochs)]
        self.sc = []
        if args.whichPlot.lower() == 'emd':
            self.make_plot(self.emd, self.ops, self.info, "EMD", self.bl[1], self.bl[0], args.filename)
        else:
            self.make_plot(self.error, self.ops, self.info, "EMD_Error", self.bl[2], self.bl[0], args.filename)

    def make_plot(self, X, Y, info, x_title, baseline_x, baseline_y, filename):
        self.fig, self.ax = plt.subplots()
        self.ax.set_yscale('log')
        labled = [False,False,False,False]

        for x,y,i in zip(X, Y, info):
                num_layers = int(i.split('[')[1].split(']')[0].count(','))+1
                if num_layers == 0:
                        if labled[0]:
                                self.sc.append(self.ax.scatter(x,y, color='lightsteelblue', picker=True, pickradius=5))
                        else:
                                self.sc.append(self.ax.scatter(x,y, color='lightsteelblue', label='0 CNN layer', picker=True, pickradius=5))
                                labled[0] = True
                elif num_layers == 1:
                        if labled[1]:
                                self.sc.append(self.ax.scatter(x,y, color='cornflowerblue', picker=True, pickradius=5))
                        else:
                                self.sc.append(self.ax.scatter(x,y, color='cornflowerblue', label='1 CNN layer', picker=True, pickradius=5))
                                labled[1] = True
                elif num_layers == 2:
                        if labled[2]:
                                self.sc.append(self.ax.scatter(x,y, color='royalblue', picker=True, pickradius=5))
                        else:
                                self.sc.append(self.ax.scatter(x,y, color='royalblue', label='2 CNN layer', picker=True, pickradius=5))
                                labled[2] = True
                else:
                        if labled[3]:
                                self.sc.append(self.ax.scatter(x,y, color='midnightblue', picker=True, pickradius=5))
                        else:
                                self.sc.append(self.ax.scatter(x,y, color='midnightblue', label='3 CNN layer', picker=True, pickradius=5))
                                labled[3] = True

        self.sc.append(self.ax.scatter(baseline_x,baseline_y, marker='*', c='gold', label="baseline model"))
        print(self.sc)
        x_lower = min(X)*0.8
        # x_upper = max(X)/2
        x_upper = 5
        self.ax.set(xlim=(x_lower, x_upper),
                ylim=(1e3, 1e5))
        plt.legend(loc='upper right')
        plt.ylabel('OPs', fontsize=16)
        plt.xlabel(x_title, fontsize=18)
        plt.savefig('./results/' + x_title + '_vs_OPS-' + filename + '.jpg', dpi=1000)

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        idx = 0
        if event.inaxes == self.ax:
            for sc in self.sc:
                idx += 1
                cont, ind = sc.contains(event)
                if cont:
                    break
            if cont:
                emd = self.emd.loc[idx]
                error = self.error.loc[idx]
                info = self.info.loc[idx]
                print_model(emd,error,info)


def print_model(emd, error, info):
    print("\n\n\t Metrics")
    print(f'EMD: {emd}\t EMD Error: {error}')
    print('\t Model')
    print(info)

if __name__ == '__main__':
    args = parser.parse_args()
    make_plots(args)
    plt.show()

        # For finding pareto front models
        
        # df = pd.read_csv('./results/run-6.csv')
        # emd = df.loc[:,"EMD"]
        # error =df.loc[:,"EMD_error"]
        # ops =df.loc[:,"OPs"]
        # info = df.loc[:,'info']
        # bl = baseline[30]

        # fig, ax = plt.subplots()
        
        # ax.set(xlim=(0.5, 3.0), ylim=(1e3, 30000))
        
        # lookup = list(zip(emd,ops,info))
        # line, = ax.plot(emd,ops, 'o',
        #                 picker=True, pickradius=5)
        
        # def onpick(event):
        #         thisline = event.artist
        #         xdata = thisline.get_xdata()
        #         ydata = thisline.get_ydata()
        #         ind = event.ind
        #         x = xdata[ind]
        #         y = ydata[ind]
        #         print("**************************************************************************")
        #         for item in lookup:
        #                if (item[0] == x).any() and (item[1] == y).any():
        #                       print(item)

        
        # fig.canvas.mpl_connect('pick_event', onpick)

        # plt.show()


