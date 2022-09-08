from re import I
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import argparse

pd.set_option('display.max_colwidth', None)

parser = argparse.ArgumentParser()

parser.add_argument('-p',"--plot", type=str, default='EMD', dest="whichPlot",
                    help="specify whether to plot FLOPS vs EMD or EMD error")


class plot():
    def __init__(self, args):
        self.df = pd.read_csv('run-1.csv')
        self.ops = self.df.loc[:,"OPs"]
        self.emd = self.df.loc[:,"EMD"]
        self.error = self.df.loc[:,"EMD_error"]
        self.info = self.df.loc[:, "info"]

        self.fig, self.ax = plt.subplots()
        self.ax.set_yscale('log')
        
        self.sc = []

        if args.whichPlot.lower() == "emd":
            labled = [False,False,False,False]
            for o,e,i in zip(self.ops, self.emd, self.info):
                num_layers = int(i.split('[')[1].split(']')[0].count(','))+1
                if num_layers == 0:
                    if labled[0]:
                        self.sc.append(self.ax.scatter(e,o, color='lightsteelblue'))
                    else:
                        self.sc.append(self.ax.scatter(e,o, color='lightsteelblue', label='0 CNN layer'))
                        labled[0] = True
                elif num_layers == 1:
                    if labled[1]:
                        self.sc.append(self.ax.scatter(e,o, color='cornflowerblue'))
                    else:
                        self.sc.append(self.ax.scatter(e,o, color='cornflowerblue', label='1 CNN layer'))
                        labled[1] = True
                elif num_layers == 2:
                    if labled[2]:
                        self.sc.append(self.ax.scatter(e,o, color='royalblue'))
                    else:
                        self.sc.append(self.ax.scatter(e,o, color='royalblue', label='2 CNN layer'))
                        labled[2] = True
                else:
                    if labled[3]:
                        self.sc.append(self.ax.scatter(e,o, color='midnightblue'))
                    else:
                        self.sc.append(self.ax.scatter(e,o, color='midnightblue', label='3 CNN layer'))
                        labled[3] = True
            self.ax.scatter(1.067,6544, marker='*', c='gold', label="baseline model")
            self.ax.set(xlim=(0.5, 3),
                ylim=(1e3, 1e5))
            plt.legend(loc='upper right')
            plt.xlabel('EMD', fontsize=18)
            plt.ylabel('OPs', fontsize=16)
            plt.savefig('EMDvsOPS.jpg', dpi=1000)
        else:
            labled = [False,False,False,False]
            for o,e,i in zip(self.ops, self.error, self.info):
                num_layers = int(i.split('[')[1].split(']')[0].count(','))+1
                if num_layers == 0:
                    if labled[0]:
                        self.sc.append(self.ax.scatter(e,o, color='lightsteelblue'))
                    else:
                        self.sc.append(self.ax.scatter(e,o, color='lightsteelblue', label='0 CNN layer'))
                        labled[0] = True
                elif num_layers == 1:
                    if labled[1]:
                        self.sc.append(self.ax.scatter(e,o, color='cornflowerblue'))
                    else:
                        self.sc.append(self.ax.scatter(e,o, color='cornflowerblue', label='1 CNN layer'))
                        labled[1] = True
                elif num_layers == 2:
                    if labled[2]:
                        self.sc.append(self.ax.scatter(e,o, color='royalblue'))
                    else:
                        self.sc.append(self.ax.scatter(e,o, color='royalblue', label='2 CNN layer'))
                        labled[2] = True
                else:
                    if labled[3]:
                        self.sc.append(self.ax.scatter(e,o, color='midnightblue'))
                    else:
                        self.sc.append(self.ax.scatter(e,o, color='midnightblue', label='3 CNN layer'))
                        labled[3] = True
            self.ax.scatter(1.067,6544, marker='*', c='gold', label="baseline model")
            self.ax.set(xlim=(0.3, 1.2),
                ylim=(1e3, 1e5))
            plt.legend(loc='upper right')
            plt.xlabel('EMD error', fontsize=18)
            plt.ylabel('OPs', fontsize=16)
            plt.savefig('EMDvsOPS.jpg', dpi=1000)

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
    

    def onclick(self, event):
        if event.inaxes == self.ax:
            for sc in self.sc:
                cont, ind = sc.contains(event)
                if cont:
                    break
            if(cont):
                emd = self.emd.loc[ind['ind']]
                error = self.error.loc[ind['ind']]
                info = self.info.loc[ind['ind']]
                out = pd.concat([emd,error, info], axis=1)
                print(out)

if __name__ == '__main__':
    args = parser.parse_args()
    coords = []
    opsPlot = plot(args)
    plt.show()
