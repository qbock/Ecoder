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
        self.df = pd.read_csv('All_Trials_OPs.csv')
        self.ops = self.df.loc[:,"OPs"]
        self.emd = self.df.loc[:,"EMD"]
        self.error = self.df.loc[:,"EMD_error"]
        self.info = self.df.loc[:, "info"]

        self.fig, self.ax = plt.subplots()
        self.ax.set_yscale('log')
        
        if args.whichPlot.lower() == "emd":
            self.sc = self.ax.scatter(self.emd,self.ops)
            self.ax.set(xlim=(1, 3),
                ylim=(1e3, 1e5))
            plt.xlabel('EMD', fontsize=18)
            plt.ylabel('OPs', fontsize=16)
            plt.savefig('EMDvsOPS.jpg', dpi=1000)
        else:
            self.sc = self.ax.scatter(self.error,self.ops)
            self.ax.set(xlim=(0.3, 1.2),
                ylim=(1e3, 1e5))
            plt.xlabel('EMD error', fontsize=18)
            plt.ylabel('OPs', fontsize=16)
            plt.savefig('EMDvsOPS.jpg', dpi=1000)

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
    

    def onclick(self, event):
        if event.inaxes == self.ax:
            cont, ind = self.sc.contains(event)
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
