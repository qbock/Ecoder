import os
import pandas as pd
from add_flops import add_flops
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-f',"--filename", type=str, default='run-1', dest="filename",
                    help="specify what results file to make plot for")

args = parser.parse_args()

# Put results from seperate experiments into one csv file
first = True
path = os.path.join(os.getcwd(), args.filename)
for filename in os.listdir(path):
    if ".csv" in filename:
        full_path = os.path.join(path,filename)
        if first:
            df = pd.read_csv(full_path)
            first = False
        else:
            df = pd.concat([df, pd.read_csv(full_path)])
        # os.remove(filename)

df.to_csv('All_Trails.csv', index=False)
add_flops('All_Trails.csv','All_Trails.csv')