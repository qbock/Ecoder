import os
import pandas as pd
from add_flops import add_flops

# Put results from seperate experiments into one csv file
first = True
for filename in os.listdir(os.getcwd()):
    if ".csv" in filename:
        if first:
            df = pd.read_csv(filename)
            first = False
        else:
            df = pd.concat([df, pd.read_csv(filename)])
        # os.remove(filename)

df.to_csv('All_Trails.csv', index=False)
add_flops('All_Trails.csv','All_Trails.csv')