import csv
import argparse
from distutils.fancy_getopt import wrap_text
from get_flops import get_flops_from_model
from denseCNN import denseCNN
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('-i',"--input", type=str, default='All_Trials.csv', dest="infile",
                    help="name of csv file to read")

parser.add_argument('-o',"--output", type=str, default='All_Trials_OPs.csv', dest="outfile",
                    help="name of csv file write to")

def main(args):
    table = []
    with open(args.infile, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i,row in enumerate(reader):

            params = {
                'CNN_layer_nodes'  : [],
                'CNN_kernel_size'  : [],
                'CNN_pool'         : [],
                'CNN_padding'      : [],
                'CNN_strides'      : [],
                'Dense_layer_nodes': [],
                'encoded_dim'      : 16,
                'shape'            : (8,8,1),
                'channels_first'   : False,
                'arrange'          : [],
                'arrMask'          : [],
                'calQMask'         : [],
                'maskConvOutput'   : [],
                'n_copy'           : 0,
                'loss'             : '',
                'activation'       : 'relu',
                'optimizer'        : 'adam',
                'learning_rate'    : 1e-3
            }
            # Ignore first row of column headers
            if not i == 0:
                # Add networks info from row into model definition
                for j, item in enumerate(row):
                    # Ignore columns that don't have network information in them
                    if not (item.lower() == "true" or item.lower() == "false" or item.lower() == "completed" or \
                            item.lower() == "sobol" or item.lower() == "gpei" or item.lower() == ""):
                        item = int(float(item))

                    if j == 4:
                        if item:
                            params['CNN_layer_nodes'].append(item)
                            params['CNN_padding'].append('same')
                    elif j == 5:
                        if item:
                            params['CNN_kernel_size'].append(item)
                    elif j == 6:
                        if item:
                            params['CNN_pool'].append(item)
                    elif j == 7:
                        if item:
                            params['CNN_strides'].append((item,item))
                    elif j == 8:
                        if item:
                            params['CNN_layer_nodes'].append(item)
                            params['CNN_padding'].append('same')
                    elif j == 9:
                        if item:
                            params['CNN_kernel_size'].append(item)
                    elif j == 10:
                        if item:
                            params['CNN_pool'].append(item)
                    elif j == 11:
                        if item:
                            params['CNN_strides'].append((item,item))
                    elif j == 12:
                        if item:
                            params['Dense_layer_nodes'].append(item)
                    elif j == 15:
                        if item:
                            params['Dense_layer_nodes'].append(item)
                    elif j == 16:
                        if item:
                            params['Dense_layer_nodes'].append(item)
                    elif j == 17:
                        if item:
                            params['CNN_layer_nodes'].append(item)
                            params['CNN_padding'].append('same')
                    elif j == 18:
                        if item:
                            params['CNN_kernel_size'].append(item)
                    elif j == 19:
                        if item:
                            params['CNN_pool'].append(item)
                    elif j == 20:
                        if item:
                            params['CNN_strides'].append((item,item))
                    if params['CNN_kernel_size'] == []:
                        params['CNN_kernel_size'].append(3)
                m = denseCNN()
                m.setpams(params)
                m.init(printSummary=False)
                m_autoCNN , m_autoCNNen = m.get_models()
                ops = get_flops_from_model(m_autoCNNen)
                row.append(ops)
                table.append(row)

            # Added OPS header
            else:
                row.append("OPs")
                table.append(row)

    df = pd.DataFrame(table)
    df.to_csv(args.outfile)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
