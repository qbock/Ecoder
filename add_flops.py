import csv
import argparse
from distutils.fancy_getopt import wrap_text
from email.quoprimime import header_check
from get_flops import get_flops_from_model
from denseCNN import denseCNN
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('-i',"--input", type=str, default='All_Trials.csv', dest="infile",
                    help="name of csv file to read")

parser.add_argument('-o',"--output", type=str, default='All_Trials_OPs.csv', dest="outfile",
                    help="name of csv file write to")

def add_flops(infile, outfile):
    excluded_columns = ['emd', 'emd_error', 'trial_index', 'arm_name', 'learning_rate', 'batch_size', 'trial_status', 'generation_method']
    table = []
    header_dict = {}
    with open(infile, newline='\n') as csvfile:
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

            # Make dictionary of headers and header positions
            if i == 0:
                for idx, header in enumerate(row):
                    header_dict[idx] = header
                row.append("OPs")
                table.append(row)
            
            # Add networks info from row into model definition
            else:
                for j, item in enumerate(row):
                    if not header_dict[j].lower() in excluded_columns:
                        if item:
                            if 'pooling' in header_dict[j]:
                                params['CNN_pool'].append(item)
                            else:
                                item = int(float(item))
                                if 'filters' in header_dict[j]:
                                    params['CNN_layer_nodes'].append(item)
                                elif 'kernel' in header_dict[j]:
                                    params['CNN_kernel_size'].append(item)
                                elif 'stride' in header_dict[j]:
                                    params['CNN_strides'].append(item)
                                elif 'units' in header_dict[j]:
                                    params['Dense_layer_nodes'].append(item)    
                                else:
                                    print("Couldn't match header")
                                    break
                    
                # Define kernel size if not defined from CNN layer
                if not params['CNN_kernel_size']:
                    params['CNN_kernel_size'] = [3]

                for i in range(len(params['CNN_layer_nodes'])):
                    params['CNN_padding'].append('same')

                print(params)
                m = denseCNN()
                m.setpams(params)
                m.init()
                m_autoCNN , m_autoCNNen = m.get_models()
                ops = get_flops_from_model(m_autoCNNen)
                row.append(ops)
                table.append(row)

    df = pd.DataFrame(table)
    print(df)
    df.to_csv(outfile, header=False, index=False)


if __name__ == '__main__':
    args = parser.parse_args()
    add_flops(args.infile, args.outfile)
