from xmlrpc.client import boolean
import numpy as np
import pandas as pd
import os

from denseCNN import denseCNN
from utils.logger import _logger
from telescope import telescopeMSE8x8
from networks import arrange_dict
from datetime import datetime

from argparse import ArgumentParser
from ax.service.ax_client import AxClient
from train import ( normalize,
                    unnormalize,
                    load_data,
                    split,
                    train,
                    evaluate_model)

parser = ArgumentParser()

parser.add_argument('-o',"--odir", type=str, default='Ax_optimzation', dest="odir",
                    help="output directory")
parser.add_argument('-i',"--inputFile", type=str, default='nElinks_5/', dest="inputFile",
                    help="input TSG files")
parser.add_argument("--loss", type=str, default=None, dest="loss",
                    help="force loss function to use")
parser.add_argument("--quantize", action='store_true', default=False, dest="quantize",
                    help="quantize the model with qKeras. Default precision is 16,6 for all values.")
parser.add_argument("--epochs", type=int, default = 200, dest="epochs",
                    help="number of epochs to train")
parser.add_argument("--nELinks", type=int, default = 5, dest="nElinks",
                    help="n of active transceiver e-links eTX")

parser.add_argument("--skipPlot", action='store_true', default=True, dest="skipPlot",
                    help="skip the plotting step")
parser.add_argument("--full", action='store_true', default = False,dest="full",
                    help="run all algorithms and metrics")

parser.add_argument("--quickTrain", action='store_true', default = False,dest="quickTrain",
                    help="train w only 5k events for testing purposes")
parser.add_argument("--retrain", action='store_true', default = False,dest="retrain",
                    help="retrain models even if weights are already present for testing purposes")
parser.add_argument("--evalOnly", action='store_true', default = False,dest="evalOnly",
                    help="only evaluate the NN on the input sample, no train")

parser.add_argument("--double", action='store_true', default = False,dest="double",
                    help="test PU400 by combining PU200 events")
parser.add_argument("--overrideInput", action='store_true', default = False,dest="overrideInput",
                    help="disable safety check on inputs")
parser.add_argument("--nCSV", type=int, default = 1, dest="nCSV",
                    help="n of validation events to write to csv")
parser.add_argument("--maxVal", type=int, default = -1, dest="maxVal",
                    help="clip outputs to maxVal")
parser.add_argument("--AEonly", type=int, default=1, dest="AEonly",
                    help="run only AE algo")
parser.add_argument("--rescaleInputToMax", action='store_true', default=False, dest="rescaleInputToMax",
                    help="rescale the input images so the maximum deposit is 1. Else normalize")
parser.add_argument("--rescaleOutputToMax", action='store_true', default=False, dest="rescaleOutputToMax",
                    help="rescale the output images to match the initial sum of charge")
parser.add_argument("--nrowsPerFile", type=int, default=500000, dest="nrowsPerFile",
                    help="load nrowsPerFile in a directory")
parser.add_argument("--occReweight", action='store_true', default = False,dest="occReweight",
                    help="train with per-event weight on TC occupancy")

parser.add_argument("--maskPartials", action='store_true', default = False,dest="maskPartials",
                    help="mask partial modules")
parser.add_argument("--maskEnergies", action='store_true', default = False,dest="maskEnergies",
                    help="Mask energy fractions <= 0.05")
parser.add_argument("--saveEnergy", action='store_true', default = False,dest="saveEnergy",
                    help="save SimEnergy from input data")
parser.add_argument("--noHeader", action='store_true', default = False,dest="noHeader",
                    help="input data has no header")

parser.add_argument('--opt_iter', type=int, default = 20, dest="opt_iter",
            help="number of iterations to use in the Bayesian Optimization loop")
parser.add_argument('--boEpochs', type=int, default = 30, dest="bayesian_op_epochs",
            help="number of epochs to train within Bayesian Optimization")
parser.add_argument('--maxLayers', type=int, default = 3, dest="max_CNN_layers",
            help="number of epochs to train within Bayesian Optimization")

def save_plot(plot, name):
    data = plot[0]['data']
    lay = plot[0]['layout']

    import plotly.graph_objects as go
    fig = {
        "data": data,
        "layout": lay,
    }
    go.Figure(fig).write_image(name)

def evaluation(args, model):
    _logger.info(args)

    # load data
    data_values,phys_values = load_data(args)

    # normalize input charge data
    normdata,maxdata,sumdata = normalize(data_values.copy(),rescaleInputToMax=args.rescaleInputToMax,sumlog2=True)
    maxdata = maxdata / 35. # normalize to units of transverse MIPs
    sumdata = sumdata / 35. # normalize to units of transverse MIPs

    # evaluate performance
    from utils.metrics import emd

    eval_dict = {
        # compare to other algorithms
        'algnames'    : ['ae','stc','thr_lo','thr_hi','bc'],
        'metrics'     : {'EMD': emd},
        "occ_nbins"   : 12,
        "occ_range"   : (0,24),
        "occ_bins"    : [0,2,5,10,15],
        "chg_nbins"   : 20,
        "chg_range"   : (0,200),
        "chglog_nbins": 20,
        "chglog_range": (0,2.5),
        "chg_bins"    : [0,2,5,10,50],
        "occTitle"    : r"occupancy [1 MIP$_{\mathrm{T}}$ TCs]"       ,
        "logMaxTitle" : r"log10(Max TC charge/MIP$_{\mathrm{T}}$)",
        "logTotTitle" : r"log10(Sum of TC charges/MIP$_{\mathrm{T}}$)",
    }

    # performance dictionary
    perf_dict={}

    #Putting back physics columns below once training is done
    print(f'len phys_values: {len(phys_values)}')
    Nphys = round(len(phys_values)*0.2)
    print(f'Nphys = {Nphys}')
    phys_val_input = phys_values[:Nphys]
    # phys_val_input=phys_val_input

    # Create directory to store results
    now = datetime.now()
    time = now.strftime("%d-%m-%Y_%H-%M-%S")
    file_name = "Experiment-" + time

    orig_dir = os.getcwd()
    if not os.path.exists(args.odir): os.mkdir(args.odir)
    os.chdir(args.odir)

    if not os.path.exists(file_name): os.mkdir(file_name)
    os.chdir(file_name)

    # train the model
    _logger.info("Model is a denseCNN")
    m = denseCNN()
    m.setpams(model['params'])
    m.init()

    shaped_data = m.prepInput(normdata)

    val_input, train_input, val_ind, train_ind = split(shaped_data)

    m_autoCNN , m_autoCNNen = m.get_models()
    model['m_autoCNN'] = m_autoCNN
    model['m_autoCNNen'] = m_autoCNNen

    val_max = maxdata[val_ind]
    val_sum = sumdata[val_ind]

    history = train(m_autoCNN,m_autoCNNen,
                    train_input,train_input,val_input,
                    name=model['name'],
                    n_epochs = args.bayesian_op_epochs,
                    )

    # evaluate model
    _logger.info('Evaluate AutoEncoder, model %s'%model['name'])
    input_Q, cnn_deQ, cnn_enQ = m.predict(val_input)

    input_calQ  = m.mapToCalQ(input_Q)   # shape = (N,48) in CALQ order
    output_calQ_fr = m.mapToCalQ(cnn_deQ)   # shape = (N,48) in CALQ order
    _logger.info('inputQ shape')
    print(input_Q.shape)
    _logger.info('inputcalQ shape')
    print(input_calQ.shape)

    _logger.info('Restore normalization')
    input_Q_abs = np.array([input_Q[i]*(val_max[i] if args.rescaleInputToMax else val_sum[i]) for i in range(0,len(input_Q))]) * 35.   # restore abs input in CALQ unit
    input_calQ  = np.array([input_calQ[i]*(val_max[i] if args.rescaleInputToMax else val_sum[i]) for i in range(0,len(input_calQ)) ])  # shape = (N,48) in CALQ order
    output_calQ =  unnormalize(output_calQ_fr.copy(), val_max if args.rescaleOutputToMax else val_sum, rescaleOutputToMax=args.rescaleOutputToMax)

    _logger.info('Renormalize inputs of AE for comparisons')
    occupancy_0MT = np.count_nonzero(input_calQ.reshape(len(input_Q),48),axis=1)
    occupancy_1MT = np.count_nonzero(input_calQ.reshape(len(input_Q),48)>1.,axis=1)

    charges = {
        'input_Q'    : input_Q,
        'input_Q_abs': input_Q_abs,
        'input_calQ' : input_calQ,            # shape = (N,48) (in abs Q)   (in CALQ 1-48 order)
        'output_calQ': output_calQ,           # shape = (N,48) (in abs Q)   (in CALQ 1-48 order)
        'output_calQ_fr': output_calQ_fr,     # shape = (N,48) (in Q fr)   (in CALQ 1-48 order)
        'cnn_deQ'    : cnn_deQ,
        'cnn_enQ'    : cnn_enQ,
        'val_sum'    : val_sum,
        'val_max'    : val_max,
    }

    aux_arrs = {
        'occupancy_1MT':occupancy_1MT
    }

    perf_dict[model['label']] , model['summary_dict'] = evaluate_model(model,charges,aux_arrs,eval_dict,args)

    os.chdir(orig_dir)

    return model['summary_dict']['EMD_ae'], model['summary_dict']['EMD_ae_err']

def build_model(args, parameterization, num_layers):

    filters, kernels, poolings, strides, paddings = [], [], [], [], []

    for i in range(num_layers):
        filters.append(parameterization.get(f"filters_{i+1}"))
        kernels.append(parameterization.get(f"kernel_{i+1}"))
        poolings.append(parameterization.get(f"pooling_{i+1}"))
        stride = parameterization.get(f"stride_{i+1}")
        strides.append((stride, stride))
        paddings.append('same')

    name = f'8x8_c[{*filters,}]_k[{*kernels,}]_p[{*poolings,}]_s{*strides,}tele'
    label = f'8x8_c[{*filters,}]_k[{*kernels,}]_p[{*poolings,}]_s[{*strides,}](tele)'

    model = {
            'name':name,
            'label':label,
            'arr_key':'8x8',
            'isQK':False,
            'params':{
                'shape':(8,8,1),
                'loss':telescopeMSE8x8,
                'CNN_layer_nodes':filters,
                'CNN_kernel_size':kernels,
                'CNN_strides':strides,
                'CNN_padding':paddings,
                'CNN_pool':poolings
             }
    }

    defaults = {'channels_first': False,
            'encoded_dim': 16,
            }

    arrange = arrange_dict[model['arr_key']]
    model['params'].update({
        'arrange': arrange['arrange'],
        'arrMask': arrange['arrMask'],
        'calQMask': arrange['calQMask'],
    })

    if not 'isDense2D' in model.keys(): model.update({'isDense2D':False})
    if not 'isQK' in model.keys(): model.update({'isQK':False})
    if not 'ws' in model.keys(): model.update({'ws':''})
    for p,v in defaults.items():
        if not p in model['params'].keys():
            model['params'].update({p:v})

    nBits_encod = dict()
    if(args.nElinks==2):
        nBits_encod  = {'total':  3, 'integer': 1,'keep_negative':0}
    elif(args.nElinks==3):
        nBits_encod  = {'total':  5, 'integer': 1,'keep_negative':0}
    elif(args.nElinks==4):
        nBits_encod  = {'total':  7, 'integer': 1,'keep_negative':0}
    elif(args.nElinks==5):
        nBits_encod  = {'total':  9, 'integer': 1,'keep_negative':0} # 0 to 2 range, 8 bit decimal
    else:
        _logger.warning('Must specify encoding bits for nElink %i'%args.nElinks)

    if not 'nBits_encod' in model['params'].keys():
        model['params'].update({'nBits_encod':nBits_encod})

    if args.loss:
        model['params']['loss'] = args.loss

    return model

def bo_evaluation(args, parameterization, num_layers):

    model = build_model(args, parameterization, num_layers)
    EMD, EMD_error = evaluation(args, model)
    return {"EMD": EMD, "EMD_error": EMD_error}

def main(args):

    ax = AxClient()

    # Run a seperate experiment for each number of layers
    for num_layers in range(1, args.max_CNN_layers):

        """ Set parameters and value of those parameters to be used in the optimization. There is a
        parameter (knob) for number of filters, kernal size, whether or not to use pooling, and the
        stride for each CNN layer. """

        ax_parameters = []

        for i in range(num_layers):
            ax_parameters.append({"name": f"filters_{i+1}",
                                  "type": "range",
                                  "bounds": [1, 64],
                                  "value_type": "int"
            })
            ax_parameters.append({"name": f"kernel_{i+1}",
                                  "type": "choice",
                                  "is_ordered": True,
                                  "value_type": "int",
                                  "values": [1,3,5]
            })
            ax_parameters.append({"name": f"pooling_{i+1}",
                                  "type": "choice",
                                  "is_ordered": True,
                                  "value_type": "bool",
                                  "values": [True, False]
            })
            ax_parameters.append({"name": f"stride_{i+1}",
                                  "type": "choice",
                                  "is_ordered": True,
                                  "value_type": "int",
                                  "values": [1,2,4]
            })

        # Create Ax experiment
        ax.create_experiment(
            name="ECON_T " + str(num_layers) + "layers",
            parameters=ax_parameters,
            objective_name="EMD",
            minimize=True
        )

        # Optimization loop
        for i in range(args.opt_iter):
            parameterization, idx = ax.get_next_trial()
            ax.complete_trial(trial_index=idx, raw_data=bo_evaluation(args, parameterization, num_layers))

        # Print results and save them to a file
        trials = ax.get_trails_data_fram().sort_values('trail_index')
        trials_vs_EMD = ax.get_optimization_trace(objective_optimum=EMD.fmin)
        trials_vs_EMD_error = ax.get_optimization_trace(objective_optimum=EMD_error.fmin)
        print(trials)

        trials.to_csv('trials_' + str(num_layers) + 'layers.csv')
        save_plot(trials_vs_EMD, 'trials_vs_EMD_' + str(num_layers) + 'layers.pdf')
        save_plot(trials_vs_EMD_error, 'trials_vs_EMD_error_' + str(num_layers) + 'layers.pdf')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)