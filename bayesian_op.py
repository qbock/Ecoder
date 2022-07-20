import numpy as np
import pandas as pd
import os

from denseCNN import denseCNN
from utils.logger import _logger
from telescope import telescopeMSE8x8
from datetime import datetime

from argparse import ArgumentParser
from ax.service.ax_client import AxClient
from train import ( normalize,
                    unnormalize,
                    load_data,
                    split,
                    train,
                    evaluate_model)

from ax import (
    ParameterType,
    RangeParameter,
    ChoiceParameter
)


parser = ArgumentParser()
parser.add_argument('-o',"--odir", type=str, default='/Ax_optimzation', dest="odir",
                    help="output directory")
parser.add_argument('-i',"--inputFile", type=str, default='nElinks_5/', dest="inputFile",
            help="input TSG files")
parser.add_argument("--quantize", action='store_true', default=False, dest="quantize",
            help="quantize the model with qKeras. Default precision is 16,6 for all values.")
parser.add_argument("--epochs", type=int, default = 200, dest="epochs",
            help="number of epochs to train")
parser.add_argument("--nELinks", type=int, default = 5, dest="nElinks",
            help="n of active transceiver e-links eTX")
parser.add_argument('--boEpochs', type=int, default = 30, dest="bayesian_op_epochs",
            help="number of epochs to train within Bayesian Optimization")
parser.add_argument('--maxLayers', type=int, default = 3, dest="max_CNN_layers",
            help="number of epochs to train within Bayesian Optimization")

def save_plot(plot):
    data = plot[0]['data']
    lay = plot[0]['layout']

    import plotly.graph_objects as go
    fig = {
        "data": data,
        "layout": lay,
    }
    go.Figure(fig).write_image("test.pdf")

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
                    n_epochs = args.epochs,
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

    return model['summary_dict']['EMD_ae'], model['summary_dict']['EMD_ae_err']

def build_model(args, parameterization, bo_epoch):

    filters, kernels, poolings, strides, paddings = [], [], [], [], []

    for i in range(3):
        filter = parameterization.get(f"filters_{i+1}")
        kernel = parameterization.get(f"kernal_{i+1}")
        pooling = parameterization.get(f"pooling_{i+1}")
        stride = parameterization.get(f"stride_{i+1}")
        if filters:
            filters.append(filter)
            kernels.append(kernel)
            poolings.append(pooling)
            strides.append(stride)
            paddings.append('same')

    name = f'8x8_c{*filters,}]_k[{*kernels,}]_p[{*poolings,}_s{*strides,}tele'
    label = f'8x8_c[{*filters,}]_k[{*kernels,}]_p[{*poolings,}]_s[{*strides,}](tele)'

    model = {
            'name':name,
            'label':label,
            'arr_key':'8x8',
            'isQK':True,
            'params':{
                'shape':(8,8,1),
                'loss':telescopeMSE8x8,
                'CNN_layer_nodes':filters,
                'CNN_kernel_size':kernels,
                'CNN_strides':strides,
                'CNN_padding':paddings,
                'CNN_pool':pooling
             }
    }

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

    model['params'].update({'nBits_encod':nBits_encod})

    if args.loss:
        model['params']['loss'] = args.loss

    return model

def bo_evaluation(args, parameterization, bo_epoch):

    model = build_model(args, parameterization, bo_epoch)
    EMD, EMD_error = evaluation(model, args)
    return {"EMD": EMD, "EMD_error": EMD_error}

def main(args):

    ax = AxClient()

    """ Set parameters and value of those parameters to be used in the optimization. There is a
    parameter (knob) for number of filters, kernal size, whether or not to use pooling, and the
    stride for each CNN layer.

    Constraints to limit the search of parameters only only when that layer exists

    Ex. filters_{i+1} * filters_{i+2} >= filters_{i+2} limits the number of filters for the {i+2}th
    layer to 0 if the previous {i+1}th layers also has 0 filters (a layer having 0 filters is the
    same as that layer not existing).
    """

    # List of parameters and parameter constraints
    ax_parameters, ax_parameter_constraints = [], []

    for i in range(args.max_CNN_layers - 1):
        ax_parameter_constraints.append(f"filters_{i+1} * filters_{i+2} >= filters_{i+2}")

    for i in range(args.max_CNN_layers):
        ax_parameters.append(RangeParameter(name=f"filters_{i+1}", parameter_type=ParameterType.INT, lower=0, upper=64))
        ax_parameters.append(ChoiceParameter(name=f"kernel_{i+1}", parameter_type=ParameterType.INT, values=[5,3,1], sort_values=True))
        ax_parameters.append(ChoiceParameter(name=f"pooling_{i+1}", parameter_type=ParameterType.INT, values=[0,1], sort_values=True))
        ax_parameters.append(RangeParameter(name=f"stride_{i+1}", parameter_type=ParameterType.INT, lower=1, upper=5))

        ax_parameter_constraints.append(f"filters_{i+1} * kernel_{i+1} >= kernel_{i+1}")
        ax_parameter_constraints.append(f"filters_{i+1} * pooling_{i+1} >= pooling_{i+1}")
        ax_parameter_constraints.append(f"filters_{i+1} * stride_{i+1} >= stride_{i+1}")

    print('\n\n')
    print(ax_parameters)
    print('\n\n')
    print(ax_parameter_constraints)
    print('\n\n')


    # Create Ax experiment
    ax.create_experiment(
        name="ECON_T Bayesian Optimization",
        parameters=ax_parameters,
        objective_name="EMD",
        minimize=True,
        parameter_constraints = ax_parameter_constraints
    )

    # Optimization loop
    for i in range(args.bayesian_op_epochs):
        parameterization, idx = ax.get_next_trial()
        ax.complete_trial(trial_index=idx, raw_data=evaluation(args, parameterization, i))

    # Print results and save them to a file
    trials = ax.get_trails_data_fram().sort_values('trail_index')
    trials_vs_EMD = ax.get_optimization_trace(objective_optimum=EMD.fmin)
    trials_vs_EMD_error = ax.get_optimization_trace(objective_optimum=EMD_error.fmin)
    print(trials)

    now = datetime()
    time = now.strftime("%d-%m-%Y_%H-%M-%S")
    original_dir = os.curdir
    new_dir = os.path.join(args.odir, "Experiment_" + time)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    os.chdir(new_dir)


    trials.to_csv('trials.csv')
    save_plot(trials_vs_EMD)
    save_plot(trials_vs_EMD_error)

    os.chdir(original_dir)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)