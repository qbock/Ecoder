import os
import boto3
import keras
import tarfile
import determined as det
import numpy as np
import argparse
from determined.keras import TFKerasTrial, TFKerasTrialContext, InputData, TFKerasExperimentalContext
from botocore.client import Config
from telescope import telescopeMSE8x8
from networks import arrange_dict
from utils.logger import _logger
from denseCNN import denseCNN
from train import load_data, normalize, unnormalize, split, evaluate_model

parser = argparse.ArgumentParser()
parser.add_argument('-o',"--odir", type=str, default='/comp_all', dest="odir",
                    help="output directory")
parser.add_argument('-i',"--inputFile", type=str, default='/tmp/data-rank0', dest="inputFile",
                    help="input TSG files")
parser.add_argument("--loss", type=str, default=None, dest="loss",
                    help="force loss function to use")
parser.add_argument("--quantize", action='store_true', default=False, dest="quantize",
                    help="quantize the model with qKeras. Default precision is 16,6 for all values.")
parser.add_argument("--epochs", type=int, default = 200, dest="epochs",
                    help="number of epochs to train")
parser.add_argument("--nELinks", type=int, default = 5, dest="nElinks",
                    help="n of active transceiver e-links eTX")

parser.add_argument("--skipPlot", action='store_true', default=False, dest="skipPlot",
                    help="skip the plotting step")
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

parser.add_argument("--maskPartials", action='store_true', default = False,dest="maskPartials",
                    help="mask partial modules")
parser.add_argument("--maskEnergies", action='store_true', default = False,dest="maskEnergies",
                    help="Mask energy fractions <= 0.05")
parser.add_argument("--saveEnergy", action='store_true', default = False,dest="saveEnergy",
                    help="save SimEnergy from input data")
parser.add_argument("--noHeader", action='store_true', default = True,dest="noHeader",
                    help="input data has no header")

parser.add_argument("--models", type=str, default="8x8_c8_S2_tele", dest="models",
                    help="models to run, if empty string run all")


def apply_constraints(context):

    # Look at hyperparameters for each CNN layers, if the number of filters in layer i is 0 then
    # subsequent layers and their hyperparameters are eliminated
    max_cnn_layers = context.get_hparam("max_cnn_layers")
    for i in range(1, max_cnn_layers+1):
        if context.get_hparam(f"filters{i}") == 0:
            for j in range(i+1, max_cnn_layers+1):
                if context.get_hparam(f"filters{j}") > 0:
                    raise det.InvalidHP(
                        "Can't produce subqequent CNN layers after CNN layer with 0 filters")

    # Do the same for Dense layers
    max_dense_layers = context.get_hparam("max_dense_layers")
    for i in range(1, max_dense_layers+1):
        if context.get_hparam(f"dense{i}") == 15:
            for j in range(i+1, max_dense_layers+1):
                if context.get_hparam(f"dense{j}") > 15:
                    raise det.InvalidHP(
                        "Can't produce subqequent Dense layers after Dense layer with 0 filters")

    # Constrain the factor by which the input dimensions are shrunk so that they that the decoder
    # doesn't upsample to a larger dimension than that of the dimension of the input
    reduction_factor = 1

    for i in range(1, max_cnn_layers+1):
        reduction_factor /= context.get_hparam(f"stride{i}")
        if context.get_hparam(f"maxpool{i}") == True:
            reduction_factor /= 2

        if reduction_factor * 8 <= 1:
            raise det.InvalidHP("Hyperparameters reduce size of the input too far")


class EMDCallback(keras.callbacks.Callback):
    def __init__(self, args=None, data_values=None, phys_values=None, model=None, m=None, interval=10):
        super(keras.callbacks.Callback, self).__init__()
        self.args = args
        self.interval = interval
        self.data_values = data_values
        self.model_dict = model
        self.m = m
        self.phys_values = phys_values

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            # normalize input charge data
            normdata,maxdata,sumdata = normalize(self.data_values.copy(),rescaleInputToMax=self.args.rescaleInputToMax,sumlog2=True)
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
            print(f'len phys_values: {len(self.phys_values)}')
            Nphys = round(len(self.phys_values)*0.2)
            print(f'Nphys = {Nphys}')
            phys_val_input = self.phys_values[:Nphys]
            # phys_val_input=phys_val_input

            shaped_data = self.m.prepInput(normdata)

            val_input, train_input, val_ind, train_ind = split(shaped_data)

            val_max = maxdata[val_ind]
            val_sum = sumdata[val_ind]

            # evaluate model
            input_Q, cnn_deQ, cnn_enQ = self.m.predict(val_input)

            input_calQ  = self.m.mapToCalQ(input_Q)   # shape = (N,48) in CALQ order
            output_calQ_fr = self.m.mapToCalQ(cnn_deQ)   # shape = (N,48) in CALQ order
            _logger.info('inputQ shape')
            print(input_Q.shape)
            _logger.info('inputcalQ shape')
            print(input_calQ.shape) 

            _logger.info('Restore normalization')
            input_Q_abs = np.array([input_Q[i]*(val_max[i] if self.args.rescaleInputToMax else val_sum[i]) for i in range(0,len(input_Q))]) * 35.   # restore abs input in CALQ unit
            input_calQ  = np.array([input_calQ[i]*(val_max[i] if self.args.rescaleInputToMax else val_sum[i]) for i in range(0,len(input_calQ)) ])  # shape = (N,48) in CALQ order
            output_calQ =  unnormalize(output_calQ_fr.copy(), val_max if self.args.rescaleOutputToMax else val_sum, rescaleOutputToMax=self.args.rescaleOutputToMax)

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

            perf_dict[self.model_dict['label']] , self.model_dict['summary_dict'] = evaluate_model(self.model_dict,charges,aux_arrs,eval_dict,self.args)

            EMD = self.model_dict['summary_dict']['EMD_ae']
            EMD_error = self.model_dict['summary_dict']['EMD_ae_err']


class ECONT(TFKerasTrial):
    def __init__(self, context: TFKerasTrialContext, nElinks = 5, loss = None):

        self.args = parser.parse_args()
        self.context = context 
        self.nElinks = nElinks
        self.loss = loss
        self.dataloc = ""
        self.x_train, self.y_train, self.x_test, self.y_test = None,None,None,None
        self.model, self.m = None,None

        apply_constraints(self.context)

        self.download_dataset()
        self.load_split_data()

    def keras_callbacks(self):
        callbacks = []

        estop = det.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=5,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,)
        callbacks.append(estop)
        print("Callback: EarlyStopping Enabled")
        emd = EMDCallback(self.args,self.data_values,self.phys_values,self.model, self.m)
        print("Callback: EMD Enabled")
        callbacks.append(emd)
        return callbacks

    def build_model(self):
        # Create lists of parameters and a name and label for the model
        filters, kernels, poolings, strides, paddings, units = [], [], [], [], [], []

        names = [''] * 5
        labels = [''] * 5
        max_cnn_layers = self.context.get_hparam("max_cnn_layers")
        if max_cnn_layers > 0:
            for i in range(4):
                names[i], labels[i] = '_', '_'

        # Populate the list of parameters
        for i in range(1, max_cnn_layers+1):
            num_filter = self.context.get_hparam(f"filters{i}")
            kernel_size = self.context.get_hparam(f"kernel{i}")
            pooling = self.context.get_hparam(f"maxpool{i}")
            stride = self.context.get_hparam(f"stride{i}")

            # Don't append parameters if the number of filters is zero
            if num_filter:
                kernels.append(kernel_size)
                filters.append(num_filter)
                strides.append((stride, stride))
                poolings.append(pooling)
                paddings.append('same')

                names[0] = names[0] + f"c{num_filter}"
                names[1] = names[1] + f"k{kernel_size}"
                names[2] = names[2] + f"p{pooling}"
                names[3] = names[3] + f"s{stride}"
                labels[0] = labels[0] + f"c[{num_filter}]"
                labels[1] = labels[1] + f"k[{kernel_size}]"
                labels[2] = labels[2] + f"p[{pooling}]"
                labels[3] = labels[3] + f"s[{stride}]"

        max_dense_layers = self.context.get_hparam("max_dense_layers")

        if max_dense_layers > 0:
            names[4], labels[4] = '_', '_'

        for i in range(1, max_dense_layers+1):
            num_units = self.context.get_hparam(f"dense{i}")

            # Don't append units if they are less than the encoding dim
            if num_units < 16:
                units.append(num_units)

                names[4] = names[4] + f"c{num_units}"
                labels[4] = labels[4] + f"c[{num_units}]"

        # If there is no kernel size added, define one to use in default CNN layers
        if not kernels:
            kernels = [(3,3)]

        # Create names, labels, and initalize model parameters
        name = f'8x8{names[0]}{names[1]}{names[2]}{names[3]}{names[4]}tele'
        label = f'8x8{labels[0]}{labels[1]}{labels[2]}{labels[3]}{labels[4]}(tele)'

        model = {
                'name':name,
                'label':label,
                'arr_key':'8x8',
                'isQK':False,
                'params':{
                    'shape':(8,8,1),
                    'loss':'telescopeMSE8x8',
                    'CNN_layer_nodes':filters,
                    'CNN_kernel_size':kernels,
                    'CNN_strides':strides,
                    'CNN_padding':paddings,
                    'CNN_pool':poolings,
                    'Dense_layer_nodes': units
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
        if(self.nElinks==2):
            nBits_encod  = {'total':  3, 'integer': 1,'keep_negative':0}
        elif(self.nElinks==3):
            nBits_encod  = {'total':  5, 'integer': 1,'keep_negative':0}
        elif(self.nElinks==4):
            nBits_encod  = {'total':  7, 'integer': 1,'keep_negative':0}
        elif(self.nElinks==5):
            nBits_encod  = {'total':  9, 'integer': 1,'keep_negative':0} # 0 to 2 range, 8 bit decimal
        else:
            _logger.warning('Must specify encoding bits for nElink %i'%self.nElinks)

        if not 'nBits_encod' in model['params'].keys():
            model['params'].update({'nBits_encod':nBits_encod})

        # Create denseCNN model from the model parameters.
        m = denseCNN()
        m.setpams(model['params'])
        # The determined model and optimizer wraping is done in the model initaliztion
        m.init(wrap=True, context=self.context)

        m_autoCNN, m_autoCNNen = m.get_models()
        self.m = m
        model['m_autoCNN'] = m_autoCNN
        model['m_autoCNNen'] = m_autoCNNen

        self.model = model

        return m_autoCNN

    def build_training_data_loader(self) -> InputData:
        print("TRAIN X shape: " + str(self.x_train.shape))
        print("TRAIN Y shape: " + str(self.y_train.shape))
        return self.x_train, self.y_train

    def build_validation_data_loader(self) -> InputData:
        print("TEST X shape: " + str(self.x_test.shape))
        print("TEST Y shape: " + str(self.y_test.shape))
        return self.x_test, self.y_test

    def download_data_from_s3(self):
        s3_bucket = self.context.get_data_config()["bucket"]
        download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        data_file = "data.tar.gz"
        #print("DEBUG - ENDPOINT: " + self.context.get_data_config()["endpoint_url"])
        #print("DEBUG - ACCESS KEY: " + self.context.get_data_config()["access_key"])
        ##print("DEBUG - SECRET KEY: " + self.context.get_data_config()["secret_key"])
        s3 = boto3.client('s3',
                            endpoint_url=self.context.get_data_config()["endpoint_url"],
                            aws_access_key_id=self.context.get_data_config()["access_key"],
                            aws_secret_access_key=self.context.get_data_config()["secret_key"],
                            config=Config(signature_version='s3v4'),
                            region_name='us-east-1')

        os.makedirs(download_directory, exist_ok=True)
        filepath = os.path.join(download_directory, data_file)
        if not os.path.exists(filepath):
            s3.download_file(s3_bucket, data_file, filepath)
        return download_directory

    def download_dataset(self):
        self.dataloc = self.download_data_from_s3()
        self.args.inputFile = self.dataloc
        filename = os.path.join(self.dataloc,"data.tar.gz")
        file = tarfile.open(filename)
        file.extractall(self.dataloc)
        file.close()
        os.remove(filename)
        #for root, dirs, files in os.walk(self.dataloc):
        #    for d in dirs:
        #        print(os.path.join(root, d))
        print("Training data downloaded & extracted successfully! Location in pod: " + self.dataloc)

    def load_split_data(self):
        data_values, phys_values = load_data(self.args)
        self.data_values = data_values
        self.phys_values = phys_values

        # measure TC occupancy
        occupancy_all = np.count_nonzero(data_values,axis=1) # measure non-zero TCs (should be all)
        occupancy_all_1MT = np.count_nonzero(data_values>35,axis=1) # measure TCs with charge > 35

        # normalize input charge data
        # rescaleInputToMax: normalizes charges to maximum charge in module
        # sumlog2 (default): normalizes charges to 2**floor(log2(sum of charge in module)) where floor is the largest scalar integer: i.e. normalizes to MSB of the sum of charges (MSB here is the most significant bit)
        # rescaleSum: normalizes charges to sum of charge in module
        normdata,maxdata,sumdata = normalize(data_values.copy(),rescaleInputToMax=self.args.rescaleInputToMax,sumlog2=True)
        maxdata = maxdata / 35. # normalize to units of transverse MIPs
        sumdata = sumdata / 35. # normalize to units of transverse MIPs

        # performance dictionary
        perf_dict={}

        #Putting back physics columns below once training is done
        print(f'len phys_values: {len(phys_values)}')
        Nphys = round(len(phys_values)*0.2)
        print(f'Nphys = {Nphys}')
        phys_val_input = phys_values[:Nphys]
        # phys_val_input=phys_val_input

        shape = (8,8,1)
        arrange_pairs = arrange_dict['8x8']

        if len(arrange_pairs['arrange'])>0:
            arrange = arrange_pairs['arrange']
            inputdata = normdata[:,arrange]
        else:
            inputdata = normdata
        if len(arrange_pairs['arrMask'])>0:
            arrMask = arrange_pairs['arrMask']
            inputdata[:,arrMask==0]=0  #zeros out repeated entries

        shaped_data = inputdata.reshape(len(inputdata),shape[0],shape[1],shape[2])

        test_input, train_input, val_ind, train_ind = split(shaped_data)
        self.x_train, self.y_train = train_input, train_input
        self.x_test, self.y_test = test_input, test_input
