import os
import boto3
import keras
import tarfile
import determined as det
from determined.keras import TFKerasTrial, TFKerasTrialContext, InputData
from botocore.client import Config
from telescope import telescopeMSE8x8
from networks import arrange_dict
from utils.logger import _logger
from denseCNN import denseCNN

def apply_constraints(context):

    # Look at hyperparameters for each CNN layers, if the number of filters in layer i is 0 then
    # subsequent layers and their hyperparameters are eliminated
    max_cnn_layers = context.get_hparam("max_cnn_layers")
    for i in range(1, max_cnn_layers+1):
        if context.get_hparam(f"filters{i}") == 0:
            for j in range(i+1, max_cnn_layers):
                if context.get_hparam(f"filters{j}") > 0:
                    det.InvalidHP(
                        "Can't produce subqequent CNN layers after CNN layer with 0 filters")

    # Do the same for Dense layers
    max_dense_layers = context.get_hparam("max_dense_layers")
    for i in range(1, max_dense_layers+1):
        if context.get_hparam(f"dense{i}") == 15:
            for j in range(i+1, max_dense_layers):
                if context.get_hparam(f"dense{j}") > 15:
                    det.InvalidHP(
                        "Can't produce subqequent Dense layers after Dense layer with 0 filters")

    # Constrain the factor by which the input dimensions are shrunk so that they that the decoder
    # doesn't upsample to a larger dimension than that of the dimension of the input
    reduction_factor = 1

    for i in range(1, max_cnn_layers+1):
        reduction_factor /= context.get_hparam(f"stride{i}")
        if context.get_hparam(f"pooling{i}") == "include":
            reduction_factor /= 2

        if reduction_factor * 8 >= 1:
            det.InvalidHP("Hyperparameters reduce size of the input too far")

class ECONT(TFKerasTrial):
    def __init__(self, context: TFKerasTrialContext, nElinks = 5, loss = None):
        self.context = context
        self.nElinks = nElinks
        self.loss = loss
        self.x_train, self.y_train, self.x_test, self.y_test = None,None,None,None

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
            kernal_size = self.context.get_hparam(f"kernel{i}")
            pooling = self.context.get_hparam(f"pooling{i}")
            stride = self.context.get_hparam(f"stride{i}")

            kernels.append(kernal_size)
            filters.append(num_filter)
            strides.append((stride, stride))
            poolings.append(pooling)
            paddings.append('same')

            names[0] = names[0] + f"c{num_filter}"
            names[1] = names[1] + f"k{kernal_size}"
            names[2] = names[2] + f"p{pooling}"
            names[3] = names[3] + f"s{stride}"
            labels[0] = labels[0] + f"c[{num_filter}]"
            labels[1] = labels[1] + f"k[{kernal_size}]"
            labels[2] = labels[2] + f"p[{pooling}]"
            labels[3] = labels[3] + f"s[{stride}]"

        max_dense_layers = self.context.get_hparam("max_dense_layers")

        if max_dense_layers > 0:
            names[4], labels[4] = '_', '_'

        for i in range(1, max_dense_layers+1):
            num_units = self.context.get_hparam(f"dense{i}")

            units.append(num_units)

            names[4] = names[4] + f"c{num_units}"
            labels[4] = labels[4] + f"c[{num_units}]"

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
                    'loss':telescopeMSE8x8,
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

        if self.loss:
            model['params']['loss'] = self.loss

        # Create denseCNN model from the model parameters.
        models = denseCNN()
        models.setpams(model['params'])
        models.init()

        # Wrap the model.
        models = self.context.wrap_model(model)

        # Compile model
        models.compileModels()

        return model

    def build_training_data_loader(self) -> InputData:
        return self.x_train, self.y_train

    def build_validation_data_loader(self) -> InputData:
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
        file = tarfile.open(os.path.join(self.dataloc,"data.tar.gz"))
        file.extractall(self.dataloc)
        file.close()
        #for root, dirs, files in os.walk(self.dataloc):
        #    for d in dirs:
        #        print(os.path.join(root, d))
        print("Training data downloaded & extracted successfully! Location in pod: " + self.dataloc)

    def load_split_data(self):
        pass