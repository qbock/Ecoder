import numpy as np
import tensorflow.keras.optimizers as opt
from telescope import telescopeMSE443,telescopeMSE663,telescopeMSE8x8

edim = 16
arrange443 = np.array([0,16, 32,
                           1,17, 33,
                           2,18, 34,
                           3,19, 35,
                           4,20, 36,
                           5,21, 37,
                           6,22, 38,
                           7,23, 39,
                           8,24, 40,
                           9,25, 41,
                           10,26, 42,
                           11,27, 43,
                           12,28, 44,
                           13,29, 45,
                           14,30, 46,
                           15,31, 47])
arrange8x8 = np.array([
    28,29,30,31,0,4,8,12,
    24,25,26,27,1,5,9,13,
    20,21,22,23,2,6,10,14,
    16,17,18,19,3,7,11,15,
    47,43,39,35,35,34,33,32,
    46,42,38,34,39,38,37,36,
    45,41,37,33,43,42,41,40,
    44,40,36,32,47,46,45,44])
arrMask  =  np.array([
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,0,0,0,0,
    1,1,1,1,0,0,0,0,
    1,1,1,1,0,0,0,0,
    1,1,1,1,0,0,0,0,])
arrMask_full  =  np.array([
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,])
calQMask  =  np.array([
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,0,0,0,0,
    1,1,1,1,0,0,0,0,
    1,1,1,1,0,0,0,0,
    1,1,1,1,0,0,0,0,])

arrange8x8_2 = np.array([
    44,45,46,47,16,20,24,28,
    40,41,42,43,17,21,25,29,
    36,37,38,39,18,22,26,30,
    32,33,34,35,19,23,27,31,
    15,11, 7, 3, 3, 2, 1, 0,
    14,10, 6, 2, 7, 6, 5, 4,
    13,9,  5, 1,11,10, 9, 8,
    12,8,  4, 0,15,14,13,12])

arrMask_split  =  np.array([
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    1,1,1,0,0,1,1,1,
    1,1,0,0,0,0,1,1,
    1,0,0,0,0,0,0,1,])

arrange663 = np.array([  0,0,0,0,0,0,
                         0,12,13,14,15,32,
                         0,8,9,10,11,33,
                         0,4,5,6,7,34,
                         0,0,1,2,3,35,
                         0,31,27,23,19,0,
                         0,0,0,0,0,0,
                         0,28,29,30,31,0,
                         0,24,25,26,27,1,
                         0,20,21,22,23,2,
                         0,16,17,18,19,3,
                         0,47,43,39,35,0,
                         0,0,0,0,0,0,
                         0,44,45,46,47,16,
                         0,40,41,42,43,17,
                         0,36,37,38,39,18,
                         0,32,33,34,35,19,
                         0,15,11,7,3,0])

arrange663_mask = np.array([  0,0,0,0,0,0,
                              0,1,1,1,1,1,
                              0,1,1,1,1,1,
                              0,1,1,1,1,1,
                              0,1,1,1,1,1,
                              0,1,1,1,1,0,
                              0,0,0,0,0,0,
                              0,1,1,1,1,1,
                              0,1,1,1,1,1,
                              0,1,1,1,1,1,
                              0,1,1,1,1,1,
                              0,1,1,1,1,0,
                              0,0,0,0,0,0,
                              0,1,1,1,1,1,
                              0,1,1,1,1,1,
                              0,1,1,1,1,1,
                              0,1,1,1,1,1,
                              0,1,1,1,1,0])
arrange663_CalQmask = np.array([  0,0,0,0,0,0,
                                  0,1,1,1,1,0,
                                  0,1,1,1,1,0,
                                  0,1,1,1,1,0,
                                  0,1,1,1,1,0,
                                  0,0,0,0,0,0,
                                  0,0,0,0,0,0,
                                  0,1,1,1,1,0,
                                  0,1,1,1,1,0,
                                  0,1,1,1,1,0,
                                  0,1,1,1,1,0,
                                  0,0,0,0,0,0,
                                  0,0,0,0,0,0,
                                  0,1,1,1,1,0,
                                  0,1,1,1,1,0,
                                  0,1,1,1,1,0,
                                  0,1,1,1,1,0,
                                  0,0,0,0,0,0])

arrange663_tr = arrange663.reshape(3,36).transpose().flatten()
arrange663_mask_tr = arrange663_mask.reshape(3,36).transpose().flatten()
arrange663_CalQmask_tr = arrange663_CalQmask.reshape(3,36).transpose().flatten()

adam_slow = opt.Adam(learning_rate=0.0005)
adam_lr_e2 = opt.Adam(learning_rate=0.01)
adam_lr_e4 = opt.Adam(learning_rate=0.0001)
adam_lr_e5 = opt.Adam(learning_rate=0.00001)
SGD       = opt.SGD()
SGD_nesterov = opt.SGD(nesterov=True)
Adadelta  = opt.Adadelta()

defaults = {    'shape':(4,4,3),
                 'channels_first': False,
                 'arrange': arrange443,
                 'encoded_dim': edim,
                 'loss': 'telescopeMSE',
                 #'nBits_input'  : {'total': 10,                 'integer': 3,'keep_negative':1},
                 #'nBits_accum'  : {'total': 11,                 'integer': 3,'keep_negative':1},
                 #'nBits_weight' : {'total':  5,                 'integer': 1,'keep_negative':1},
                
                 ### Nov30 new default
                 # ap_fixed<6,1> model_default_t; // weights and biases only
                 # ap_ufixed<8,1> input_t; // inputs
                 # ap_fixed<9,1> layer2_t; // conv2d out
                 # ap_ufixed<8,1> layer3_t; // relu out
                 # ap_fixed<10,1> layer4_t; // dense out
                 # ap_ufixed<9,0> result_t; // relu out

                 #ap_fixed<6,1> model_default_t;
                 #ap_ufixed<8,1> input_t; // inputs
                 #ap_fixed<9,2> layer2_t; // conv2d out
                 #ap_ufixed<8,1> layer3_t; // relu out
                 #ap_fixed<10,2> layer4_t; // dense out
                 #ap_ufixed<9,1> result_t; // relu out
                 'nBits_weight' : {'total':  6,       'integer': 0,'keep_negative':1},  #-1 to 1 range, 5 bit decimal
                 'nBits_input'  : {'total':  8,       'integer': 1,'keep_negative':0},  # 0 to 2 range, 7 bit decimal
                 'nBits_accum'  : {'total':  8,       'integer': 1,'keep_negative':0},  # 0 to 2 range, 7 bit decimal
                 #'nBits_conv'   : {'total':  9,       'integer': 1,'keep_negative':1},  # -2 to 2 range, 7 bit decimal #nBits_conv = weights in conv
                 #'nBits_dense'  : {'total': 10,       'integer': 1,'keep_negative':1},  # -2 to 2 range, 8 bit decimal #nBits_dense = weights in dense
                 #'nBits_encod'  : {'total':  9,       'integer': 1,'keep_negative':0}   # 0 to 2 range, 8 bit decimal
}
models = [
    #{'name':'Sep1_CNN_keras_norm','label':'nom','pams':{
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_padding':['same'],
    #         'CNN_pool':[False],
    #    },
    #},

#    {'name':'Sep1_CNN_keras_v12','label':'dim12','pams':{
#             'CNN_layer_nodes':[8],
#             'CNN_kernel_size':[3],
#             'CNN_pool':[0],
#             'encoded_dim': 12,
#        },
#    },
#    {'name':'Sep1_CNN_keras_v13','label':'dim20','pams':{
#             'CNN_layer_nodes':[8],
#             'CNN_kernel_size':[3],
#             'CNN_pool':[0],
#             'encoded_dim': 20,
#        },
#    },

#    {'name':'Sep1_CNN_keras_v14','label':'den16','pams':{
#             'CNN_layer_nodes':[8],
#             'CNN_kernel_size':[3],
#             'CNN_pool':[0],
#             'Dense_layer_nodes':[16] ,
#        },
#    },
#
#    {'name':'Sep1_CNN_keras_v8','label':'k[5]','pams':{
#             'CNN_layer_nodes':[8],
#             'CNN_kernel_size':[5],
#             'CNN_pool':[0],
#        },
#    },
#    {'name':'Sep1_CNN_keras_v9','label':'c[12]','pams':{
#             'CNN_layer_nodes':[12],
#             'CNN_kernel_size':[3],
#             'CNN_pool':[0],
#        },
#    },
#    {'name':'Sep1_CNN_keras_v10','label':'pool','pams':{
#             'CNN_layer_nodes':[8],
#             'CNN_kernel_size':[3],
#             'CNN_pool':[1],
#        },
#    },



#    {'name':'Sep1_CNN_keras_v1','label':'c[8,8]','pams':{
#             'CNN_layer_nodes':[8,8],
#             'CNN_kernel_size':[3,3],
#             'CNN_pool':[False,False],
#        },
#    },
#    {'name':'Sep1_CNN_keras_v2','label':'c[8,8,8]','pams':{
#             'CNN_layer_nodes':[8,8,8],
#             'CNN_kernel_size':[3,3,3],
#             'CNN_pool':[False,False,False],
#        },
#    },
#    {'name':'Sep1_CNN_keras_v3','label':'c[4,4,4]','pams':{
#             'CNN_layer_nodes':[4,4,4],
#             'CNN_kernel_size':[3,3,3],
#             'CNN_pool':[False,False,False],
#        },
#    },
#    {'name':'Sep1_CNN_keras_v4','label':'c[8,4,2]','pams':{
#             'CNN_layer_nodes':[8,4,2],
#             'CNN_kernel_size':[3,3,3],
#             'CNN_pool':[False,False,False],
#        },
#    },
#   {'name':'Sep1_CNN_keras_v7','label':'c[8,4,4,4,2],','pams':{
#            'CNN_layer_nodes':[8,4,4,4,2],
#            'CNN_kernel_size':[3,3,3,3,3],
#            'CNN_pool':[0,0,0,0,0],
#       },
#   },

#    {'name':'Sep1_CNN_keras_v5','label':'c[8,4,2]_k[5,5,3]','pams':{
#             'CNN_layer_nodes':[8,4,2],
#             'CNN_kernel_size':[5,5,3],
#             'CNN_pool':[False,False,False],
#        },
#    },
#    {'name':'Sep1_CNN_keras_v6','label':'c[4,4,4]_k[5,5,3]','pams':{
#             'CNN_layer_nodes':[4,4,4],
#             'CNN_kernel_size':[5,5,3],
#             'CNN_pool':[False,False,False],
#        },
#    },
#    {'name':'Sep1_CNN_keras_v15','label':'c[4,4,4]_k[5,5,3]_den[16]','pams':{
#             'CNN_layer_nodes':[4,4,4],
#             'CNN_kernel_size':[5,5,3],
#             'CNN_pool':[False,False,False],
#             'Dense_layer_nodes':[16] ,
#        },
#    },
#    {'name':'Sep1_CNN_keras_v16','label':'c[20]_pool','pams':{
#             'CNN_layer_nodes':[20],
#             'CNN_kernel_size':[3],
#             'CNN_pool':[True],
#        },
#    },
    #{'name':'Sep9_CNN_keras_8x8_v1','label':'8x8_c[8]','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #    },
    #},
    #{'name':'Sep9_CNN_keras_8x8_v2','label':'8x8_c[2]','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[2],
    #         'CNN_kernel_size':[3],
    #    },
    #},
    #{'name':'Sep9_CNN_keras_8x8_v3','label':'8x8_c[8]_pool','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[True],
    #    },
    #},
    #{'name':'Sep9_CNN_keras_8x8_v5','label':'8x8_c[4]','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[4],
    #         'CNN_kernel_size':[3],
    #    },
    #},
    #{'name':'Sep9_CNN_keras_8x8_v6','label':'8x8_c[6]','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[6],
    #         'CNN_kernel_size':[3],
    #    },
    #},
    #{'name':'Sep9_CNN_keras_8x8_v4','label':'8x8_c[8]_v2','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8_2,'arrMask':arrMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[False],
    #    },
    #},

    #{'name':'Sep9_CNN_keras_8x8_v7','label':'8x8_c[8]_mask','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'calQMask':calQMask,'loss':'weightedMSE','maskConvOutput':arrMask,
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[False],
    #    },
    #},
    #{'name':'Sep9_CNN_keras_8x8_v8','label':'8x8_c[6]_mask','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'calQMask':calQMask,'loss':'weightedMSE','maskConvOutput':arrMask,
    #         'CNN_layer_nodes':[6],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[False],
    #    },
    #},
    #{'name':'Sep9_CNN_keras_8x8_v9','label':'8x8_c[4]_mask','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'calQMask':calQMask,'loss':'weightedMSE','maskConvOutput':arrMask,
    #         'CNN_layer_nodes':[4],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[False],
    #    },
    #},
    #{'name':'Sep9_CNN_keras_8x8_v7.2','label':'8x8_c[8]_pool','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'calQMask':calQMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[True],
    #    },
    #},
    #{'name':'Sep9_CNN_keras_8x8_v8.2','label':'8x8_c[6]_pool','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'calQMask':calQMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[6],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[True],
    #    },
    #},
    #{'name':'Sep9_CNN_keras_8x8_v9.2','label':'8x8_c[4]_pool','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'calQMask':calQMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[4],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[True],
    #    },
    #},

    #{'name':'Sep9_CNN_keras_8x8_v10','label':'8x8_c[8]_dup','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask_full,'calQMask':calQMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[False],
    #    },
    #},
    #{'name':'Sep9_CNN_keras_8x8_v11','label':'8x8_c[8]_dup_mask','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask_full,'calQMask':calQMask,'loss':'weightedMSE','maskConvOutput':calQMask,
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[False],
    #    },
    #},


    #{'name':'Sep21_CNN_keras_SepConv_v1','label':'SepConv','isDense2D':True,'pams':{
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[False],
    #    },
    #},
    #{'name':'Sep21_CNN_keras_SepConv_v2','label':'SepConv_NoShareFilter','isDense2D':True,'pams':{
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[False],
    #         'share_filters'    : False,
    #    },
    #},
    #{'name':'Sep21_CNN_keras_SepConv_v3','label':'SepConv_pool','isDense2D':True,'pams':{
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[True],
    #    },
    #},
    ## with extra decoder layer
    #{'name':'Oct8_SepConv_663_pool','label':'SepConv_663_pool_decoder','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663,'arrMask':arrange663_mask,'calQMask':arrange663_CalQmask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[True],
    #         'CNN_padding':['valid'],
    #    },
    #},
    #{'name':'Oct8_SepConv_663_pool_v3','label':'SepConv_663_pool_trv3','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663,'arrMask':arrange663_mask,'calQMask':arrange663_CalQmask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[True],
    #         'CNN_padding':['valid'],
    #    },
    #},


    #{'name':'Oct8_SepConv_663_pool','label':'SepConv_663_pool','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663_tr,'arrMask':arrange663_mask_tr,'calQMask':arrange663_CalQmask_tr,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[True],
    #         'CNN_padding':['valid'],
    #    },
    #},
    #{'name':'Oct11_SepConv_663_pool','label':'SepConv_663_pool','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663,'arrMask':arrange663_mask,'calQMask':arrange663_CalQmask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[True],
    #         'CNN_padding':['valid'],
    #    },
    #},

    #{'name':'Oct8_SepConv_663','label':'SepConv_663','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663_tr,'arrMask':arrange663_mask_tr,'calQMask':arrange663_CalQmask_tr,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[False],
    #         'CNN_padding':['valid'],
    #    },
    #},
    #{'name':'Oct8_SepConv_663_pool_noShareFilters','label':'SepConv_663_pool_noShareFilters','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663_tr,'arrMask':arrange663_mask_tr,'calQMask':arrange663_CalQmask_tr,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[True],
    #         'CNN_padding':['valid'],
    #         'share_filters'    : False,
    #    },
    #},
    #{'name':'Oct8_663','label':'Conv_663','isDense2D':False,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663_tr,'arrMask':arrange663_mask_tr,'calQMask':arrange663_CalQmask_tr,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[False],
    #         'CNN_padding':['valid'],
    #    },
    #},
    #{'name':'Oct8_SepConv_663_c4','label':'SepConv_663_c[4]','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663_tr,'arrMask':arrange663_mask_tr,'calQMask':arrange663_CalQmask_tr,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[4],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[False],
    #         'CNN_padding':['valid'],
    #    },
    #},
    #{'name':'Oct8_SepConv_663_c2','label':'SepConv_663_c[2]','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663_tr,'arrMask':arrange663_mask_tr,'calQMask':arrange663_CalQmask_tr,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[2],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[False],
    #         'CNN_padding':['valid'],
    #    },
    #},
    #{'name':'Oct8_SepConv_663_c4_pool','label':'SepConv_663_c[4]_pool','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663_tr,'arrMask':arrange663_mask_tr,'calQMask':arrange663_CalQmask_tr,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[4],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[True],
    #         'CNN_padding':['valid'],
    #    },
    #},
    #{'name':'Oct8_SepConv_663_c2_pool','label':'SepConv_663_c[2]_pool','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663_tr,'arrMask':arrange663_mask_tr,'calQMask':arrange663_CalQmask_tr,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[2],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[True],
    #         'CNN_padding':['valid'],
    #    },
    #},

    #{'name':'Oct8_SepConv_663_c8_k5_vpad','label':'SepConv_663_c8_k5_vpad','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663_tr,'arrMask':arrange663_mask_tr,'calQMask':arrange663_CalQmask_tr,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[5],
    #         'CNN_pool':[False],
    #         'CNN_padding':['valid'],
    #    },
    #},

    #{'name':'Oct8_SepConv663_c10_k5_vpad','label':'SepConv_663_c10_k5_vpad','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663_tr,'arrMask':arrange663_mask_tr,'calQMask':arrange663_CalQmask_tr,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[10],
    #         'CNN_kernel_size':[5],
    #         'CNN_pool':[False],
    #         'CNN_padding':['valid'],
    #    },
    #},

    #######wrong arrange
    #{'name':'Oct11_SepConv_663','label':'SepConv_663','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663,'arrMask':arrange663_mask,'calQMask':arrange663_CalQmask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[False],
    #         'CNN_padding':['valid'],
    #    },
    #},
    #{'name':'Oct11_SepConv_663_pool_noShareFilters','label':'SepConv_663_pool_noShareFilters','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663,'arrMask':arrange663_mask,'calQMask':arrange663_CalQmask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[True],
    #         'CNN_padding':['valid'],
    #         'share_filters'    : False,
    #    },
    #},
    #{'name':'Oct11_663','label':'Conv_663','isDense2D':False,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663,'arrMask':arrange663_mask,'calQMask':arrange663_CalQmask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[False],
    #         'CNN_padding':['valid'],
    #    },
    #},
    #{'name':'Oct11_SepConv_663_c4','label':'SepConv_663_c[4]','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663,'arrMask':arrange663_mask,'calQMask':arrange663_CalQmask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[4],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[False],
    #         'CNN_padding':['valid'],
    #    },
    #},
    #{'name':'Oct11_SepConv_663_c2','label':'SepConv_663_c[2]','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663,'arrMask':arrange663_mask,'calQMask':arrange663_CalQmask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[2],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[False],
    #         'CNN_padding':['valid'],
    #    },
    #},
    #{'name':'Oct11_SepConv_663_c4_pool','label':'SepConv_663_c[4]_pool','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663,'arrMask':arrange663_mask,'calQMask':arrange663_CalQmask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[4],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[True],
    #         'CNN_padding':['valid'],
    #    },
    #},
    #{'name':'Oct11_SepConv_663_c2_pool','label':'SepConv_663_c[2]_pool','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663,'arrMask':arrange663_mask,'calQMask':arrange663_CalQmask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[2],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[True],
    #         'CNN_padding':['valid'],
    #    },
    #},

    #{'name':'Oct11_SepConv_663_c8_k5_vpad','label':'SepConv_663_c8_k5_vpad','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663,'arrMask':arrange663_mask,'calQMask':arrange663_CalQmask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[5],
    #         'CNN_pool':[False],
    #         'CNN_padding':['valid'],
    #    },
    #},

    #{'name':'Oct11_SepConv663_c10_k5_vpad','label':'SepConv_663_c10_k5_vpad','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663,'arrMask':arrange663_mask,'calQMask':arrange663_CalQmask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[10],
    #         'CNN_kernel_size':[5],
    #         'CNN_pool':[False],
    #         'CNN_padding':['valid'],
    #    },
    #},

    #{'name':'Oct12_CNN_keras_norm_AdamSlow','label':'nom_AdamSlow','pams':{
    #         'CNN_layer_nodes':[8],             'CNN_kernel_size':[3],
    #         'CNN_padding':['same'],             'CNN_pool':[False],
    #         'optimizer':adam_slow,
    #    },
    #},
    #{'name':'Oct12_CNN_keras_norm_adam_lr_e2','label':'nom_adam_lr_e2','pams':{
    #         'CNN_layer_nodes':[8],             'CNN_kernel_size':[3],
    #         'CNN_padding':['same'],             'CNN_pool':[False],
    #         'optimizer':adam_lr_e2,
    #    },
    #},
    #{'name':'Oct12_CNN_keras_norm_adam_lr_e4','label':'nom_adam_lr_e4','pams':{
    #         'CNN_layer_nodes':[8],             'CNN_kernel_size':[3],
    #         'CNN_padding':['same'],             'CNN_pool':[False],
    #         'optimizer':adam_lr_e4,
    #    },
    #},
    #{'name':'Oct12_CNN_keras_norm_adam_lr_e5','label':'nom_adam_lr_e5','pams':{
    #         'CNN_layer_nodes':[8],             'CNN_kernel_size':[3],
    #         'CNN_padding':['same'],             'CNN_pool':[False],
    #         'optimizer':adam_lr_e5,
    #    },
    #},

    #{'name':'Oct12_CNN_keras_norm_SGD','label':'nom_SGD','pams':{
    #         'CNN_layer_nodes':[8],             'CNN_kernel_size':[3],
    #         'CNN_padding':['same'],             'CNN_pool':[False],
    #         'optimizer':SGD,
    #    },
    #},
    #{'name':'Oct12_CNN_keras_norm_SGD_nesterov','label':'nom_SGD_nesterov','pams':{
    #         'CNN_layer_nodes':[8],             'CNN_kernel_size':[3],
    #         'CNN_padding':['same'],             'CNN_pool':[False],
    #         'optimizer':SGD_nesterov,
    #    },
    #},
    #{'name':'Oct12_CNN_keras_norm_Adadelta','label':'nom_Adadelta','pams':{
    #         'CNN_layer_nodes':[8],             'CNN_kernel_size':[3],
    #         'CNN_padding':['same'],             'CNN_pool':[False],
    #         'optimizer':Adadelta,
    #    },
    #},
    #{'name':'Oct12_CNN_keras_norm_adamRDOP_p5','label':'nom_AdamRDOP','pams':{
    #         'CNN_layer_nodes':[8],             'CNN_kernel_size':[3],
    #         'CNN_padding':['same'],             'CNN_pool':[False],
    #    },
    #},


    #{'name':'Oct12_SepConv_663_AdamSlow','label':'SepConv_663_AdamSlow','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663_tr,'arrMask':arrange663_mask_tr,'calQMask':arrange663_CalQmask_tr,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],             'CNN_kernel_size':[3],
    #         'CNN_pool':[False],             'CNN_padding':['valid'],
    #        'optimizer':adam_slow,
    #    },
    #},

    #{'name':'Oct12_SepConv_663_RDOP_p5','label':'SepConv_663_RDOP','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663_tr,'arrMask':arrange663_mask_tr,'calQMask':arrange663_CalQmask_tr,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],             'CNN_kernel_size':[3],
    #         'CNN_pool':[False],             'CNN_padding':['valid'],
    #    },
    #},


    #{'name':'Oct12_SepConv_663_SGD','label':'SepConv_663_SGD','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663_tr,'arrMask':arrange663_mask_tr,'calQMask':arrange663_CalQmask_tr,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],             'CNN_kernel_size':[3],
    #         'CNN_pool':[False],             'CNN_padding':['valid'],
    #        'optimizer':SGD,
    #    },
    #},
    #{'name':'Oct12_SepConv_663_SGD_nesterov','label':'SepConv_663_SGD_nesterov','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663_tr,'arrMask':arrange663_mask_tr,'calQMask':arrange663_CalQmask_tr,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],             'CNN_kernel_size':[3],
    #         'CNN_pool':[False],             'CNN_padding':['valid'],
    #        'optimizer':SGD_nesterov,
    #    },
    #},
    #{'name':'Oct12_SepConv_663_Adadelta','label':'SepConv_663_Adadelta','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663_tr,'arrMask':arrange663_mask_tr,'calQMask':arrange663_CalQmask_tr,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],             'CNN_kernel_size':[3],
    #         'CNN_pool':[False],             'CNN_padding':['valid'],
    #        'optimizer':Adadelta,
    #    },
    #},

    #{'name':'Oct12_SepConv_663_adam_lr_e2','label':'SepConv_663_adam_lr_e2','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663_tr,'arrMask':arrange663_mask_tr,'calQMask':arrange663_CalQmask_tr,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],             'CNN_kernel_size':[3],
    #         'CNN_pool':[False],             'CNN_padding':['valid'],
    #        'optimizer':adam_lr_e2,
    #    },
    #},
    #{'name':'Oct12_SepConv_663_adam_lr_e4','label':'SepConv_663_adam_lr_e4','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663_tr,'arrMask':arrange663_mask_tr,'calQMask':arrange663_CalQmask_tr,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],             'CNN_kernel_size':[3],
    #         'CNN_pool':[False],             'CNN_padding':['valid'],
    #        'optimizer':adam_lr_e4,
    #    },
    #},
    #{'name':'Oct12_SepConv_663_adam_lr_e5','label':'SepConv_663_adam_lr_e5','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663_tr,'arrMask':arrange663_mask_tr,'calQMask':arrange663_CalQmask_tr,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],             'CNN_kernel_size':[3],
    #         'CNN_pool':[False],             'CNN_padding':['valid'],
    #        'optimizer':adam_lr_e5,
    #    },
    #},

    #{'name':'Oct30_8x8_k5','label':'8x8_k5','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'calQMask':calQMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[5],
    #    },
    #},
    #{'name':'Oct30_8x8_k5_pool','label':'8x8_k5_pool','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'calQMask':calQMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[5],
    #         'CNN_pool':[True],
    #    },
    #},
    #{'name':'Oct30_8x8_c6_k5_pool','label':'8x8_c6_k5_pool','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'calQMask':calQMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[6],
    #         'CNN_kernel_size':[5],
    #         'CNN_pool':[True],
    #    },
    #},
    #{'name':'Oct30_8x8_c4_k5_pool','label':'8x8_c4_k5_pool','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'calQMask':calQMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[4],
    #         'CNN_kernel_size':[5],
    #         'CNN_pool':[True],
    #    },
    #},

    #{'name':'Oct12_CNN_keras_8x8_pool_RDOP_p5','label':'8x8_c[8]_pool','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'calQMask':calQMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[True],
    #    },
    #},
    #{'name':'Oct12_CNN_keras_8x8_c8_pool_Adamp5','label':'8x8_c[8]_pool','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'calQMask':calQMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[True],
    #         'optimizer':adam_slow,
    #    },
    #},
    #{'name':'Oct12_CNN_keras_8x8_c6_pool_Adamp5','label':'8x8_c[6]_pool','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'calQMask':calQMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[6],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[True],
    #         'optimizer':adam_slow,
    #    },
    #},
    #{'name':'Oct12_CNN_keras_8x8_c4_pool_Adamp5','label':'8x8_c[4]_pool','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'calQMask':calQMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[4],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[True],
    #         'optimizer':adam_slow,
    #    },
    #},
    #{'name':'Nov10_8x8_c6_S2','label':'8x8_c[6]_S2','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'calQMask':calQMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[6],
    #         'CNN_kernel_size':[3],
    #         'CNN_strides':[(2,2)],
    #    },
    #},
    #{'name':'Nov10_8x8_c8_S2','label':'8x8_c[8]_S2','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'calQMask':calQMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_strides':[(2,2)],
    #    },
    #},
    #{'name':'Nov10_8x8_c10_S2','label':'8x8_c[10]_S2','pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'calQMask':calQMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[10],
    #         'CNN_kernel_size':[3],
    #         'CNN_strides':[(2,2)],
    #    },
    #},

    #{'name':'Nov10_8x8_c8_S2_qK','label':'8x8_c[8]_S2','isQK':True,'pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'calQMask':calQMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_strides':[(2,2)],
    #    },
    #},

    #{'name':'Nov10_8x8_c8_S2_qK_1b7','label':'8x8_c[8]_S2','isQK':True,'pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'calQMask':calQMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_strides':[(2,2)],
    #         'nBits_encod':{'total':7, 'integer':1, 'keep_negative':0}
    #    },
    #},
    #{'name':'Nov10_8x8_c8_S2_qK_1b5','label':'8x8_c[8]_S2','isQK':True,'pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'calQMask':calQMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_strides':[(2,2)],
    #         'nBits_encod':{'total':5, 'integer':1, 'keep_negative':0}
    #    },
    #},
    #{'name':'Nov10_8x8_c8_S2_qK_1b3','label':'8x8_c[8]_S2','isQK':True,'pams':{
    #         'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'calQMask':calQMask,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_strides':[(2,2)],
    #         'nBits_encod':{'total':3, 'integer':1, 'keep_negative':0}
    #    },
    #},

    {'name':'Nov10_8x8_c8_S2_qK_RTL','label':'8x8_c[8]_S2','isQK':True,'pams':{
             'shape':(8,8,1),'arrange': arrange8x8,'arrMask':arrMask,'calQMask':calQMask,'loss':'weightedMSE',
             'CNN_layer_nodes':[8],
             'CNN_kernel_size':[3],
             'CNN_strides':[(2,2)],
             'nBits_encod'  : {'total':  9,       'integer': 1,'keep_negative':0}   # 0 to 2 range, 8 bit decimal
        },
    },








    #{'name':'Oct12_SepConv_663_c2_Adamp5','label':'SepConv_663_c[2]','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663_tr,'arrMask':arrange663_mask_tr,'calQMask':arrange663_CalQmask_tr,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[2],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[False],
    #         'CNN_padding':['valid'],
    #         'optimizer':adam_slow,
    #    },
    #},
    #{'name':'Oct12_SepConv_663_c4_pool_Adamp5','label':'SepConv_663_c[4]_pool','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663_tr,'arrMask':arrange663_mask_tr,'calQMask':arrange663_CalQmask_tr,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[4],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[True],
    #         'CNN_padding':['valid'],
    #         'optimizer':adam_slow,
    #    },
    #},
    #{'name':'Oct12_SepConv_663_c2_pool_Adamp5','label':'SepConv_663_c[2]_pool','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663_tr,'arrMask':arrange663_mask_tr,'calQMask':arrange663_CalQmask_tr,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[2],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[True],
    #         'CNN_padding':['valid'],
    #         'optimizer':adam_slow,
    #    },
    #},
    #{'name':'Oct12_SepConv_443_pool_Adamp5','label':'SepConv_443_pool','isDense2D':True,'pams':{
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[True],
    #         'CNN_padding':['same'],
    #         'optimizer':adam_slow,
    #    },
    #},
    #{'name':'Oct12_SepConv_663_pool_Adamp5','label':'SepConv_663_pool','isDense2D':True,'pams':{
    #         'shape':(6,6,3),'arrange':arrange663_tr,'arrMask':arrange663_mask_tr,'calQMask':arrange663_CalQmask_tr,'loss':'weightedMSE',
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_pool':[True],
    #         'CNN_padding':['valid'],
    #         'optimizer':adam_slow,
    #    },
    #},

    #{'name':'Nov9_QK_norm','label':'nom','isQK':True,'pams':{
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_padding':['same'],
    #         'CNN_pool':[False],
    #    },
    #},
    #{'name':'Nov9_QK_norm_9bI4','label':'nom','isQK':True,'pams':{
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_padding':['same'],
    #         'CNN_pool':[False],
    #    },
    #}
    #{'name':'Nov20_QK_norm','label':'nom','isQK':True,'pams':{
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_padding':['same'],
    #         'CNN_pool':[False],
    #    },
    #},

    #{'name':'Nov9_QK_norm_ws','label':'nom','isQK':True,
    #    'ws':'/uscms/home/kkwok/work/HGC/CMSSW_11_1_0_pre6/src/Ecoder/V11/signal/nElinks_5/Sep1_CNN_keras_norm/Sep1_CNN_keras_norm.hdf5',
    #    'pams':{
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_padding':['same'],
    #         'CNN_pool':[False],
    #    },
    ##},
    #{'name':'Nov16_CNN_keras_nom_tele','label':'nom_tele','pams':{
    #         'CNN_layer_nodes':[8],
    #         'CNN_kernel_size':[3],
    #         'CNN_padding':['same'],
    #         'CNN_pool':[False],
    #         'loss':telescopeMSE443,
    #    },
    #},





]
for m in models:
   if not 'isDense2D' in m.keys(): m.update({'isDense2D':False})
   if not 'isQK' in m.keys(): m.update({'isQK':False})
   if not 'ws' in m.keys(): m.update({'ws':''})
   for p,v in defaults.items():
        if not p in m['pams'].keys(): 
            m['pams'].update({p:v})
