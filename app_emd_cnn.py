import numpy as np
import mplhep as hep
import pickle

import os
import pandas as pd

import itertools
import sys
sys.path.insert(0, "../")

import matplotlib
import matplotlib.pyplot as plt

from utils.wafer import plot_wafer as plotWafer

from utils.metrics import emd
from utils.metrics import hexMetric

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, BatchNormalization, Activation, Average, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2
        
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class app_EMD_CNN:
    
    X1_train=[]
    X2_train=[]
    
    def ittrain(real_calQ_data,num_filt, kernel_size, num_dens_neurons, num_dens_layers, num_conv_2d, num_epochs):
        
        current_directory=os.getcwd()
        
        #Arranging the hexagon
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

        def load_ae_data(inputFile):
            data=pd.read_csv(inputFile, dtype=np.float64)
            data_values=data.values

            return data_values
        
        #Take data from previous Autoencoder Training
        csv_directory=os.path.join(current_directory,'test_ae','8x8_c8_S2_tele')
        input_loc=os.path.join(csv_directory,'verify_input_calQ.csv')

        q_input_data=load_ae_data(input_loc)

        output_loc=os.path.join(csv_directory,'verify_decoded_calQ.csv')

        q_output_data=load_ae_data(output_loc)

        print(q_input_data.shape)
        print(q_output_data.shape)
        
        #Getting data from real input
        
        q_real_data=real_calQ_data

        print(q_real_data.shape)
        
        #Formatting data for [input,AE(input)]

        calQ1     = q_input_data
        sumQ1     = calQ1.sum(axis=1)
        calQ1     = calQ1[sumQ1>0]
        sumQ1     = sumQ1[sumQ1>0]

        calQ1_443 = (calQ1/np.expand_dims(sumQ1,-1))[:,arrange443].reshape(-1,4,4,3)

        calQ2     = q_output_data
        sumQ2     = calQ2.sum(axis=1)
        calQ2     = calQ2[sumQ2>0]
        sumQ2     = sumQ2[sumQ2>0]

        calQ2_443 = (calQ2/np.expand_dims(sumQ2,-1))[:,arrange443].reshape(-1,4,4,3)

        """
        #Generate True EMD for testing_data
        test_index=min(len(calQ1),len(calQ2))
        test_indices = range(0,test_index)

        ae_emd_values = np.array([emd(calQ1[i],calQ2[j]) for i, j in zip(test_indices,test_indices)])
        """
        
        #Formatting data for real pairs
        calQ     = q_real_data
        sumQ     = calQ.sum(axis=1)
        calQ     = calQ[sumQ>0]
        sumQ     = sumQ[sumQ>0]

        calQ_443 = (calQ/np.expand_dims(sumQ,-1))[:,arrange443].reshape(-1,4,4,3)
        
        """

        test_indices = range(0,len(calQ))

        idx1_test = np.array([i for i,j in itertools.product(test_indices,test_indices)])
        idx2_test = np.array([j for i,j in itertools.product(test_indices,test_indices)])

        pair_emd_values = np.array([emd(calQ[i],calQ[j]) for i, j in zip(idx1_test, idx2_test)])
                             
        print(ae_emd_values.shape)
        print(pair_emd_values.shape)
        
        union_emd_values=np.concatenate((ae_emd_values,pair_emd_values))
        print(union_emd_values.shape)
        
        
                        
        
        fig=plt.figure()
        fig=plt.hist(union_emd_values, alpha=1, bins=np.arange(0, 7.5,0.01), label='TrueEMD')
        fig=plt.xlabel('EMD [GeV]')
        fig=plt.ylabel('Samples')
        fig=plt.legend()
        plt.savefig(os.path.join(current_directory,img_directory,'UnionEMD.png'))
        plt.show()
        """                     
        #Splitting into training and validation data for AE data

        ae_train_indices = range(0, int(0.6*len(calQ1)))
        ae_val_indices = range(int(0.6*len(calQ1)), len(calQ1))

        ae_train_index=int(0.6*len(calQ1))

        ae_idx1_train = np.array([i for i in ae_train_indices])
        ae_idx2_train = np.array([j for j in ae_train_indices])

        ae_X1 = calQ1_443
        ae_X2 = calQ2_443

        ae_X1_train = ae_X1[0:ae_train_index]
        ae_X2_train = ae_X2[0:ae_train_index]

        ae_y_train = np.array([emd(calQ1[i],calQ2[j]) for i, j in zip(ae_train_indices,ae_train_indices)])

        ae_X1_val = ae_X1[ae_train_index:]
        ae_X2_val = ae_X2[ae_train_index:]
        ae_y_val = np.array([emd(calQ1[i],calQ2[j]) for i, j in zip(ae_val_indices, ae_val_indices)])

        print(ae_X1_train.shape)
        print(ae_X2_train.shape)
        print(ae_y_train.shape)

        print(ae_X1_val.shape)
        print(ae_X2_val.shape)
        print(ae_y_val.shape)                      
        
        #Splitting into training and validation for real pairs data
        pair_train_indices = range(0, int(0.6*len(calQ)))
        pair_val_indices = range(int(0.6*len(calQ)), len(calQ))

        pair_idx1_train = np.array([i for i,j in itertools.product(pair_train_indices,pair_train_indices)])
        pair_idx2_train = np.array([j for i,j in itertools.product(pair_train_indices,pair_train_indices)])

        pair_X = calQ_443

        pair_X1_train = pair_X[pair_idx1_train]
        pair_X2_train = pair_X[pair_idx2_train]
        pair_y_train = np.array([emd(calQ[i],calQ[j]) for i, j in zip(pair_idx1_train,pair_idx2_train)])

        pair_idx1_val = np.array([i for i,j in itertools.product(pair_val_indices,pair_val_indices)])
        pair_idx2_val = np.array([j for i,j in itertools.product(pair_val_indices,pair_val_indices)])

        pair_X1_val = pair_X[pair_idx1_val]
        pair_X2_val = pair_X[pair_idx2_val]
        pair_y_val = np.array([emd(calQ[i],calQ[j]) for i, j in zip(pair_idx1_val, pair_idx2_val)])

        print(pair_X1_train.shape)
        print(pair_X2_train.shape)
        print(pair_y_train.shape)

        print(pair_X1_val.shape)
        print(pair_X2_val.shape)
        print(pair_y_val.shape)
                             
        #Joining Training and Validation Sets
         
        X1_train = np.concatenate((ae_X1_train,pair_X1_train))
        X2_train = np.concatenate((ae_X2_train,pair_X2_train))
        y_train  = np.concatenate((ae_y_train,pair_y_train))

        X1_val   = np.concatenate((ae_X1_val,pair_X1_val))
        X2_val   = np.concatenate((ae_X2_val,pair_X2_val))
        y_val    = np.concatenate((ae_y_val,pair_y_val)) 
        
        print(X1_train.shape)
        print(X2_train.shape)
        print(y_train.shape)
    
        print(X1_val.shape)
        print(X2_val.shape)
        print(y_val.shape)
                             
        #Building CNN
        
        # make a convolutional model as a more advanced PoC
        input1 = Input(shape=(4, 4, 3,), name='input_1')
        input2 = Input(shape=(4, 4, 3,), name='input_2')
        x = Concatenate(name='concat')([input1, input2])

        #Number of Conv2D Layers
        for i in range(1,num_conv_2d+1):
            ind=str(i)
            x = Conv2D(num_filt, kernel_size, strides=(1, 1), name='conv2d_'+ind, padding='same', kernel_regularizer=l1_l2(l1=0,l2=1e-4))(x)
            x = BatchNormalization(name='batchnorm_'+ind)(x)
            x = Activation('relu', name='relu_'+ind)(x)

        x = Flatten(name='flatten')(x)

        #Number of Dense Layers
        for i in range(1,num_dens_layers+1):
            ind=str(i)
            jind=str(i+num_conv_2d)
            x = Dense(num_dens_neurons, name='dense_'+ind, kernel_regularizer=l1_l2(l1=0,l2=1e-4))(x)
            x = BatchNormalization(name='batchnorm'+jind)(x)
            x = Activation('relu', name='relu_'+jind)(x)

        output = Dense(1, name='output')(x)
        model = Model(inputs=[input1, input2], outputs=output, name='base_model')
        model.summary()

        # make a model that enforces the symmetry of the EMD function by averging the outputs for swapped inputs
        output = Average(name='average')([model((input1, input2)), model((input2, input1))])
        sym_model = Model(inputs=[input1, input2], outputs=output, name='sym_model')
        sym_model.summary()

        final_directory=os.path.join(current_directory,r'app_emd_loss_models')
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)
        callbacks = [ModelCheckpoint('app_emd_loss_models/'+str(num_filt)+str(kernel_size)+str(num_dens_neurons)+str(num_dens_layers)+str(num_conv_2d)+str(num_epochs)+'best.h5', monitor='val_loss', verbose=1, save_best_only=True),
                        ModelCheckpoint('app_emd_loss_models/'+str(num_filt)+str(kernel_size)+str(num_dens_neurons)+str(num_dens_layers)+str(num_conv_2d)+str(num_epochs)+'last.h5', monitor='val_loss', verbose=1, save_last_only=True),
                    ]

        sym_model.compile(optimizer='adam', loss='huber_loss', metrics=['mse', 'mae', 'mape', 'msle'])
        history = sym_model.fit((X1_train, X2_train), y_train, 
                            validation_data=((X1_val, X2_val), y_val),
                            epochs=num_epochs, verbose=1, batch_size=32, callbacks=callbacks)
        
        img_directory=os.path.join(current_directory,r'APP EMD PLots')
        if not os.path.exists(img_directory):
            os.makedirs(img_directory)
                             
        #Plot Validation loss and training loss
        
        plt.close()
        fig=plt.plot(history.history['loss'], label='Train')
        fig=plt.plot(history.history['val_loss'], label='Val.')
        fig=plt.xlabel('Epoch')
        fig=plt.ylabel('MSLE loss')
        fig=plt.legend()
        plt.savefig(img_directory+"/"+str(num_filt)+str(kernel_size)+str(num_dens_neurons)+str(num_dens_layers)+str(num_conv_2d)+str(num_epochs)+"Loss.png")
        plt.close()
        
        #Plots True EMD and Pred Emd Histogram
        
        plt.close()
        y_val_preds = sym_model.predict((X1_val, X2_val))
        fig=plt.figure()
        fig=plt.hist(y_val, alpha=0.5, bins=np.arange(0, 7.5, 0.01), label='True')
        fig=plt.hist(y_val_preds, alpha=0.5, bins=np.arange(0, 7.5, 0.01), label='Pred.')
        fig=plt.xlabel('EMD [GeV]')
        fig=plt.ylabel('Samples')
        fig=plt.legend()
        fig=plt.savefig(img_directory+"/"+str(num_filt)+str(kernel_size)+str(num_dens_neurons)+str(num_dens_layers)+str(num_conv_2d)+str(num_epochs)+"Hist.png")
        plt.close()
        
        #Plot Relative Difference
        
        plt.close()
        rel_diff = (y_val_preds[y_val>0].flatten()-y_val[y_val>0].flatten())/y_val[y_val>0].flatten()
        fig=plt.figure()
        fig=plt.hist(rel_diff, bins=np.arange(-1, 1, 0.01), color='green', label = 'mean = {:.3f}, std. = {:.3f}'.format(np.mean(rel_diff), np.std(rel_diff)))
        fig=plt.xlabel('EMD rel. diff.')
        fig=plt.ylabel('Samples')
        fig=plt.legend()
        fig=plt.savefig(img_directory+"/"+str(num_filt)+str(kernel_size)+str(num_dens_neurons)+str(num_dens_layers)+str(num_conv_2d)+str(num_epochs)+"RelD.png")
        plt.close()
        
        #Plot True EMD vs Pred Emd Graphic
        
        plt.close()
        fig, ax = plt.subplots(figsize =(5, 5)) 
        x_bins = np.arange(0, 7.5, 0.01)
        y_bins = np.arange(0, 7.5, 0.01)
        plt.hist2d(y_val.flatten(), y_val_preds.flatten(), bins=[x_bins,y_bins])
        plt.plot([0, 15], [0, 15], color='gray', alpha=0.5)
        ax.set_xlabel('True EMD [GeV]')
        ax.set_ylabel('Pred. EMD [GeV]')
        fig=plt.savefig(img_directory+"/"+str(num_filt)+str(kernel_size)+str(num_dens_neurons)+str(num_dens_layers)+str(num_conv_2d)+str(num_epochs)+"Graphic.png")
        plt.close()
        
        return(np.mean(rel_diff),np.std(rel_diff))