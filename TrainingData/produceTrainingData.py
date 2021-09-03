import uproot
import numpy as np
import pandas as pd
import awkward as ak

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input',dest='inputRootFile', default='ntuple.root', help="TPG Ntuple file to process")
parser.add_argument('-d','--dir',dest='rootFileTDirectory', default='FloatingpointThreshold0DummyHistomaxGenmatchGenclustersntuple', help="Directory within input root file to find HGCalTriggerNtuple TTree")
parser.add_argument('-N',dest='N_eLinks',type=int,default=5,help='Number of eRx to select')
parser.add_argument('--ADC',dest='useADC',default=False,action='store_true',help='Use ADC rather than transverse ADC')
parser.add_argument('-o','--output',dest='outputFileName',default='CALQ.csv',help='Output file name (either a .csv or .pkl file name)')

args = parser.parse_args()


_tree = uproot.open(args.inputRootFile)[f'{args.rootFileTDirectory}/HGCalTriggerNtuple']

hasSimEnergy = 'tc_simenergy' in _tree
if hasSimEnergy:
    arrays = _tree.arrays(['tc_zside','tc_layer','tc_waferu','tc_waferv','tc_cellu','tc_cellv','tc_data','tc_eta','tc_simenergy'])
else:
    arrays = _tree.arrays(['tc_zside','tc_layer','tc_waferu','tc_waferv','tc_cellu','tc_cellv','tc_data','tc_eta'])

select_eLinks = {5 : (arrays[b'tc_layer']==9),
                 4 : (arrays[b'tc_layer']==7) | (arrays[b'tc_layer']==11),
                 3 : (arrays[b'tc_layer']==13),
                 2 : (arrays[b'tc_layer']<7) | (arrays[b'tc_layer']>13),
                 -1 : (arrays[b'tc_layer']>0)}

assert args.N_eLinks in select_eLinks

df = ak.to_pandas(arrays[select_eLinks[args.N_eLinks]])

dfRemap = pd.read_csv('tcRemap.csv')
df = df.reset_index().merge(dfRemap)
df['ADCT'] = (df.tc_data* ((1./np.cosh(df.tc_eta))*2**12).astype(int)/2**12).astype(int)

#create new unique index (can't get pivot working on multi-indexing, but this is essentially the same)
df['WaferEntryIdx'] = (df.entry*1000000 + df.tc_layer*10000 + df.tc_waferu*100 + df.tc_waferv)*df.tc_zside

val = 'ADCT'
if args.useADC:
    val='tc_data'
dfTrainData = df.pivot(index='WaferEntryIdx',columns='tc_cell_train',values=val).fillna(0).astype(int)
dfTrainData.columns = [f'CALQ_{i}' for i in range(48)]

dfTrainData[['entry','zside','layer','waferu','waferv']] = df.groupby(['WaferEntryIdx'])[['entry','tc_zside','tc_layer','tc_waferu','tc_waferv']].mean()

if hasSimEnergy:
    dfTrainData['simenergy'] = df.groupby(['WaferEntryIdx'])[['tc_simenergy']].sum()

dfTrainData.reset_index(drop=True,inplace=True)

if '.csv' in args.outputFileName:
    dfTrainData.to_csv(args.outputFileName,index=False)
if '.pkl' in args.outputFileName:
    import pickle
    pickle.dump(dfTrainData, open(args.outputFileName,'wb'))
