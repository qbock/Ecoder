## Load Training Data

Loads data from root ntuples into a format usable for training.

### Input data

Input data comes from the output of the HGCAL TPG ntuplizer: https://twiki.cern.ch/twiki/bin/viewauth/CMS/HGCALTriggerPrimitivesSimulation

The ntuplizer should be run with a threshold sum algorithm, with the threshold set to 0, such that all trigger cells are saved in the resulting root file.

