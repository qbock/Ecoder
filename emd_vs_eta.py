import numpy as np
import matplotlib.pyplot as plt

from utils.metrics import emd
from utils.metrics import hexMetric


def plot_eta(input_Q, output_Q, phys):
  
  indices = range(0,(len(input_Q)))
  emd_values = np.array([emd(input_Q[i],output_Q[j]) for i, j in zip(indices,indices)])
  phi=[]
  for i in indices:
    phi=np.append(phi,phys[i][0])
    
  plt.plot(phi.flatten(),emd_values.flatten())
  
  plt.xlabel(r'$\eta$')
  plt.ylabel('EMD')
  
  plt.savefig('./EMD_vs_eta.pdf')
    
 
