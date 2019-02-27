import pywt
import matplotlib.pyplot as plt
import numpy as np

data = np.load('raw_data/0.npy')
w = pywt.DiscreteContinuousWavelet('db2')
coef, freqs=pywt.cwt(data[0,0:1250],np.arange(1,50),'gaus3', sampling_period=0.008)
#odata = coef[25:1225]
#48x640