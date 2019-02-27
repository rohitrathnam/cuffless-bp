import numpy
import h5py
import scipy.io
import matplotlib
import os

f = scipy.io.loadmat("orig/part_12.mat")
x = 11000
for patient_no in range(0,1000):
	temp, total_no = f['p'][0, patient_no].shape
	a = f['p'][0, patient_no][:, :]
	name="raw_data/"+str(x)+".npy"
	numpy.save(name, a)
	x = x+1
print(x)
