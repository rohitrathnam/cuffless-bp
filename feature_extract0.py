import numpy as np
import matplotlib.pyplot as plt
import pywt


def get_bp(data):
    sbp = []
    dbp = []
    sbpval = 0
    dbpval = 0
    status = 0
    for no in range(0, 689):
        x, length = data.shape
        i = 1
        while i < length - 1:
            if (data[1, i] - data[1, i - 1]) <= 0 < (data[1, i + 1] - data[1, i]):
                dbp.append(data[1, i])
            if (data[1, i] - data[1, i - 1]) >= 0 > (data[1, i + 1] - data[1, i]):
                sbp.append(data[1, i])
            i = i + 1
        if np.std(sbp) <= 2 and np.std(dbp) <= 2:
            status = 1
            sbpval = int(np.mean(sbp))
            dbpval = int(np.mean(dbp))
    return [sbpval, dbpval, status]


def get_scal(data):
    coef, freqs = pywt.cwt(data, np.arange(2, 50), 'gaus3', sampling_period=0.008)
    return coef[:, 25:665]


file = 0
k = 0
inputs = np.empty((50, 48, 640))
outputs = np.empty((50, 2))
for i in range(0, 1499):
    data = np.load('raw_data/' + str(i) + '.npy')
    x, y = data.shape
    n = int(float(y) / 690) - 1
    j = 0
    while j <= n:
        #if i==3 and j<45:
        #    j = 45
        [sbp, dbp, status] = get_bp(data[:, j * 690:j * 690 + 689])
        print('                   %d' % j, end='\r')		
        if status == 1:
            inputs[k] = get_scal(data[0, j * 690:j * 690 + 689])
            outputs[k, 0] = sbp
            outputs[k, 1] = dbp
            k = k + 1
            print(k)
            if k == 50:
                np.save('dataset/i' + str(file) + '.npy', inputs)
                np.save('dataset/o' + str(file) + '.npy', outputs)
                print('writing to file '+str(file))
                file = file + 1
                break
        j = j + 1
    if k == 50:
        k = 0
        print("i = " + str(i) + "   j = " + str(j))
#i=82 j=83 file=4
