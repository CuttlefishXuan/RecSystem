import numpy as np
import scipy.sparse as sparse
import datetime

ITEM_NUM = 624961
USER_NUM = 19835
DATA_NUM = 5001507


path = 'utility_matrix.npy'
coomat = np.load(path)[()]
mat = coomat.tocsc()
print(type(mat))
print("mat.shape", mat.shape)


bi = np.load("bi_csc.npy")[()]

mean = mat.mean() * 624961 * 19835 / DATA_NUM
print("mean = ", mean)



for i in range(ITEM_NUM):
    if(np.abs(bi[i]+49)<2):
        bi[i] = 0
    if(i%1000==0):
        print(i, "bi", bi[i])
np.save("bi_csc_no0.npy", bi)
print("\n--------------Compute bi Over----------\n\n")

'''
for x in range(USER_NUM):
    if(bx[x]==0):
        continue
    else:
        bx[x] = bx[x] + oldmean - mean
    if(x%1000==0):
        print(x, "bx", bx[x])
np.save("bx_csc.npy", bx)
print("\n--------------Compute bx Over----------\n")
'''