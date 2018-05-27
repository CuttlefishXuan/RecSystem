import numpy as np
import scipy.sparse as sparse

ITEM_NUM = 624961
USER_NUM = 19835
DATA_NUM = 5001507


path = 'utility_matrix.npy'
mat = np.load(path)[()]
print("mat.shape", mat.shape)


bi = np.zeros(ITEM_NUM)
bx = np.zeros(USER_NUM)

mean = mat.mean() * 624961 * 19835 / DATA_NUM
print("mean = ", mean)



for x in range(USER_NUM):
    gradeArr = mat.getcol(x)
    if gradeArr.size == 0:
        bx[x] = 0
    else:
        bx[x] = np.mean(gradeArr.data) - mean
        if(np.mean(gradeArr.data)<0):
            print("------------ERROR  ERROR   bx mean<0-----------")
    if(x%1000==0):
        print(x, "bx", bx[x])
np.save("bx_new.npy", bx)
print("------------------bx Over-------------\n\n")



for i in range(ITEM_NUM):
    gradeArr = mat.getrow(i)
    if gradeArr.size == 0:
        bi[i] = 0
    else:
        bi[i] = np.mean(gradeArr.data) - mean
        if(np.mean(gradeArr.data)<0):
            print("------------ERROR  ERROR   bi mean<0-----------")
    # print(i, "bi-compute", bi[i], "bi-load", biload[i])
    if(i%1000==0):
        print(i, "bi", bi[i])
np.save("bi_new.npy", bi)
print("------------------bi Over-------------\n\n")