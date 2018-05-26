import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
import datetime

ITEM_NUM = 624961
USER_NUM = 19835
DATA_NUM = 5001507

path = 'utility_matrix.npy'
mat = np.load(path)[()]
print("mat.shape", mat.shape)

i = np.load("i.npy")[()]
j = np.load("j.npy")[()]
data = np.load("data.npy")[()]

i_train, i_test, j_train, j_test, data_train, data_test = train_test_split(i, j, data, test_size=0.15)

print("train.shape", i_train.shape)
print("test.shape", i_test.shape)

coomat_train = sparse.coo_matrix((data_train, (i_train, j_train)), shape=(ITEM_NUM, USER_NUM))
coomat_test = sparse.coo_matrix((data_test, (i_test, j_test)), shape=(ITEM_NUM, USER_NUM))
mat_train = coomat_train.tocsc()
mat_test = coomat_test.tocsc()
size_train = mat_train.size
size_test = mat_test.size
print("size_train", size_train)
print("train_shape", mat_train.shape)


# u, sigma, vt = svds(mat_train, 50)

featureK = 50
Q = np.random.rand(mat_train.shape[0], featureK) * 0.2
PT = np.random.rand(featureK, mat_train.shape[1]) * 0.2
print(Q.shape)
print(PT.shape)
print("-------------Init Q PT finished------------\n")


def Predict(x, i, meanOverAll, bx, bi, Q, PT):
    return meanOverAll + bx[x] + bi[i] + np.sum(Q[i,:].dot(PT[:,x]))


# meanOverAll,bx,bi = init()
meanOverAll = mat_train.mean() * 624961 * 19835 / size_train
bx = np.zeros(USER_NUM)
bi = np.zeros(ITEM_NUM)
print("meanOverAll = ", meanOverAll)


steps = 100
gamma = 10**(-4) # 大一点
Lambda = 0.05 # 0.005

SSE = 0.0
for x in range(USER_NUM):
    userArray = mat_train.getcol(x)
    gradeArray = userArray.data
    indexArray = userArray.indices # user X
    predictArray = np.zeros(indexArray.size)
    index = 0
    for i in indexArray:
        predictArray[index] = Predict(x, i, meanOverAll, bx, bi, Q, PT)
        epsiloxi = gradeArray[index] - predictArray[index]
        # SSE += epsiloxi**2
        SSE += epsiloxi**2 + Lambda * (np.sum(np.square(Q[i,:])) + np.sum(np.square(PT[:,x])) + (bx[x]**2) + np.sum(bi[i]**2))
        index = index + 1

RMSE = np.sqrt(SSE/size_train)
print("\n-------------Init RMSE", RMSE)



print("\n\n-------------------Start SGD-----------------")




print("gamma", gamma, "Lambda", Lambda, "steps", steps)
print("---------------------------------------------\n\n")

for step in range(steps):
    print('the ', step+1, 'th  step is running')
    starttime = datetime.datetime.now()
    SSE = 0.0

    for x in range(USER_NUM):
        userArray = mat_train.getcol(x)
        gradeArray = userArray.data
        indexArray = userArray.indices # user X
        predictArray = np.zeros(indexArray.size)
        index = 0
        for i in indexArray:
            predictArray[index] = Predict(x, i, meanOverAll, bx, bi, Q, PT)
            epsiloxi = gradeArray[index] - predictArray[index]
            # SSE += epsiloxi**2  # +正则   /2
            SSE += epsiloxi**2 + Lambda * (np.sum(np.square(Q[i,:])) + np.sum(np.square(PT[:,x])) + (bx[x]**2) + np.sum(bi[i]**2))
            tempQi = Q[i,:]
            Q[i,:] += gamma * (epsiloxi * np.transpose(PT[:,x]) - Lambda * Q[i,:])
            PT[:,x] += gamma * (epsiloxi * np.transpose(tempQi) - Lambda * PT[:,x])
            bx[x] += gamma * (epsiloxi - Lambda * bx[x])
            bi[i] += gamma * (epsiloxi - Lambda * bi[i])
            
            index = index + 1

    RMSE = np.sqrt(SSE/size_train)
    print("End", RMSE)
    endtime = datetime.datetime.now()
    print("Time", (endtime-starttime).seconds)
    # gamma = gamma * 0.93
    print("---------------------------------------------\n")


np.save("Q_bxi_0", Q)
np.save("PT_bxi_0", PT)
np.save("bx_0", bx)
np.save("bi_0", bi)


print("\n--------------------train finished.----------------\n")

test_SSE = 0.0

for x in range(USER_NUM):
    userArray = mat_test.getcol(x)
    gradeArray = userArray.data
    indexArray = userArray.indices
    predictArray = np.zeros(indexArray.size)
    index = 0
    for i in indexArray:
        predictArray[index] = Predict(x, i, meanOverAll, bx, bi, Q, PT)
        epsiloxi = gradeArray[index] - predictArray[index]
        # test_SSE += epsiloxi**2
        test_SSE += epsiloxi**2 + Lambda * (np.sum(np.square(Q[i,:])) + np.sum(np.square(PT[:,x])) + (bx[x]**2) + np.sum(bi[i]**2))
        index += 1
RMSE = np.sqrt(test_SSE/size_test)
print("Test Result: ", RMSE)
print("------------------------------------")