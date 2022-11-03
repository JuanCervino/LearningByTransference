import flockingGNN_small_train
import plotFlocking
import datetime


lInitialNodes  = [10,10,10]
lFinalNodes = [22,22,22]
lNodesAddedPerEpoch = [1,2,3]
lNodesInTest = [30,30,30]

for i in range(len(lInitialNodes)):
    nEpochs=(lFinalNodes[i]-lInitialNodes[i])/(lNodesAddedPerEpoch[i])+1
    assert nEpochs.is_integer()
    nEpochs=int(nEpochs)
    # get folder to save
    today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # First we go with small train
    saveDir=flockingGNN_small_train.flockingSmallTrain(lInitialNodes[i],lNodesAddedPerEpoch[i],lNodesInTest[i],nEpochs,today)
    # Now we train normally
    flockingGNN_small_train.flockingSmallTrain(lFinalNodes[i],0,lNodesInTest[i],nEpochs,today)

    # plotMovie.plotMovie('/'+saveDir)