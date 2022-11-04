import movieGNN_small_train
import plotMovie
import datetime


lInitialNodes  = [1000,1000,1000,1000,1000,1000]
lFinalNodes = [2000,2000,2000,2000,2000,2000]
lNodesAddedPerEpoch = [50,50,50,50,50,50]
lNodesInTest = [2100,2200,2300,2400,2500,2600]

for i in range(len(lInitialNodes)):
    nEpochs=(lFinalNodes[i]-lInitialNodes[i])/(lNodesAddedPerEpoch[i])+1
    assert nEpochs.is_integer()
    nEpochs=int(nEpochs)
    # get folder to save
    today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # First we go with small train
    saveDir=movieGNN_small_train.MovieSmallTrain(lInitialNodes[i],lNodesAddedPerEpoch[i],lNodesInTest[i],nEpochs,today)
    # Now we train normally
    movieGNN_small_train.MovieSmallTrain(lFinalNodes[i],0,lNodesInTest[i],nEpochs,today)

    plotMovie.plotMovie('/'+saveDir)    

# lInitialNodes  = [200,200,200,200,200,200]
# lFinalNodes = [1000,1000,1000,1000,1000,1000]
# lNodesAddedPerEpoch = [25,25,25,25,25,25]
# lNodesInTest = [1100,1200,1300,1400,1500,1600]

# for i in range(len(lInitialNodes)):
#     nEpochs=(lFinalNodes[i]-lInitialNodes[i])/(lNodesAddedPerEpoch[i])+1
#     assert nEpochs.is_integer()
#     nEpochs=int(nEpochs)
#     # get folder to save
#     today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
#     # First we go with small train
#     saveDir=movieGNN_small_train.MovieSmallTrain(lInitialNodes[i],lNodesAddedPerEpoch[i],lNodesInTest[i],nEpochs,today)
#     # Now we train normally
#     movieGNN_small_train.MovieSmallTrain(lFinalNodes[i],0,lNodesInTest[i],nEpochs,today)

#     plotMovie.plotMovie('/'+saveDir)

# lInitialNodes  = [200,200,200,200,200,200,200,200]
# lFinalNodes = [800,800,800,800,800,800,800,800]
# lNodesAddedPerEpoch = [25,25,25,25,25,25,25,25]
# lNodesInTest = [900,1000,1100,1200,1300,1400,1500,1600]

# for i in range(len(lInitialNodes)):
#     nEpochs=(lFinalNodes[i]-lInitialNodes[i])/(lNodesAddedPerEpoch[i])+1
#     assert nEpochs.is_integer()
#     nEpochs=int(nEpochs)
#     # get folder to save
#     today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
#     # First we go with small train
#     saveDir=movieGNN_small_train.MovieSmallTrain(lInitialNodes[i],lNodesAddedPerEpoch[i],lNodesInTest[i],nEpochs,today)
#     # Now we train normally
#     movieGNN_small_train.MovieSmallTrain(lFinalNodes[i],0,lNodesInTest[i],nEpochs,today)

#     plotMovie.plotMovie('/'+saveDir)
