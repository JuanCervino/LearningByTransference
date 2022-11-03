import os
import numpy as np
import matplotlib
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['font.family'] = 'serif'
# matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']
import matplotlib.pyplot as plt
import pickle
import datetime
from copy import deepcopy

# import torch; torch.set_default_dtype(torch.float64)
# import torch.nn as nn
# import torch.optim as optim
#
# #\\\ Own libraries:
# import alegnn.utils.dataToolsOriginal as dataTools
# import alegnn.utils.graphML as gml
# import alegnn.modules.architecturesTime as architTime
# import alegnn.modules.model as model
# import alegnn.modules.evaluation as evaluation
#
# #\\\ Separate functions:
# from alegnn.utils.miscTools import writeVarValues
# from alegnn.utils.miscTools import saveSeed

# Start measuring time
startRunTime = datetime.datetime.now()

#%%##################################################################
#                                                                   #
#                    SETTING PARAMETERS                             #
#                                                                   #
#####################################################################



cwd=os.getcwd()
folderDir=cwd+'/experiments'

# trainDir = '/flockingGNNflockingGNN-010-20210504164030'
# trainDir = '/flockingGNNflockingGNN-010-20210504175133'
trainDir = '/flockingGNNflockingGNN-010-20210504182919'

# smallTrain = open (folderDir+smallTrainDir+'/figs'+"/figVars.pkl", "rb")
smallTrain = open (folderDir+trainDir+'/smallTrain'+'/figs'+"/figVars.pkl", "rb")
smallTrainFigs = pickle.load(smallTrain)


normalTrain = open (folderDir+trainDir+'/normalTrain'+'/figs'+"/figVars.pkl", "rb")
normalTrainFigs = pickle.load(normalTrain)
# normalTrainData = pickle.load(normalTrain)




# print(smallTrainFigs['nEpochs'],smallTrainFigs['nBatches'])
# print(len(smallTrainFigs['meanCostBestFull']['LocalGNN']))
# print(len(smallTrainFigs['meanCostBestEnd']['LocalGNN']))
# print(len(smallTrainFigs['meanCostLastFull']['LocalGNN']))

for key in smallTrainFigs.keys():
    print('-----------')
    print(key)
    # print(smallTrainFigs[key])

smallTrainEpochs=np.array(smallTrainFigs['costBestFull'][0]['LocalGNN'])
# print(smallTrainEpochs)
smallTrainTestMean=np.mean(smallTrainEpochs,axis=0)
smallTrainTestStd=np.std(smallTrainEpochs,axis=0)


normalTrainEpochs=np.array(normalTrainFigs['costBestFull'][0]['LocalGNN'])

normalTrainTestMean=np.mean(normalTrainEpochs,axis=0)
normalTrainTestStd=np.std(normalTrainEpochs,axis=0)

nodesInTest=30
nEpochs=smallTrainFigs['nEpochs']
nodesAddedPerEpoch=3
maxNodes=10

########
# PLOT #
########
figSize = 5 # Overall size of the figure that contains the plot
lineWidth = 2 # Width of the plot lines
markerShape = 'o' # Shape of the markers
markerSize = 3 # Size of the markers

# validationInterval = 5 # How many training steps to do the validation
# xAxisMultiplierTrain = 1
# xAxisMultiplierValid = 1
# # Compute the x-axis
xEpochs = np.arange(0, smallTrainFigs['nEpochs'] )
#
# for key in ['LocalGNN']:
#     print(len(xValidST), len(smallTrainFigs['meanLossTrain'][key]))

plt.errorbar(xEpochs, smallTrainTestMean, yerr=smallTrainTestStd,
                 linewidth=lineWidth, marker=markerShape, markersize=markerSize)
plt.errorbar(xEpochs, normalTrainTestMean, yerr=normalTrainTestStd,
                 linewidth=lineWidth, marker=markerShape, markersize=markerSize)
plt.legend(['Small Train','Normal Train'])
plt.title('Test Error for LocalGNN test nodes  ' + str(nodesInTest) + ' epochs ' + str(nEpochs) + '(' + str(
    nodesAddedPerEpoch) + ') start/final ' + str(maxNodes) + '/' + str(maxNodes + nodesAddedPerEpoch * (nEpochs - 1)))

# plt.show()
plt.savefig(cwd+'/experiments'+trainDir+'/results.png')