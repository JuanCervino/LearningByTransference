import os
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']
import matplotlib.pyplot as plt
import pickle
import datetime
from copy import deepcopy

import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim

#\\\ Own libraries:
import alegnn.utils.graphTools as graphTools
import alegnn.utils.dataToolsRandom
import alegnn.utils.graphML as gml
import alegnn.modules.architectures as archit
import alegnn.modules.model as model
import alegnn.modules.training_juan as training
import alegnn.modules.evaluation as evaluation
import alegnn.modules.loss as loss

#\\\ Separate functions:
from alegnn.utils.miscTools import writeVarValues
from alegnn.utils.miscTools import saveSeed


def plotMovie(trainDir):
    cwd=os.getcwd()
    folderDir=cwd+'/experiments'
    # This is Testing on 1400 with 13 epochs
    # Max Graph Size is 800
    # Small start at 200 with 50 nodes of increments

    # trainDir = '/movieGNN-movie-050-20210428183507'





    smallTrainDir = trainDir+'/smallTrain'
    normalTrainDir = trainDir+'/normalTrain'

    smallHyper = open (folderDir+smallTrainDir+'/hyperparms.pkl', "rb")
    smallHyper = pickle.load(smallHyper)

    nodesInTest=smallHyper['nodesInTest']
    nEpochs=smallHyper['nEpochs']
    maxNodes=smallHyper['maxNodes']
    nodesAddedPerEpoch=smallHyper['NodesAddedPerEpoch']

    # This is Testing on 1200 with 13 epochs
    # Max Graph Size is 800
    # Small start at 200 with 50 nodes of increments
    # smallTrainDir = '/movieGNN-movie-050-20210427183216'
    # normalTrainDir = '/movieGNN-movie-050-20210427183258'
    # nodesInTest=1400
    # nEpochs=13
    # maxNodes=200
    # nodesAddedPerEpoch=50


    smallTrain = open (folderDir+smallTrainDir+'/figs'+"/figVars.pkl", "rb")
    normalTrain = open (folderDir+normalTrainDir+'/figs'+"/figVars.pkl", "rb")
    smallTrainFigs = pickle.load(smallTrain)
    normalTrainFigs = pickle.load(normalTrain)
    smallTrain = open (folderDir+smallTrainDir+'/hyperparameters.txt', "r")
    normalTrain = open (folderDir+normalTrainDir+'/hyperparameters.txt', "r")

    # for x in smallTrain:
    #     print(x)
    #     print('aa')
    #
    # smallTrainData =
    # normalTrainData =

    figSize = 5 # Overall size of the figure that contains the plot
    lineWidth = 2 # Width of the plot lines
    markerShape = 'o' # Shape of the markers
    markerSize = 3 # Size of the markers

    xAxisMultiplierTrain=10
    xAxisMultiplierValid=10
    # validationInterval=5

    # selectSamplesValid = np.arange(0, len(meanLossValid[thisModel]), \
    #                                        xAxisMultiplierValid)

    xTrainST = np.arange(0, smallTrainFigs['nEpochs']*smallTrainFigs['nBatches'], xAxisMultiplierTrain, dtype=int)
    xValidST = np.arange(0, smallTrainFigs['nEpochs']*smallTrainFigs['nBatches'],   smallTrainFigs['validationInterval'] * xAxisMultiplierValid, dtype=int)
    xTestST = np.arange(smallTrainFigs['nBatches'], (smallTrainFigs['nEpochs'] + 1) * smallTrainFigs['nBatches'], smallTrainFigs['nBatches'])

    xTrainNT = np.arange(0, normalTrainFigs['nEpochs']*normalTrainFigs['nBatches'], xAxisMultiplierTrain, dtype=int)
    xValidNT = np.arange(0, normalTrainFigs['nEpochs']*normalTrainFigs['nBatches'],   normalTrainFigs['validationInterval'] * xAxisMultiplierValid, dtype=int)
    xTestNT = np.arange(normalTrainFigs['nBatches'], (normalTrainFigs['nEpochs'] + 1) * normalTrainFigs['nBatches'], normalTrainFigs['nBatches'])

    print(smallTrainFigs['meanLossTrain']['LclGNN1Ly'])
    print(normalTrainFigs['meanLossTrain']['LclGNN1Ly'])
    for key in ['LclGNN1Ly','LclGNN2Ly']:
        plt.figure()
        # Train
        plt.errorbar(xValidST, smallTrainFigs['meanLossTrain'][key][xValidST], yerr=smallTrainFigs['stdDevLossTrain'][key][xValidST],
                     linewidth=lineWidth,
                     marker=markerShape, markersize=markerSize)
        plt.errorbar(xValidST, normalTrainFigs['meanLossTrain'][key][xValidNT], yerr=normalTrainFigs['stdDevLossTrain'][key][xValidNT],
                     linewidth=lineWidth, marker=markerShape, markersize=markerSize)
        plt.title('Train Error for '+key)
        plt.legend(['Small Train','Normal Train'])
        plt.savefig(folderDir+trainDir+'/train'+key+'.png')
        # Test
        plt.figure()
        plt.errorbar(xTestST, smallTrainFigs['meanCostLast'][key], yerr=smallTrainFigs['stdDevCostLast'][key],
                     linewidth=lineWidth,
                     marker=markerShape, markersize=markerSize)
        plt.errorbar(xTestNT, normalTrainFigs['meanCostLast'][key], yerr=normalTrainFigs['stdDevCostLast'][key],
                     linewidth=lineWidth, marker=markerShape, markersize=markerSize)
        plt.title('Test Error for '+key+' test nodes  '+str(nodesInTest)+' epochs '+str(nEpochs)+'('+str(nodesAddedPerEpoch)+') start/final '+str(maxNodes)+'/'+str(maxNodes+nodesAddedPerEpoch*(nEpochs-1)))
        plt.legend(['Small Train','Normal Train'])
        plt.savefig(folderDir+trainDir+'/test'+key+'.png')
        plt.close('all')

