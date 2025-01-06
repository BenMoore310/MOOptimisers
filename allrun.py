import optimiserBank as opt
import functionBank as func
import matplotlib.pyplot as plt
import importlib
importlib.reload(opt)
importlib.reload(func)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc 

functionDict = {func.binhAndKorn:[(0,5), (0,3)], 
                func.chankongHaimes:[(-20,20), (-20,20)], 
                func.fonsecaFleming:[(-4,4), (-4,4)], 
                func.ctp1:[(0,1), (0,1)], 
                func.constrEx:[(0.1,1), (0,5)], 
                func.testFunction4:[(-7,4), (-7,4)]}

weights = np.array((0.5, 0.5))
scalarisingList = [func.chebyshev,
                    func.weightedSum, 
                    func.EWC, 
                    func.weightedPower,
                    func.weightedNorm, 
                    func.augmentedChebychev, 
                    func.modifiedChebychev, 
                    func.PBI, 
                    func.PAPBI]

for key, value in functionDict.items():

    print(key.__name__, value)

    initSampleSize = 20
    bounds = np.array(value)
    lowBounds = bounds[:,0]
    highBounds = bounds[:,1]


    #generate one LHS for each test function, to be used for all optimisers/scalarisers
    #using a population size of 20
    sampler = qmc.LatinHypercube(d=len(bounds))
    sample = sampler.random(n=initSampleSize)
    initPopulation = qmc.scale(sample, lowBounds, highBounds)

    #check for and systematically replace nan values in initial population
    #(requires evaluating initial population)
    #TODO: this evaluation is currently repeated when initiating each SAEA.
    
    objvTargets = np.empty((0,2))

    for i in range(0, initSampleSize):

        newObjvTgt = opt.MOobjective_function(initPopulation[i], key, len(bounds))
        # print(newObjvTgt)
        while np.any(np.isnan(newObjvTgt)):
            newSample = np.random.uniform(lowBounds, high=highBounds, size=(2,))
            newObjvTgt = opt.MOobjective_function(newSample, key, len(bounds))
            initPopulation[i] = newSample

        # objvTargets = np.vstack((objvTargets, newObjvTgt))

    print('Initial Population:')
    print(initPopulation)

    for scalarisingFunction in scalarisingList:

        print(scalarisingFunction.__name__)

        LSADE = opt.LSADE(value, 20, key, scalarisingFunction, 2, weights, useInitialPopulation=True, initialPopulation=initPopulation)
        LSADE.optimizerStep()

        features = np.loadtxt('LSADEFeatures.txt')
        np.savetxt(f'LSADEFeatures{key.__name__}{scalarisingFunction.__name__}.txt', features)

        scalarisedTargets = np.loadtxt('LSADEScalarisedTargets.txt')
        np.savetxt(f'LSADEScalarisedTargets{key.__name__}{scalarisingFunction.__name__}.txt', scalarisedTargets)

        objtTargets = np.loadtxt('LSADEObjectiveTargets.txt')
        np.savetxt(f'LSADEObjtvTargets{key.__name__}{scalarisingFunction.__name__}.txt', objtTargets)

        PSO = opt.TS_DDEO(value, 20, key, scalarisingFunction, 2, weights, useInitialPopulation=True, initialPopulation=initPopulation)
        PSO.stage1()
        PSO.stage2()

        features = np.loadtxt('TSDDEOFeatures.txt')
        np.savetxt(f'BOFeatures{key.__name__}{scalarisingFunction.__name__}.txt', features)

        scalarisedTargets = np.loadtxt('TSDDEOTargets.txt')
        np.savetxt(f'TSDDEOScalarisedTargets{key.__name__}{scalarisingFunction.__name__}.txt', scalarisedTargets)

        objtTargets = np.loadtxt('TSDDEOObjectiveTargets.txt')
        np.savetxt(f'TSDDEOObjtvTargets{key.__name__}{scalarisingFunction.__name__}.txt', objtTargets)

        
        bayesianRun = opt.bayesianOptimiser(value, 15, key, scalarisingFunction, 2, weights, useInitialPopulation=True, initialPopulation=initPopulation)
        bayesianRun.runOptimiser()

        features = np.loadtxt('BOFeatures.txt')
        np.savetxt(f'BOFeatures{key.__name__}{scalarisingFunction.__name__}.txt', features)

        scalarisedTargets = np.loadtxt('BOScalarisedTargets.txt')
        np.savetxt(f'BOScalarisedTargets{key.__name__}{scalarisingFunction.__name__}.txt', scalarisedTargets)

        objtTargets = np.loadtxt('BOObjectiveTargets.txt')
        np.savetxt(f'BOObjtvTargets{key.__name__}{scalarisingFunction.__name__}.txt', objtTargets)


        ESA = opt.ESA(value, 20, 10, 0.25, key, scalarisingFunction, 2, weights, 0.9, 80, useInitialPopulation=True, initialPopulation=initPopulation)
        ESA.mainMenu(initialAction=2)

        features = np.loadtxt('ESAFeatures.txt')
        np.savetxt(f'ESAFeatures{key.__name__}{scalarisingFunction.__name__}.txt', features)

        scalarisedTargets = np.loadtxt('ESAScalarisedTargets.txt')
        np.savetxt(f'ESAScalarisedTargets{key.__name__}{scalarisingFunction.__name__}.txt', scalarisedTargets)

        objtTargets = np.loadtxt('ESAObjectiveTargets.txt')
        np.savetxt(f'ESAObjtvTargets{key.__name__}{scalarisingFunction.__name__}.txt', objtTargets)