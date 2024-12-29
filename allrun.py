import optimiserBank as opt
import functionBank as func
import matplotlib.pyplot as plt
import importlib
importlib.reload(opt)
importlib.reload(func)
import numpy as np
import matplotlib.pyplot as plt

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

    for scalarisingFunction in scalarisingList:

        print(scalarisingFunction.__name__)

        LSADE = opt.LSADE(value, 15, key, scalarisingFunction, 2, weights)
        LSADE.optimizerStep()

        features = np.loadtxt('LSADEFeatures.txt')
        np.savetxt(f'LSADEFeatures{key.__name__}{scalarisingFunction.__name__}.txt', features)

        scalarisedTargets = np.loadtxt('LSADEScalarisedTargets.txt')
        np.savetxt(f'LSADEScalarisedTargets{key.__name__}{scalarisingFunction.__name__}.txt', scalarisedTargets)

        objtTargets = np.loadtxt('LSADEObjectiveTargets.txt')
        np.savetxt(f'LSADEObjtvTargets{key.__name__}{scalarisingFunction.__name__}.txt', objtTargets)

        PSO = opt.TS_DDEO(value, 15, key, scalarisingFunction, 2, weights)
        PSO.stage1()
        PSO.stage2()

        features = np.loadtxt('TSDDEOFeatures.txt')
        np.savetxt(f'BOFeatures{key.__name__}{scalarisingFunction.__name__}.txt', features)

        scalarisedTargets = np.loadtxt('TSDDEOTargets.txt')
        np.savetxt(f'TSDDEOScalarisedTargets{key.__name__}{scalarisingFunction.__name__}.txt', scalarisedTargets)

        objtTargets = np.loadtxt('TSDDEOObjectiveTargets.txt')
        np.savetxt(f'TSDDEOObjtvTargets{key.__name__}{scalarisingFunction.__name__}.txt', objtTargets)

        
        bayesianRun = opt.bayesianOptimiser(value, 15, key, scalarisingFunction, 2, weights)
        bayesianRun.runOptimiser()

        features = np.loadtxt('BOFeatures.txt')
        np.savetxt(f'BOFeatures{key.__name__}{scalarisingFunction.__name__}.txt', features)

        scalarisedTargets = np.loadtxt('BOScalarisedTargets.txt')
        np.savetxt(f'BOScalarisedTargets{key.__name__}{scalarisingFunction.__name__}.txt', scalarisedTargets)

        objtTargets = np.loadtxt('BOObjectiveTargets.txt')
        np.savetxt(f'BOObjtvTargets{key.__name__}{scalarisingFunction.__name__}.txt', objtTargets)


        ESA = opt.ESA(value, 15, 10, 0.25, key, scalarisingFunction, 2, weights, 0.9, 80)
        ESA.mainMenu(initialAction=2)

        features = np.loadtxt('ESAFeatures.txt')
        np.savetxt(f'ESAFeatures{key.__name__}{scalarisingFunction.__name__}.txt', features)

        scalarisedTargets = np.loadtxt('ESAScalarisedTargets.txt')
        np.savetxt(f'ESAScalarisedTargets{key.__name__}{scalarisingFunction.__name__}.txt', scalarisedTargets)

        objtTargets = np.loadtxt('ESAObjectiveTargets.txt')
        np.savetxt(f'ESAObjtvTargets{key.__name__}{scalarisingFunction.__name__}.txt', objtTargets)