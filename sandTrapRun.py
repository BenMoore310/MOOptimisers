import optimiserBank as opt
import functionBank as func
import matplotlib.pyplot as plt
import importlib

importlib.reload(opt)
importlib.reload(func)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

# run with:

# nohup python3.11 -u allrun.py > allRunLogNormalised 2>&1 &


sandTrapDict = {
    func.sandTrap: [(0.0001, 0.0005), (0, 0.075), (0.05, 0.2), (0.1, 2.0), (0.001, 0.1)]
}
# order is: inletConc, wallRoughness, maxCellSize, sigmaTurbConstant, turbVisc

sandTrapWeights = np.array((1 / 3, 1 / 3, 1 / 3))

scalarisingList = [
    func.HypI,
    func.chebyshev,
    func.weightedSum,
    func.EWC,
    func.weightedPower,
    func.weightedNorm,
    func.augmentedChebychev,
    func.modifiedChebychev,
    func.PBI,
    func.PAPBI,
]



# temporary list to run only the new HypI scalarising function
# scalarisingList = [func.HypI]

# initPopulation = np.array([[2.71774478e-04, 1.06624662e-02, 1.61572910e-01, 7.88754700e-01,
#   8.49862317e-02],
#  [2.26544913e-04 ,4.20017446e-02, 1.20045421e-01 ,9.31354269e-01,
#   7.55967468e-03],
#  [1.11567477e-04 ,4.56233761e-02, 1.81794950e-01 ,1.98978413e+00,
#   4.93982515e-02],
#  [4.39090704e-04 ,1.54648860e-02 ,9.60211045e-02, 3.91748950e-01,
#   3.76541760e-02],
#  [3.65540217e-04 ,7.01556452e-02 ,6.53417300e-02, 1.24438485e+00,
#   7.40832369e-02]])

# initialObjvTargets = np.array([
#     [100.74126249999998, 36.15734375, 121.6074875],
#     [47.50642500000001, 21.31479375, 58.419337500000005],
#     [133.50328125000001, 60.9302875, 48.79143625],
#     [141.05033749999996, 55.3736625, 72.153948125],
#     [86.5700875, 41.091212500000005, 71.36274374999999]])

initPopulation = np.loadtxt('sandTrapInitPop.txt')
initialObjvTargets = np.loadtxt('sandTrapInitObjvTargets.txt')

for key, value in sandTrapDict.items():
    print(key.__name__, value)

    # initSampleSize = 20
    # bounds = np.array(value)
    # lowBounds = bounds[:, 0]
    # highBounds = bounds[:, 1]

    # # Generate one Latin Hypercube Sample (LHS) for each test function,
    # # to be used for all optimisers/scalarisers using a population size of 20
    # sampler = qmc.LatinHypercube(
    #     d=bounds.shape[0]
    # )  # Dimension is determined from bounds
    # sample = sampler.random(n=initSampleSize)
    # initPopulation = qmc.scale(sample, lowBounds, highBounds)

    # # Check for and systematically replace NaN values in initial population
    # # Requires evaluating initial population
    # initialObjvTargets = np.empty((0, 3))

    # for i in range(initSampleSize):
    #     candidate = initPopulation[i]
    #     newObjvTgt = opt.MOobjective_function(candidate, key, 3)

    #     # Replace NaN values in the objective function result with valid samples
    #     while np.any(np.isnan(newObjvTgt)):
    #         candidate = np.random.uniform(
    #             lowBounds, highBounds
    #         )  # Match dimension automatically
    #         newObjvTgt = opt.MOobjective_function(candidate, key, 3)
    #         initPopulation[i] = candidate
    #     initialObjvTargets = np.vstack((initialObjvTargets, newObjvTgt))

    # print("Initial Population:")
    # print(initPopulation)
    # np.savetxt('sandTrapInitPop.txt', initPopulation)
    # np.savetxt('sandTrapInitObjvTargets.txt', initialObjvTargets)

    for scalarisingFunction in scalarisingList:

        print(scalarisingFunction.__name__)

        LSADE = opt.LSADE(
            value,
            20,
            key,
            scalarisingFunction,
            3,
            sandTrapWeights,
            useInitialPopulation=True,
            initialPopulation=initPopulation,
            initialObjvValues=initialObjvTargets
        )
        LSADE.optimizerStep()

        features = np.loadtxt("LSADEFeatures.txt")
        np.savetxt(
            f"LSADEFeatures{key.__name__}{scalarisingFunction.__name__}.txt", features
        )

        scalarisedTargets = np.loadtxt("LSADEScalarisedTargets.txt")
        np.savetxt(
            f"LSADEScalarisedTargets{key.__name__}{scalarisingFunction.__name__}.txt",
            scalarisedTargets,
        )

        objtTargets = np.loadtxt("LSADEObjectiveTargets.txt")
        np.savetxt(
            f"LSADEObjtvTargets{key.__name__}{scalarisingFunction.__name__}.txt",
            objtTargets,
        )

        PSO = opt.TS_DDEO(
            value,
            20,
            key,
            scalarisingFunction,
            3,
            sandTrapWeights,
            useInitialPopulation=True,
            initialPopulation=initPopulation,
            initialObjvValues=initialObjvTargets
        )
        PSO.stage1()
        PSO.stage2()

        features = np.loadtxt("TSDDEOFeatures.txt")
        np.savetxt(
            f"TSDDEOFeatures{key.__name__}{scalarisingFunction.__name__}.txt", features
        )

        scalarisedTargets = np.loadtxt("TSDDEOTargets.txt")
        np.savetxt(
            f"TSDDEOScalarisedTargets{key.__name__}{scalarisingFunction.__name__}.txt",
            scalarisedTargets,
        )

        objtTargets = np.loadtxt("TSDDEOObjectiveTargets.txt")
        np.savetxt(
            f"TSDDEOObjtvTargets{key.__name__}{scalarisingFunction.__name__}.txt",
            objtTargets,
        )

        bayesianRun = opt.bayesianOptimiser(
            value,
            15,
            key,
            scalarisingFunction,
            3,
            sandTrapWeights,
            useInitialPopulation=True,
            initialPopulation=initPopulation,
            initialObjvValues=initialObjvTargets
        )
        bayesianRun.runOptimiser()

        features = np.loadtxt("BOFeatures.txt")
        np.savetxt(
            f"BOFeatures{key.__name__}{scalarisingFunction.__name__}.txt", features
        )

        scalarisedTargets = np.loadtxt("BOScalarisedTargets.txt")
        np.savetxt(
            f"BOScalarisedTargets{key.__name__}{scalarisingFunction.__name__}.txt",
            scalarisedTargets,
        )

        objtTargets = np.loadtxt("BOObjectiveTargets.txt")
        np.savetxt(
            f"BOObjtvTargets{key.__name__}{scalarisingFunction.__name__}.txt",
            objtTargets,
        )

        ESA = opt.ESA(
            value,
            20,
            10,
            0.25,
            key,
            scalarisingFunction,
            3,
            sandTrapWeights,
            0.9,
            80,
            useInitialPopulation=True,
            initialPopulation=initPopulation,
            initialObjvValues=initialObjvTargets
        )
        ESA.mainMenu(initialAction=2)

        features = np.loadtxt("ESAFeatures.txt")
        np.savetxt(
            f"ESAFeatures{key.__name__}{scalarisingFunction.__name__}.txt", features
        )

        scalarisedTargets = np.loadtxt("ESAScalarisedTargets.txt")
        np.savetxt(
            f"ESAScalarisedTargets{key.__name__}{scalarisingFunction.__name__}.txt",
            scalarisedTargets,
        )

        objtTargets = np.loadtxt("ESAObjectiveTargets.txt")
        np.savetxt(
            f"ESAObjtvTargets{key.__name__}{scalarisingFunction.__name__}.txt",
            objtTargets,
        )
