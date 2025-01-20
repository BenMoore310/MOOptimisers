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

functionDict = {
    func.binhAndKorn: [(0, 5), (0, 3)],
    func.chankongHaimes: [(-20, 20), (-20, 20)],
    func.fonsecaFleming: [(-4, 4), (-4, 4)],
    func.ctp1: [(0, 1), (0, 1)],
    func.constrEx: [(0.1, 1), (0, 5)],
    func.testFunction4: [(-7, 4), (-7, 4)],
}

sandTrapDict = {
    func.sandTrap: [(0.0001, 0.0005), (0, 0.075), (0.05, 0.2), (0.1, 2.0), (0.001, 0.1)]
}
# order is: inletConc, wallRoughness, maxCellSize, sigmaTurbConstant, turbVisc

sandTrapWeights = np.array((1 / 3, 1 / 3, 1 / 3))

weights = np.array((0.5, 0.5))
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

for key, value in functionDict.items():
    print(key.__name__, value)

    initSampleSize = 20
    bounds = np.array(value)
    lowBounds = bounds[:, 0]
    highBounds = bounds[:, 1]

    # Generate one Latin Hypercube Sample (LHS) for each test function,
    # to be used for all optimisers/scalarisers using a population size of 20
    sampler = qmc.LatinHypercube(
        d=bounds.shape[0]
    )  # Dimension is determined from bounds
    sample = sampler.random(n=initSampleSize)
    initPopulation = qmc.scale(sample, lowBounds, highBounds)

    # Check for and systematically replace NaN values in initial population
    # Requires evaluating initial population
    objvTargets = np.empty((0, 2))

    for i in range(initSampleSize):
        candidate = initPopulation[i]
        newObjvTgt = opt.MOobjective_function(candidate, key, bounds.shape[0])

        # Replace NaN values in the objective function result with valid samples
        while np.any(np.isnan(newObjvTgt)):
            candidate = np.random.uniform(
                lowBounds, highBounds
            )  # Match dimension automatically
            newObjvTgt = opt.MOobjective_function(candidate, key, bounds.shape[0])
            initPopulation[i] = candidate

    print("Initial Population:")
    print(initPopulation)

    for scalarisingFunction in scalarisingList:

        print(scalarisingFunction.__name__)

        LSADE = opt.LSADE(
            value,
            20,
            key,
            scalarisingFunction,
            2,
            weights,
            useInitialPopulation=True,
            initialPopulation=initPopulation,
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
            2,
            weights,
            useInitialPopulation=True,
            initialPopulation=initPopulation,
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
            2,
            weights,
            useInitialPopulation=True,
            initialPopulation=initPopulation,
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
            2,
            weights,
            0.9,
            80,
            useInitialPopulation=True,
            initialPopulation=initPopulation,
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
