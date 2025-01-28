import optimiserBank as opt
import functionBank as func
import matplotlib.pyplot as plt
import importlib
importlib.reload(opt)
importlib.reload(func)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc 
from itertools import product
from pymoo.problems import get_problem

dtlzProblems = ['dtlz1','dtlz2','dtlz3','dtlz4','dtlz5','dtlz6','dtlz7']

n_var = 4
n_obj = 2

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

weights = np.array((0.5, 0.5))


for function in dtlzProblems:

    print('current function  = ', function)

    problem, bounds = func.getPyMooProblem(function, n_var, n_obj)

    initSampleSize = 20
    # bounds = np.array(value)
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
    initialObjvTargets = np.empty((0, n_obj))


    for i in range(initSampleSize):

        newObjvTgt = opt.MOobjective_function(initPopulation[i], problem, n_obj)
        initialObjvTargets = np.vstack((initialObjvTargets, newObjvTgt))

    print("Initial Population:")
    print(initPopulation)
    print("initial targets:\n", initialObjvTargets )

    for scalarisingFunction in scalarisingList:

        print(scalarisingFunction.__name__)

        LSADE = opt.LSADE(
            bounds,
            20,
            problem,
            scalarisingFunction,
            n_obj,
            weights,
            useInitialPopulation=True,
            initialPopulation=initPopulation,
            initialObjvValues=initialObjvTargets
        )
        LSADE.optimizerStep()

        features = np.loadtxt("LSADEFeatures.txt")
        np.savetxt(
            f"LSADEFeatures{function}{scalarisingFunction.__name__}.txt", features
        )

        scalarisedTargets = np.loadtxt("LSADEScalarisedTargets.txt")
        np.savetxt(
            f"LSADEScalarisedTargets{function}{scalarisingFunction.__name__}.txt",
            scalarisedTargets,
        )

        objtTargets = np.loadtxt("LSADEObjectiveTargets.txt")
        np.savetxt(
            f"LSADEObjtvTargets{function}{scalarisingFunction.__name__}.txt",
            objtTargets,
        )

        PSO = opt.TS_DDEO(
            bounds,
            20,
            problem,
            scalarisingFunction,
            n_obj,
            weights,
            useInitialPopulation=True,
            initialPopulation=initPopulation,
            initialObjvValues=initialObjvTargets
        )
        PSO.stage1()
        PSO.stage2()

        features = np.loadtxt("TSDDEOFeatures.txt")
        np.savetxt(
            f"TSDDEOFeatures{function}{scalarisingFunction.__name__}.txt", features
        )

        scalarisedTargets = np.loadtxt("TSDDEOTargets.txt")
        np.savetxt(
            f"TSDDEOScalarisedTargets{function}{scalarisingFunction.__name__}.txt",
            scalarisedTargets,
        )

        objtTargets = np.loadtxt("TSDDEOObjectiveTargets.txt")
        np.savetxt(
            f"TSDDEOObjtvTargets{function}{scalarisingFunction.__name__}.txt",
            objtTargets,
        )

        bayesianRun = opt.BOZeroMax(
            bounds,
            15,
            problem,
            scalarisingFunction,
            n_obj,
            weights,
            useInitialPopulation=True,
            initialPopulation=initPopulation,
            initialObjvValues=initialObjvTargets
        )
        bayesianRun.runOptimiser()

        features = np.loadtxt("BOFeatures.txt")
        np.savetxt(
            f"BOFeatures{function}{scalarisingFunction.__name__}.txt", features
        )

        scalarisedTargets = np.loadtxt("BOScalarisedTargets.txt")
        np.savetxt(
            f"BOScalarisedTargets{function}{scalarisingFunction.__name__}.txt",
            scalarisedTargets,
        )

        objtTargets = np.loadtxt("BOObjectiveTargets.txt")
        np.savetxt(
            f"BOObjtvTargets{function}{scalarisingFunction.__name__}.txt",
            objtTargets,
        )

        ESA = opt.ESA(
            bounds,
            20,
            10,
            0.25,
            problem,
            scalarisingFunction,
            n_obj,
            weights,
            0.9,
            80,
            useInitialPopulation=True,
            initialPopulation=initPopulation,
            initialObjvValues=initialObjvTargets
        )
        ESA.mainMenu(initialAction=2)

        features = np.loadtxt("ESAFeatures.txt")
        np.savetxt(
            f"ESAFeatures{function}{scalarisingFunction.__name__}.txt", features
        )

        scalarisedTargets = np.loadtxt("ESAScalarisedTargets.txt")
        np.savetxt(
            f"ESAScalarisedTargets{function}{scalarisingFunction.__name__}.txt",
            scalarisedTargets,
        )

        objtTargets = np.loadtxt("ESAObjectiveTargets.txt")
        np.savetxt(
            f"ESAObjtvTargets{function}{scalarisingFunction.__name__}.txt",
            objtTargets,
        )
