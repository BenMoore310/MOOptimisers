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

dtlzProblems = ['dtlz1','dtlz2']

n_var = 4
n_obj = 2

scalarisingList = [
    func.HypI,
    func.chebyshev,
    # func.weightedSum,
    func.EWC,
    func.weightedPower,
    # func.weightedNorm,
    # func.augmentedChebychev,
    # func.modifiedChebychev,
    func.PBI,
    # func.PAPBI,
]

weights = np.array((0.5, 0.5))


for function in dtlzProblems:

    print('current function  = ', function)

    problem, bounds = func.getPyMooProblem(function, n_var, n_obj)

    initSampleSize = 50
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


        bayesianRun = opt.bayesianOptimiser(
            bounds,
            initSampleSize,
            problem,
            scalarisingFunction,
            n_obj,
            weights,
            useInitialPopulation=True,
            initialPopulation=initPopulation,
            initialObjvValues=initialObjvTargets,
            maxFE=100
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
