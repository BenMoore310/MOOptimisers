import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from itertools import product
from scipy.stats import qmc  # For Latin Hypercube Sampling
import torch
import gpytorch
import random
from PIL import Image
from datetime import datetime
#import scienceplots
from scipy.stats import norm
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


# def MOobjective_function(vec, currentFunction, nObjectives, scalarisingFunction, zbests, weights):
#     """Objective function wrapper for optimization.
#     Args:
#         vec (np.ndarray): A vector representing candidate solution (x, y).
#         make zbests an array of shape (0, n_objs)
#     Returns:
#         float: Fitness value of the solution.
#     """

#     objectiveArray = np.empty((1,nObjectives))
#     print('updated')
#     x, y = vec #this gives the separate objective values to be returned for pareto plotting purposes
#     objectiveArray[0] = currentFunction(x,y)

#     #now scalarise the objectives
#     scalarisedObjective, zbests = scalarisingFunction(objectiveArray, zbests, weights)


#     return objectiveArray, scalarisedObjective, zbests


# OBJECTIVE FUNCTIONS


def ackley_function(x, y, a=20, b=0.2, c=2 * np.pi):
    term1 = -a * np.exp(-b * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(c * x) + np.cos(c * y)))
    return term1 + term2 + a + np.exp(1)


def binhAndKorn(x, y):
    f1 = (4 * (x**2)) + (4 * (y**2))
    f2 = (x - 5) ** 2 + (y - 5) ** 2

    if (x-5)**2 + y**2 <= 25 and (x-8)**2 + (y+3)**2 >= 7.7:

        return [f1, f2]
    else:
        # print('here', x, y)
        return [np.nan, np.nan]


def chankongHaimes(x, y):
    f1 = 2 + (x - 2) ** 2 + (y - 1) ** 2
    f2 = (9 * x) - (y - 1) ** 2

    # return [f1, f2]

    if x**2 + y**2 <= 225 and x - 3*y + 10 <= 0:

        return [f1, f2]
    else:
        # print('here', x, y)
        return [np.nan, np.nan]


# def fonsecaFleming(vec):
#     f1InnerTerm = 0
#     f2InnerTerm = 0
#     for i in range(0, len(vec)):
#         f1InnerTerm += (vec[i] - (1 / (i + 1) ** 0.5)) ** 2
#         f2InnerTerm += (vec[i] + (1 / (i + 1) ** 0.5)) ** 2

#     f1 = 1 - np.exp(-1 * f1InnerTerm)
#     f2 = 1 - np.exp(-1 * f2InnerTerm)

#     return [f1, f2]

def fonsecaFleming(x):
    """
    Evaluate the Fonseca-Fleming function for a given input vector x.

    Parameters:
        x (numpy.ndarray): Input vector of decision variables, typically in [-4, 4] for each dimension.

    Returns:
        tuple: A tuple containing the two objective values (f1, f2).
    """
    x = np.asarray(x)  # Ensure x is a numpy array

    # First objective function
    f1 = 1 - np.exp(-np.sum((x - 1/np.sqrt(len(x)))**2))

    # Second objective function
    f2 = 1 - np.exp(-np.sum((x + 1/np.sqrt(len(x)))**2))

    return [f1, f2]


def ctp1(x, y):
    f1 = x
    f2 = (1 + y) * np.exp(-1 * (x / (1 + y)))

    if f2/(0.858*np.exp(-0.541*f1))>=1 and f2/(0.728*np.exp(-0.295*f1))>=1:

        return [f1, f2]
    else:
        # print('here', x, y)
        return [np.nan, np.nan]


def constrEx(x, y):
    f1 = x
    f2 = (1 + y) / x


    if y + 9*x >= 6 and -1*y + 9*x >=1:

        return [f1, f2]
    else:
        # print('here', x, y)
        return [np.nan, np.nan]


def testFunction4(x, y):
    f1 = x**2 - y
    f2 = -0.5 * x - y - 1

    if 6.5-(x/6)-y >= 0 and 7.5-0.5*x-y >=0 and 30 -5*x - y >=0:

        return [f1, f2]
    else:
        # print('here', x, y)
        return [np.nan, np.nan]


# SCALARISING FUNCTIONS


def chebyshev(objs, z, w):
    # print(objs.shape, z.shape)
    # replace utopian value for each objective with new value if better
    # for i in range(0, len(objs)+1):
    #     if objs[0,i] < z[0,i]:
    #         z[0,i] = objs[0,i]
    # print(objs, objs.shape)
    # compute chebyshev
    objSums = np.empty((2,))
    # print(objSums.shape)
    for i in range(0, len(objs)):
        # print(i)
        objSums[i] = w[i] * abs(objs[i] - z[i])
        # print(objSums[i])
    g = np.max(objSums)

    # return zBests array so it can be updated after each function evaluation
    return g


def weightedSum(objs, z, w):
    objSum = 0

    for i in range(0, len(objs)):
        objSum += w[i] * objs[i]

    return objSum


def EWC(objs, z, w):
    # from chugh: p=100
    p = 1

    objSum = 0

    for i in range(0, len(objs)):
        objSum += np.exp(((p * w[i]) - 1)) * np.exp((p * objs[i]))

    return objSum


def weightedPower(objs, z, w):
    # from chugh: p=3
    p = 3

    objSum = 0

    for i in range(0, len(objs)):
        objSum += w[i] * (objs[i]) ** p

    return objSum


def weightedNorm(objs, z, w):
    # from chugh: p=0.5
    p = 0.5

    objSum = 0

    for i in range(0, len(objs)):
        objSum += w[i] * (np.abs(objs[i]) ** p)

    g = objSum ** (1 / p)

    return g


def augmentedChebychev(objs, z, w):
    # from chugh =>  alpha = 0.0001
    alpha = 0.0001

    objSums = np.empty((2,))

    # calculate augmented term
    augTerm = 0

    for i in range(0, len(objs)):
        augTerm += np.abs(objs[i] - z[i])

    for i in range(0, len(objs)):
        objSums[i] = w[i] * abs(objs[i] - z[i])

    g = np.max(objSums) + (alpha * augTerm)

    return g


def modifiedChebychev(objs, z, w):
    # from chugh =>  alpha = 0.0001
    alpha = 0.0001

    objSums = np.empty((2,))

    # calculate augmented term
    augTerm = 0

    for i in range(0, len(objs)):
        augTerm += np.abs(objs[i] - z[i])

    for i in range(0, len(objs)):
        objSums[i] = (w[i] * abs(objs[i] - z[i])) + (alpha * augTerm)

    g = np.max(objSums)

    return g


def PBI(objs, z, w, theta=5.0):
    # from chugh: theta = 5
    # theta = 5

    # calculating values for d1

    d1term1 = np.linalg.norm(np.dot((np.transpose((objs - z))), w))

    #d1 term calculation changed from norm to abs:
    # d1term1 = np.abs(np.dot((np.transpose((objs - z))), w))

    d1term2 = np.linalg.norm(w)

    d1 = d1term1 / d1term2

    d2 = np.linalg.norm(objs - (d1 * (w / np.linalg.norm(w)) + z))

    g = d1 + theta * d2

    return g


def PAPBI(objs, z, w, currentGen, maxGen, thetaStore={"previousTheta": 1.0}):
    if random.random() > currentGen / maxGen:
        thetaK = [0, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 5.0]

        xK = np.empty((8, np.ma.size(objs, axis=1)))
        xKperpDistance = np.empty((8,))

        for i in range(0, len(thetaK)):
            scalarisedArray = np.empty(len(objs))

            for j in range(0, len(objs)):
                scalarisedArray[j] = PBI(objs[j], z, w, theta=thetaK[i])
            minIndex = np.argmin(scalarisedArray)

            xK[i] = objs[minIndex]

            # calculate perp. distance of xK[i] to weight vector
            xKperpDistance[i] = np.linalg.norm(
                xK[i] - ((np.dot(xK[i], w) / (np.linalg.norm(w)) ** 2) * w)
            )
            # projection = (np.dot(xK[i], w) / np.linalg.norm(w)**2)
            # xKperpDistance[i] = np.linalg.norm(xK[i] - projection)

        minIndex = np.argmin(xKperpDistance)

        selectedTheta = thetaK[minIndex]
        print("previousTheta = ", thetaStore["previousTheta"])

        thetaStore["previousTheta"] = selectedTheta

    else:
        selectedTheta = thetaStore["previousTheta"]

    print("selectedTheta = ", selectedTheta)

    # run PBI with optimised theta argument
    scalarisedArray = np.empty(len(objs))
    for i in range(0, len(objs)):
        scalarisedArray[i] = PBI(objs[i], z, w, theta=selectedTheta)

    return scalarisedArray


#method and functions for HypI hypervolume improvement scalarising function

def computeParetoShells(X):
    remaining = X.copy()
    shells = []

    while len(remaining) > 0:
        # Compute non-dominated front
        nds = NonDominatedSorting().do(remaining, only_non_dominated_front=True)
        paretoFront = remaining[nds]
        shells.append(paretoFront)

        # Remove selected Pareto front from remaining points
        remaining = np.delete(remaining, nds, axis=0)

    return shells

#calculate worst set of objective values in current dataset (nadir)



# Step 2: Compute hypervolume indicator
def computeHypervolume(X, ref_point):
    hv = HV(ref_point)
    return hv.do(X)

# Step 3: Compute hypervolume improvement
def hypervolumeImprovement(x, ref_point, paretoShells):
    # Find the first Pareto shell that does not dominate x
    # pareto_shells = compute_pareto_shells(X)

    pareto_k = None
    # for shell in paretoShells:
    #     if not np.any(np.all(shell <= x, axis=1)):
    #         pareto_k = shell
    #         break
    for i in range(0, len(paretoShells)-1):
        if not np.any(np.all(paretoShells[i] <= x, axis=1)):
            pareto_k = paretoShells[i+1]

    # Check if pareto_k was assigned; if not, default to the last shell
    if pareto_k is None:
        pareto_k = paretoShells[-1]

    # Compute hypervolume with x added to the shell
    hv_before = computeHypervolume(pareto_k, ref_point)
    hv_after = computeHypervolume(np.vstack([pareto_k, x]), ref_point)

    return hv_after - hv_before

def HypI(objs):

    # print('before function - ', objs)


    #compute nadir vector (to be the reference vector)
    #problem occurs if nan value encountered - refVector set as nan??
    #does setting refVector to 1,1 (as values are normalised) fix this?
    # refVector = np.max(objs, axis=0)

    refVector = np.array((1,1))

    # print('refVector =', refVector)

    paretoShells = computeParetoShells(objs)

    # np.savetxt('paretoShells.txt', np.array(paretoShells))

    scalarisedValues = np.empty(len(objs))

    for i in range(0, len(objs)):
        scalarisedValues[i] = hypervolumeImprovement(objs[i], refVector, paretoShells)
    # print('after function - ', (1-scalarisedValues))
    # return (1-scalarisedValues), paretoShells
    return (1-scalarisedValues)

