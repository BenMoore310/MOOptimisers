import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import itertools
from itertools import product
from scipy.stats import qmc  # For Latin Hypercube Sampling
import torch
import gpytorch
import random
from PIL import Image
from datetime import datetime

# import scienceplots
from scipy.stats import norm
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import subprocess
import tempfile
import shutil
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from pymoo.problems import get_problem


# FUNCTIONS FOR RANDOMISING WEIGHT VECTORS A LA PAREGO

def generateWeightVectors(k, s):
    """
    Generate the set of weight vectors Λ for k objectives and granularity s.
    
    Parameters:
        k (int): Number of objectives (dimensions of the weight vector).
        s (int): Granularity (number of divisions for each weight component).
    
    Returns:
        list of tuples: A list containing all possible weight vectors in Λ.
    """
    # Generate all combinations of s+1 integers that sum to s
    levels = range(s + 1)  # Possible values for each λ_j: {0, 1/s, 2/s, ..., 1}
    # print(levels)
    combinations = [comb for comb in itertools.product(levels, repeat=k-1)]

    
    # Convert combinations into weight vectors (ensure they sum to 1)
    weight_vectors = []
    for comb in combinations:
        lam = np.zeros(k)
        # print(lam, comb)
        lam[:k-1] = np.array(comb) / s
        lam[-1] = 1 - sum(lam[:-1])  # Ensure sum of λ equals 1
        if all(0 <= val <= 1 for val in lam):
            weight_vectors.append(tuple(lam))
    
    return weight_vectors


def sampleWeightVector(weight_vectors):
    """
    Randomly sample a weight vector from the generated set Λ.
    
    Parameters:
        weight_vectors (list of tuples): The set of all possible weight vectors.
    
    Returns:
        tuple: A randomly selected weight vector.
    """

    # print(len(weight_vectors))
    # print(np.linspace(0, len(weight_vectors), len(weight_vectors)+1))

    idx = int(np.random.choice(np.linspace(0, len(weight_vectors)-1, len(weight_vectors))))
    # print(idx)
    return weight_vectors[idx]



# SANDTRAP OPTIMISATION FUNCTIONS


def updateTurbulence(file, newValue):

    file["RAS"]["sigmaEps"] = f"{newValue}"

    file.writeFile()
    print(f"Turbulence Constant updated to {newValue}")


def updateConc(file, newValue):

    file["boundaryField"]["inletWater"]["value"] = f"uniform {newValue}"

    file.writeFile()
    print(f"Inlet concentration updated to {newValue}")


def updateRoughness(file, newValue):
    for surface in file["boundaryField"]:
        # print(surface)
        for entry in file["boundaryField"][surface]:
            # print(entry)
            if "Ks" in entry:
                file["boundaryField"][surface]["Ks"] = f"uniform {newValue}"
                file["boundaryField"][surface]["roughnessHeight"] = f"{newValue}"

    file.writeFile()
    print(f"Wall roughness updated to {newValue}")


def updateMaxCellSize(file, newValue):
    file["maxCellSize"] = f"{newValue}"

    file.writeFile()
    print(f"Max Cell Size updated to {newValue}")


def updateTurbVisc(file, newValue):
    file["boundaryField"]["inletWater"]["value"] = f"uniform {newValue}"

    file.writeFile()
    print(f"Inlet Turbulent Viscosity updated to {newValue}")


def calculateAverageDisplacement(labDataConc, simDataConc, probeHeights):

    simDataIndices = []

    data_gaps = []

    for i in probeHeights:
        # print(np.searchsorted(concData[:,0], i))
        simDataIndices.append(np.searchsorted(simDataConc[:, 0], i))

    for i in range(0, 4, 1):
        gap = abs(labDataConc[i] - (simDataConc[simDataIndices[i], 1]) * 1e6)
        # print(probeAConcs[i], (concData[probeAIndices[i],1])*1e6)
        data_gaps.append(gap)
    averageDisplacement = np.average(data_gaps)
    print(averageDisplacement)

    return averageDisplacement


def sandTrap(features):

    probeAHeights = [0.099681, 0.40639, 0.8179, 1.2831]
    probeAConcs = [220.6415, 216.2483, 217.2245, 163.5286]

    probeBHeights = [0.05063, 0.3063, 0.6152, 1.2329]
    probeBConcs = [141.037, 129.8962, 146.2518, 97.4222]

    probeCHeights = [0.1051, 0.2078, 0.5158, 1.1821]
    probeCConcs = [106.3158, 109.1729, 81.9549, 50.5263]

    sandTrapDir = "/home/bm424/Projects/MOOptimisers/sandTrapCaseDir/"

    tmpdir = tempfile.mkdtemp(dir="/home/bm424/Projects/MOOptimisers/tempDirs/")

    shutil.copytree(sandTrapDir, tmpdir, dirs_exist_ok=True)

    turbulenceProperties = ParsedParameterFile(
        tmpdir + "/constant/turbulenceProperties"
    )
    conc01 = ParsedParameterFile(tmpdir + "/0/Conc01")
    conc02 = ParsedParameterFile(tmpdir + "/0/Conc02")
    conc03 = ParsedParameterFile(tmpdir + "/0/Conc03")
    conc045 = ParsedParameterFile(tmpdir + "/0/Conc045")
    # need 4 concs
    eddyvisc = ParsedParameterFile(tmpdir + "/0/eddyvisc")
    kineticenergy = ParsedParameterFile(tmpdir + "/0/kineticenergy")
    nut = ParsedParameterFile(tmpdir + "/0/nut")
    meshDict = ParsedParameterFile(tmpdir + "/system/meshDict")

    # order is: inletConc, Wall Roughness, maxCellSize, sigmaTurbConstant, turbVisc
    updateConc(conc01, features[0])
    updateConc(conc02, features[0])
    updateConc(conc03, features[0])
    updateConc(conc045, features[0])

    updateRoughness(eddyvisc, features[1])
    updateRoughness(kineticenergy, features[1])
    updateRoughness(nut, features[1])

    updateMaxCellSize(meshDict, features[2])

    updateTurbulence(turbulenceProperties, features[3])

    updateTurbVisc(eddyvisc, features[4])

    subprocess.run(["cartesianMesh"], shell=False, cwd=tmpdir, check=False)
    subprocess.run(["renumberMesh"], shell=False, cwd=tmpdir, check=False)
    subprocess.run(["decomposePar"], shell=False, cwd=tmpdir, check=False)

    # subprocess.run(["mpirun", "-np", "10", "sediDriftFoam", "-parallel", ">", "tempSandTrapLog"], cwd=tmpdir, check=True)

    with open("tempSandTrapLog", "w") as log_file:
        subprocess.run(
            ["mpirun", "-np", "10", "sediDriftFoam", "-parallel"],
            cwd=tmpdir,
            check=True,
            stdout=log_file
        )

    subprocess.run(["reconstructPar"], shell=False, cwd=tmpdir, check=False)

    

    subprocess.run(
        ["postProcess", "-func", "sampleDict"],
        cwd=tmpdir,
        shell=False,
        check=True,
        capture_output=False,
    )

    concDataA = np.loadtxt(
        tmpdir
        + "/postProcessing/sampleDict/300/point_a_Conc01_Conc02_Conc03_Conc045.xy"
    )
    concDataB = np.loadtxt(
        tmpdir
        + "/postProcessing/sampleDict/300/point_b_Conc01_Conc02_Conc03_Conc045.xy"
    )
    concDataC = np.loadtxt(
        tmpdir
        + "/postProcessing/sampleDict/300/point_c_Conc01_Conc02_Conc03_Conc045.xy"
    )

    mergedConcsA = np.zeros((154, 2))
    mergedConcsB = np.zeros((154, 2))
    mergedConcsC = np.zeros((154, 2))

    for i in range(0, 154):
        mergedConcsA[i, 0] = concDataA[i, 0]
        mergedConcsA[i, 1] = (
            concDataA[i, 1] + concDataA[i, 2] + concDataA[i, 3] + concDataA[i, 4]
        ) / 4

    for i in range(0, 154):
        mergedConcsB[i, 0] = concDataB[i, 0]
        mergedConcsB[i, 1] = (
            concDataB[i, 1] + concDataB[i, 2] + concDataB[i, 3] + concDataB[i, 4]
        ) / 4

    for i in range(0, 154):
        mergedConcsC[i, 0] = concDataC[i, 0]
        mergedConcsC[i, 1] = (
            concDataC[i, 1] + concDataC[i, 2] + concDataC[i, 3] + concDataC[i, 4]
        ) / 4

    avgDispA = calculateAverageDisplacement(probeAConcs, mergedConcsA, probeAHeights)
    avgDispB = calculateAverageDisplacement(probeBConcs, mergedConcsB, probeBHeights)
    avgDispC = calculateAverageDisplacement(probeCConcs, mergedConcsC, probeCHeights)

    shutil.rmtree(tmpdir)

    return [avgDispA, avgDispB, avgDispC]


# BENCHMARKING OBJECTIVE FUNCTIONS

# Benchmarking functions have been updated - they now return a third objective value based on whether the objective values are valid or not - 
# returns a 0 if objectives are valid, returns 1 if they are invalid. 


# PYMOO problems

def getPyMooProblem(function, n_var, n_obj):

    problem = get_problem(function, n_var=n_var, n_obj=n_obj)

    bl = problem.xl
    bu = problem.xu
    bounds = []

    for i in range(n_var):
        bounds.append([bl[i], bu[i]])

    return problem, np.array(bounds)

def evalPyMooProblem(function, vec):

    result = function.evaluate(vec)
    # result = np.append(result, [0])

    return result





def ackley_function(x, y, a=20, b=0.2, c=2 * np.pi):
    term1 = -a * np.exp(-b * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(c * x) + np.cos(c * y)))
    return term1 + term2 + a + np.exp(1)


def binhAndKorn(vec):

    x, y = vec

    f1 = (4 * (x**2)) + (4 * (y**2))
    f2 = (x - 5) ** 2 + (y - 5) ** 2

    if (x - 5) ** 2 + y**2 <= 25 and (x - 8) ** 2 + (y + 3) ** 2 >= 7.7:

        return [f1, f2, 0]
    else:
        # print('here', x, y)
        return [f1, f2, 1]


def chankongHaimes(vec):
    x, y = vec
    f1 = 2 + (x - 2) ** 2 + (y - 1) ** 2
    f2 = (9 * x) - (y - 1) ** 2

    # return [f1, f2]

    if x**2 + y**2 <= 225 and x - 3 * y + 10 <= 0:

        return [f1, f2, 0]
    else:
        # print('here', x, y)
        return [f1, f2, 2]


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
    f1 = 1 - np.exp(-np.sum((x - 1 / np.sqrt(len(x))) ** 2))

    # Second objective function
    f2 = 1 - np.exp(-np.sum((x + 1 / np.sqrt(len(x))) ** 2))

    return [f1, f2, 0]


def ctp1(vec):
    x, y = vec
    f1 = x
    f2 = (1 + y) * np.exp(-1 * (x / (1 + y)))

    if (
        f2 / (0.858 * np.exp(-0.541 * f1)) >= 1
        and f2 / (0.728 * np.exp(-0.295 * f1)) >= 1
    ):

        return [f1, f2, 0]
    else:
        # print('here', x, y)
        return [f1, f2, 1]


def constrEx(vec):
    x, y = vec
    f1 = x
    f2 = (1 + y) / x

    if y + 9 * x >= 6 and -1 * y + 9 * x >= 1:

        return [f1, f2, 0]
    else:
        # print('here', x, y)
        return [f1, f2, 1]


def testFunction4(vec):
    x, y = vec
    f1 = x**2 - y
    f2 = -0.5 * x - y - 1

    if 6.5 - (x / 6) - y >= 0 and 7.5 - 0.5 * x - y >= 0 and 30 - 5 * x - y >= 0:

        return [f1, f2, 0]
    else:
        # print('here', x, y)
        return [f1, f2, 1]



def viennetFunction(vector):
 
    x, y = vector

    # Objective function f1
    f1 = (0.5 * (x**2 + y**2)) + np.sin(x**2 + y**2)

    # Objective function f2
    f2 = ((((3 * x) - (2 * y) + 4)**2) / 8) + (((x - y + 1)**2) / 27) + 15

    # Objective function f3
    f3 = (1 / (x**2 + y**2 + 1)) - 1.1 * np.exp(-(x**2 + y**2))

    return [f1, f2, f3, 0]

# SCALARISING FUNCTIONS


def chebyshev(objs, z, w):
    # print(objs.shape, z.shape)
    # replace utopian value for each objective with new value if better
    # for i in range(0, len(objs)+1):
    #     if objs[0,i] < z[0,i]:
    #         z[0,i] = objs[0,i]
    # print(objs, objs.shape)
    # compute chebyshev
    objSums = np.empty((len(objs),))
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

    objSums = np.empty((len(objs),))

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

    objSums = np.empty((len(objs),))

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

    # d1 term calculation changed from norm to abs:
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


# method and functions for HypI hypervolume improvement scalarising function


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


# calculate worst set of objective values in current dataset (nadir)


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
    for i in range(0, len(paretoShells) - 1):
        if not np.any(np.all(paretoShells[i] <= x, axis=1)):
            pareto_k = paretoShells[i + 1]

    # Check if pareto_k was assigned; if not, default to the last shell
    if pareto_k is None:
        pareto_k = paretoShells[-1]

    # Compute hypervolume with x added to the shell
    # print(pareto_k, ref_point)
    hv_before = computeHypervolume(pareto_k, ref_point)
    hv_after = computeHypervolume(np.vstack([pareto_k, x]), ref_point)
    # print('Hypervolumes:')
    # print(hv_after - hv_before)
    return hv_after - hv_before


def HypI(objs):

    # print('before function - ', objs)

    # compute nadir vector (to be the reference vector)
    # problem occurs if nan value encountered - refVector set as nan??
    # does setting refVector to 1,1 (as values are normalised) fix this?
    # refVector = np.max(objs, axis=0)

    # Filter the rows where the last column is 0

    #TODO need some handling where the below option is turned
    #on when im using my functions and not when using pyMoo

    # valid_rows = objs[objs[:, -1] == 0]

    # max_values = np.max(valid_rows[:, :-1], axis=0)

    # refVector = np.append(max_values, 0)
    refVector = np.ones((len(objs[-1],)))

    # print('refVector =', refVector)

    paretoShells = computeParetoShells(objs)

    # print(paretoShells)

    # np.savetxt('paretoShells.txt', np.array(paretoShells))

    scalarisedValues = np.empty(len(objs))

    for i in range(0, len(objs)):
        scalarisedValues[i] = hypervolumeImprovement(objs[i], refVector, paretoShells)
    # print('after function - ', (1-scalarisedValues))
    # return (1-scalarisedValues), paretoShells
    return 1 - scalarisedValues


def APD(F, ref_dirs, FE, FE_max, k, alpha):
    """
    Compute the Angle Penalized Distance (APD) for a set of solutions.
    
    Parameters:
        F (numpy.ndarray): Objective values of solutions (m x k matrix).
        ref_dirs (numpy.ndarray): Reference directions (n x k matrix).
        FE (int): Current number of function evaluations.
        FE_max (int): Maximum number of function evaluations.
        alpha (float): Penalty exponent.
        k (float): Penalty scaling factor.
    
    Returns:
        numpy.ndarray: APD values for each solution.
    """
    # Normalize the objective values
    F_norm = F / np.linalg.norm(F, axis=1, keepdims=True)

    # Compute cosine similarity and find the nearest reference direction
    cosine = np.dot(F_norm, ref_dirs.T)
    nearest_idx = np.argmax(cosine, axis=1)  # Index of the nearest reference direction
    nearest_ref = ref_dirs[nearest_idx]      # Nearest reference directions for each solution

    # Compute distances (Euclidean norm) between F and their nearest reference vector
    distances = np.linalg.norm(F - nearest_ref, axis=1)

    # Compute the angle (1 - cosine) between each solution and its nearest reference vector
    angles = 1 - cosine[np.arange(len(F)), nearest_idx]

    # Calculate gamma (minimum angle between the selected reference vector and other reference vectors)
    gamma_values = []
    for idx in nearest_idx:
        # Compute angles (cosine values) between the chosen reference vector and all other reference vectors
        selected_ref = ref_dirs[idx]  # The reference vector assigned to the current solution
        other_refs = np.delete(ref_dirs, idx, axis=0)  # Remove the selected reference vector
        cosine_to_others = np.dot(selected_ref, other_refs.T)
        gamma = 1 - np.max(cosine_to_others)  # Smallest angle (highest cosine similarity)
        gamma_values.append(gamma)
    gamma_values = np.array(gamma_values)

    # Calculate the progress term FE / FE_max
    progress = FE / FE_max
    penalty_factor = k * (progress ** alpha)

    # Compute the APD value for each solution
    apd = distances * (1 + penalty_factor * (angles / gamma_values))
    return apd
