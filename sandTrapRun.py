import numpy as np 
import matplotlib.pyplot as plt
import math
import torch
import gpytorch
import PyFoam
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from pyDOE import lhs
from multiprocessing import Pool
import subprocess
import tempfile
import shutil
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler





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
        simDataIndices.append(np.searchsorted(simDataConc[:,0], i))

    for i in range(0,4,1):
        gap = abs(labDataConc[i]-(simDataConc[simDataIndices[i],1])*1e6)
        # print(probeAConcs[i], (concData[probeAIndices[i],1])*1e6)
        data_gaps.append(gap)
    averageDisplacement = np.average(data_gaps)
    print(averageDisplacement)
    
    return averageDisplacement 


def simulate(features):

    probeAHeights = [0.099681, 0.40639, 0.8179, 1.2831]
    probeAConcs = [220.6415, 216.2483, 217.2245, 163.5286]

    probeBHeights = [0.05063, 0.3063, 0.6152, 1.2329]
    probeBConcs = [141.037, 129.8962, 146.2518, 97.4222]

    probeCHeights = [0.1051, 0.2078, 0.5158, 1.1821]
    probeCConcs = [106.3158, 109.1729, 81.9549, 50.5263]

    sandTrapDir = "/home/bm424/OpenFOAM/bm424-v2312/run/sandTrap/monoSurrogateSetup/sandTrap4ConcsOrig"

    tmpdir = tempfile.mkdtemp(dir = "/home/bm424/OpenFOAM/bm424-v2312/run/sandTrap/monoSurrogateSetup/")

    shutil.copytree(sandTrapDir, tmpdir, dirs_exist_ok=True)

    turbulenceProperties = ParsedParameterFile(tmpdir + "/constant/turbulenceProperties")
    conc01 = ParsedParameterFile(tmpdir + "/0/Conc01")
    conc02 = ParsedParameterFile(tmpdir + "/0/Conc02")
    conc03 = ParsedParameterFile(tmpdir + "/0/Conc03")
    conc045 = ParsedParameterFile(tmpdir + "/0/Conc045")
    #need 4 concs
    eddyvisc = ParsedParameterFile(tmpdir + "/0/eddyvisc")
    kineticenergy = ParsedParameterFile(tmpdir + "/0/kineticenergy")
    nut = ParsedParameterFile(tmpdir + "/0/nut")
    meshDict = ParsedParameterFile(tmpdir + "/system/meshDict")

    # order is: Wall Roughness, inletConc, maxCellSize, sigmaTurbConstant, turbVisc

    updateRoughness(eddyvisc, features[1])
    updateRoughness(kineticenergy, features[1])
    updateRoughness(nut, features[1])

    updateConc(conc01, features[0])
    updateConc(conc02, features[0])
    updateConc(conc03, features[0])
    updateConc(conc045, features[0])

    updateMaxCellSize(meshDict, features[2])

    updateTurbulence(turbulenceProperties, features[3])

    updateTurbVisc(eddyvisc, features[4])


    subprocess.run(["cartesianMesh"], shell=False, cwd=tmpdir, check=False)

    subprocess.run(["sediDriftFoam"], cwd=tmpdir, check=True)

    subprocess.run(['postProcess', '-func', 'sampleDict'], cwd=tmpdir, shell=False, check=True, capture_output=False)

    concDataA = np.loadtxt(tmpdir + '/postProcessing/sampleDict/500/point_a_Conc01_Conc02_Conc03_Conc045.xy')
    concDataB = np.loadtxt(tmpdir + '/postProcessing/sampleDict/500/point_b_Conc01_Conc02_Conc03_Conc045.xy')
    concDataC = np.loadtxt(tmpdir + '/postProcessing/sampleDict/500/point_c_Conc01_Conc02_Conc03_Conc045.xy')

    mergedConcsA = np.zeros((154,2))
    mergedConcsB = np.zeros((154,2))
    mergedConcsC = np.zeros((154,2))

    for i in range(0,154):
        mergedConcsA[i,0] = concDataA[i,0]
        mergedConcsA[i,1] = (concDataA[i,1] + concDataA[i,2] + concDataA[i,3] + concDataA[i,4])/4

    for i in range(0,154):
        mergedConcsB[i,0] = concDataB[i,0]
        mergedConcsB[i,1] = (concDataB[i,1] + concDataB[i,2] + concDataB[i,3] + concDataB[i,4])/4

    for i in range(0,154):
        mergedConcsC[i,0] = concDataC[i,0]
        mergedConcsC[i,1] = (concDataC[i,1] + concDataC[i,2] + concDataC[i,3] + concDataC[i,4])/4


    avgDispA = calculateAverageDisplacement(probeAConcs, mergedConcsA, probeAHeights)
    avgDispB = calculateAverageDisplacement(probeBConcs, mergedConcsB, probeBHeights)
    avgDispC = calculateAverageDisplacement(probeCConcs, mergedConcsC, probeCHeights)

    shutil.rmtree(tmpdir) 

    
    return avgDispA, avgDispB, avgDispC