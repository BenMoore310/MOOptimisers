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
import functionBank as func
# import scienceplots
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
import inspect
from pymoo.core.problem import Problem  # PyMoo Problem base class
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.problems import get_problem


def MOobjective_function(vec, currentFunction, nObjectives):
    """Objective function wrapper for optimization.
    Args:
        vec (np.ndarray): A vector representing candidate solution (x, y).
        make zbests an array of shape (0, n_objs)
    Returns:
        float: Fitness value of the solution.
    """
    objectiveArray = np.empty((1, nObjectives))

    if inspect.isfunction(func):  # Checks if it's a user-defined function

        objectiveArray[0] = currentFunction(vec)

    else:   
        objectiveArray[0] = func.evalPyMooProblem(currentFunction, vec)

    return objectiveArray
def scalariseValues(
    scalarisingFunction, objectiveArray, zBests, weights, currentGen, maxGen
):
    scaler = MinMaxScaler(feature_range=(0, 1))

    objsNormalised = scaler.fit_transform(objectiveArray)

    # print(objsNormalised)

    weightVector = func.sampleWeightVector(weights)
    print(f'New weight vector = {weightVector}')

    z0 = np.zeros_like(zBests)

    if scalarisingFunction == func.PAPBI:
        print("using PAPBI!")
        scalarisedArray = scalarisingFunction(
            objsNormalised, z0, weightVector, currentGen, maxGen
        )
    elif scalarisingFunction == func.HypI:
        # print('using HypI')
        scalarisedArray = scalarisingFunction(objsNormalised)
    elif scalarisingFunction == func.APD:
        # print('1')
        ref_dirs = get_reference_directions("das-dennis", len(objectiveArray[-1]), n_partitions=len(objectiveArray[-1])*3)

        # not using normalised values here, as it does not mix well with reference directions

        scalarisedArray =scalarisingFunction(objectiveArray, ref_dirs, currentGen, maxGen, len(objectiveArray[-1]), alpha = 2)
        # print('2')
    else:
        scalarisedArray = np.empty(len(objectiveArray))
        for i in range(0, len(objectiveArray)):
            scalarisedArray[i] = scalarisingFunction(objsNormalised[i], z0, weightVector)

    # print(scalarisedArray)
    # unNormalisedScalarArray = scaler.inverse_transform(scalarisedArray)
    # print(unNormalisedScalarArray)

    return scalarisedArray

class BayesianDifferentialEvolution:
    def __init__(
        self,
        surrogateModel,
        bounds,
        bestTarget,
        max_generations,
        pop_size=50,
        mutation_factor=0.8,
        crossover_prob=0.7,
        method="random",
    ):
        """
        Initialize the Differential Evolution (DE) optimizer for use with BO.

        Parameters:
        bounds (list of tuple): List of (min, max) bounds for each dimension.
        pop_size (int): Number of candidate solutions in the population.
        mutation_factor (float): Scaling factor for mutation [0, 2].
        crossover_prob (float): Crossover probability [0, 1].
        max_generations (int): Maximum number of generations to evolve.
        method (str): Population initialization method ('random' or 'lhs').
        """
        self.bounds = np.array(bounds)
        self.dimensions = len(bounds)
        self.pop_size = pop_size
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.max_generations = max_generations
        self.method = method
        self.surrogateModel = surrogateModel
        self.bestTarget = bestTarget

        # Initialize population
        self.population = self.initialize_population()
        self.best_solution = None
        self.best_fitness = 0

    def initialize_population(self):
        """Initialize population using random sampling or Latin Hypercube Sampling."""
        if self.method == "lhs":
            # Latin Hypercube Sampling
            sampler = qmc.LatinHypercube(d=self.dimensions)
            sample = sampler.random(n=self.pop_size)
            population = qmc.scale(sample, self.bounds[:, 0], self.bounds[:, 1])
        else:
            # Random Sampling
            population = np.random.rand(self.pop_size, self.dimensions)
            for i in range(self.dimensions):
                population[:, i] = self.bounds[i, 0] + population[:, i] * (
                    self.bounds[i, 1] - self.bounds[i, 0]
                )

        return population

    def mutate(self, target_idx):
        """Mutation using DE/best/1 strategy."""
        # Choose three random and distinct individuals different from target_idx
        indices = [idx for idx in range(self.pop_size) if idx != target_idx]
        np.random.shuffle(indices)
        r1, r2, r3 = indices[:3]

        # Best individual in current population
        # best_idx = np.argmin([expectedImprovement(self.surrogateModel, ind, self.bestTarget, 0.01) for ind in self.population])

        predictedEI = expectedImprovement(
            self.surrogateModel, self.population, self.bestTarget, 0.01
        )

        best_idx = np.argsort(predictedEI)[-1]
        # print('in mutate function, best = ', predictedEI[best_idx], self.population[best_idx])
        best = self.population[best_idx]

        # Mutant vector: v = best + F * (r1 - r2)
        mutant = best + self.mutation_factor * (
            self.population[r1] - self.population[r2]
        )

        # Ensure mutant vector is within bounds
        mutant = np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])

        return mutant

    def crossover(self, target, mutant):
        """Crossover to create trial vector."""
        trial = np.copy(target)
        for i in range(self.dimensions):
            if np.random.rand() < self.crossover_prob or i == np.random.randint(
                self.dimensions
            ):
                # print(trial[i], mutant[i])
                # print(trial.shape, mutant.shape)
                trial[i] = mutant[i]
        return trial

    def select(self, target, trial):
        """Selection: Return the individual with the better fitness."""
        # print(trial.shape)

        trialEI = expectedImprovement(self.surrogateModel, trial, self.bestTarget, 0.01)
        targetEI = expectedImprovement(
            self.surrogateModel, target, self.bestTarget, 0.01
        )

        selectedValues = np.copy(target)

        for i in range(self.pop_size):
            if trialEI[i] > targetEI[i]:
                selectedValues[i] = trial[i]

        return selectedValues


    def optimize(self):
        """Run the Differential Evolution optimization."""


        for generation in range(self.max_generations):
            # print(generation)
            new_population = np.zeros_like(self.population)

            targetArray = np.zeros_like(self.population)
            trialArray = np.zeros_like(self.population)

            for i in range(self.pop_size):
                target = self.population[i]
                mutant = self.mutate(i)
                mutant = np.reshape(mutant, (self.dimensions,))

                trial = self.crossover(target, mutant)

                targetArray[i] = target
                trialArray[i] = trial

            new_population = self.select(targetArray, trialArray)

            # Update the population
            self.population = new_population

            # Track the best solution

            predictedEI = expectedImprovement(
                self.surrogateModel, self.population, self.bestTarget, 0.0001
            )

            # print(predictedEI)

            best_idx = np.argsort(predictedEI)[-1]

            best_fitness = np.max(predictedEI)


            if best_fitness > self.best_fitness:
                self.best_fitness = best_fitness
                self.best_solution = self.population[best_idx]
                # print('"best solution"', self.best_solution)



        return self.best_solution, self.best_fitness

def BOGPEval(model, newFeatures):
    model.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = model(torch.from_numpy(newFeatures))

    mean_pred = observed_pred.mean.numpy()
    stdDev = observed_pred.stddev.numpy()

    return mean_pred, stdDev

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, meanPrior):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        if meanPrior == "max":
            # self.mean_module = gpytorch.means.ZeroMean()
            self.mean_module = gpytorch.means.ConstantMean()
            # self.mean_module.constant = torch.nn.Parameter(torch.tensor(torch.max(train_y)))
            self.mean_module.constant.data = torch.max(train_y).clone().detach()

        else:
            # self.mean_module = gpytorch.means.ConstantMean(constant_prior=torch.max(train_y))
            self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def GPTrain(features, targets, meanPrior):
    tensorSamplesXY = torch.from_numpy(features)
    tensorSamplesZ = torch.from_numpy(targets)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(tensorSamplesXY, tensorSamplesZ, likelihood, meanPrior)
    likelihood.noise = 1e-4
    likelihood.noise_covar.raw_noise.requires_grad_(False)

    training_iter = 200
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.05
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(tensorSamplesXY)
        # Calc loss and backprop gradients
        loss = -mll(output, tensorSamplesZ)
        loss.backward()
        # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        #     i + 1, training_iter, loss.item(),
        #     model.covar_module.base_kernel.lengthscale.item(), #.kernels[0] after base_kernel if have multiple kernels
        #     model.likelihood.noise.item()
        # ))
        optimizer.step()

    return model

def expectedImprovement(currentGP, feature, bestY, epsilon):
    yPred, yStd = BOGPEval(currentGP, feature)

    # TODO check that signs are the correct way round in ei and z equations.

    z = (bestY - yPred - epsilon) / yStd
    ei = ((bestY - yPred - epsilon) * norm.cdf(z)) + yStd * norm.pdf(z)
    return ei


class bayesianOptimiser:
    def __init__(
        self,
        bounds,
        pop_size,
        objFunction,
        scalarisingFunction,
        nObjectives,
        weights,
        DEIter,
        useInitialPopulation,
        initialPopulation,
        initialObjvValues,
        maxFE
    ):
        self.globalBounds = np.array(bounds)
        self.dimensions = len(bounds)
        self.feFeatures = np.empty((0, self.dimensions))
        self.pop_size = pop_size
        self.nObjectives = nObjectives
        self.objectiveTargets = np.empty((0, self.nObjectives))
        self.scalarisedTargets = np.empty(0)
        self.x_bestSolution = 0
        self.bestEI = 100
        self.objFunction = objFunction
        self.maxFE = maxFE
        self.DEIter = DEIter

        self.scalarisingFunction = scalarisingFunction
        self.zbests = np.empty((0))
        self.weights = weights
        if useInitialPopulation == True:
            self.population = initialPopulation
            self.objectiveTargets = initialObjvValues
        else:
            self.population = self.initialisePopulation()
            self.evaluateInitialPopulation()
        self.scalariseInitialPopulation()

    def initialiseDatabase(self):
        sampler = qmc.LatinHypercube(d=self.dimensions)
        sample = sampler.random(n=self.pop_size)
        population = qmc.scale(sample, self.globalBounds[:, 0], self.globalBounds[:, 1])

        return population

    def evaluateInitialPopulation(self):

        for i in range(0, len(self.population)):
            newObjectiveTargets = MOobjective_function(
                self.population[i], self.objFunction, self.nObjectives
            )
            self.objectiveTargets = np.vstack(
                (self.objectiveTargets, newObjectiveTargets)
            )

    def scalariseInitialPopulation(self):

        for i in range(0, len(self.population)):
            # newObjectiveTargets = MOobjective_function(
            #     self.population[i], self.objFunction, self.nObjectives
            # )
            # self.objectiveTargets = np.vstack(
            #     (self.objectiveTargets, newObjectiveTargets)
            # )
            self.feFeatures = np.vstack((self.feFeatures, self.population[i]))

        # find minimum in boths columns - new zbest values

        self.zbests = np.min(self.objectiveTargets, axis=0)

        self.scalarisedTargets = scalariseValues(
            self.scalarisingFunction,
            self.objectiveTargets,
            self.zbests,
            self.weights,
            0,
            100,
        )


    def runOptimiser(self):
        iteration = 0

        # while self.bestEI > 1e-7:
        # while iteration < 80:
        while len(self.feFeatures) < (self.pop_size + self.maxFE):
            best_idx = np.argmin(self.scalarisedTargets)
            bestFeature = self.feFeatures[best_idx]
            bestTarget = self.scalarisedTargets[best_idx]

            globalGP = GPTrain(
                self.feFeatures, self.scalarisedTargets, meanPrior="zero"
            )



            eiDEGlobal = BayesianDifferentialEvolution(globalGP, self.globalBounds, bestTarget, max_generations=self.DEIter)
            newSolution, newFitness = eiDEGlobal.optimize()

 

            newObjectiveTargets = MOobjective_function(
                newSolution, self.objFunction, self.nObjectives
            )
            self.objectiveTargets = np.vstack(
                (self.objectiveTargets, newObjectiveTargets)
            )
            self.feFeatures = np.vstack((self.feFeatures, newSolution))

            # find minimum in boths columns - new zbest values

            self.zbests = np.min(self.objectiveTargets, axis=0)

            self.scalarisedTargets = scalariseValues(
                self.scalarisingFunction,
                self.objectiveTargets,
                self.zbests,
                self.weights,
                iteration,
                50,
            )




            print(f"BO Iteration {iteration}, Best found solution = ", bestTarget)
            print(f"Evaluated points = {len(self.feFeatures)}")



            self.bestEI = newFitness

            positions = np.arange(len(self.scalarisedTargets))


            plt.show()

            np.savetxt("BOFeatures.txt", self.feFeatures)
            np.savetxt("BOScalarisedTargets.txt", self.scalarisedTargets)
            np.savetxt("BOObjectiveTargets.txt", self.objectiveTargets)

            iteration += 1





function = 'dtlz2'

DEIter = [10, 25, 50, 75, 100, 200, 500]

n_var = 6
n_obj = 2
# 'granularity' of weight vector spacing
s = 8

weightVectors = func.generateWeightVectors(n_obj, s)
print(f'Generated {len(weightVectors)} weight vectors.')

# weightVectors = np.array((0.5, 0.5))


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

    newObjvTgt = MOobjective_function(initPopulation[i], problem, n_obj)
    initialObjvTargets = np.vstack((initialObjvTargets, newObjvTgt))

print("Initial Population:")
print(initPopulation)
print("initial targets:\n", initialObjvTargets )

for iterNum in DEIter:

    # try:

    bayesianRun = bayesianOptimiser(
        bounds,
        initSampleSize,
        problem,
        func.chebyshev,
        n_obj,
        weightVectors,
        DEIter=iterNum,
        useInitialPopulation=True,
        initialPopulation=initPopulation,
        initialObjvValues=initialObjvTargets,
        maxFE=100
    )
    bayesianRun.runOptimiser()
    # except TypeError:
        # print('Error during optimisation, skipping...')


    features = np.loadtxt("BOFeatures.txt")
    np.savetxt(
        f"BOFeatures{function}{iterNum}Iter.txt", features
    )

    scalarisedTargets = np.loadtxt("BOScalarisedTargets.txt")
    np.savetxt(
        f"BOScalarisedTargets{function}{iterNum}Iter.txt",
        scalarisedTargets,
    )

    objtTargets = np.loadtxt("BOObjectiveTargets.txt")
    np.savetxt(
        f"BOObjtvTargets{function}{iterNum}Iter.txt",
        objtTargets,
    )

