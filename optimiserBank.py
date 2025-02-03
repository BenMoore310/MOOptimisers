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
import functionBank as func
from sklearn.preprocessing import MinMaxScaler
import inspect
from pymoo.core.problem import Problem  # PyMoo Problem base class



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
        # isinstance(func, Problem):  # Checks if it's a PyMoo test function
   
        objectiveArray[0] = func.evalPyMooProblem(currentFunction, vec)

    
    # print(currentFunction)
    # print(vec)
    # if currentFunction == func.fonsecaFleming or currentFunction == func.sandTrap or currentFunction == func.viennetFunction:
    #     objectiveArray[0] = currentFunction(vec)
    #     # objectiveArray = np.empty((1,nObjectives))

    # else:
    #     x, y = vec

        # this gives the separate objective values to be returned for pareto plotting purposes
    # objectiveArray[0] = currentFunction(vec)
        # print(objectiveArray.shape)

    # noww scalarise the objectives
    # scalarisedObjective, newZBests = scalarisingFunction(objectiveArray, zbests, weights)

    # print(objectiveArray, scalarisedObjective)

    return objectiveArray


def scalariseValues(
    scalarisingFunction, objectiveArray, zBests, weights, currentGen, maxGen
):
    scaler = MinMaxScaler(feature_range=(0, 1))

    objsNormalised = scaler.fit_transform(objectiveArray)

    # print(objsNormalised)

    z0 = np.zeros_like(zBests)

    if scalarisingFunction == func.PAPBI:
        print("using PAPBI!")
        scalarisedArray = scalarisingFunction(
            objsNormalised, z0, weights, currentGen, maxGen
        )
    elif scalarisingFunction == func.HypI:
        # print('using HypI')
        scalarisedArray = scalarisingFunction(objsNormalised)

    else:
        scalarisedArray = np.empty(len(objectiveArray))
        for i in range(0, len(objectiveArray)):
            scalarisedArray[i] = scalarisingFunction(objsNormalised[i], z0, weights)

    # print(scalarisedArray)
    # unNormalisedScalarArray = scaler.inverse_transform(scalarisedArray)
    # print(unNormalisedScalarArray)

    return scalarisedArray


def removeNans(features, targets, objTargets):
    """
    Finds the location of NaN values in the targets array, identifies the worst (largest) value in the targets array,
    and replaces the corresponding values in the targets and objTargets arrays with the worst value.

    Parameters:
        features (np.ndarray): Input array with function inputs.
        targets (np.ndarray): Output array with function outputs.
        objTargets (np.ndarray): Array with additional output targets.

    Returns:
        tuple: A tuple containing modified inputs and outputs arrays.
    """
    import numpy as np

    # Log the indices of NaN values
    nan_indices = np.where(np.isnan(targets))[0]
    if nan_indices.size > 0:
        print(f"NaN values found at indices: {nan_indices}")
    else:
        print("No NaN values found.")
        return features, targets, objTargets

    # Find the index of the worst (largest) value in the targets array
    valid_indices = ~np.isnan(targets)
    worst_index = np.argmax(targets[valid_indices])
    worst_value_targets = targets[valid_indices][worst_index]
    worst_value_objTargets = objTargets[valid_indices][worst_index]

    # Replace NaN values with the worst values
    for idx in nan_indices:
        targets[idx] = worst_value_targets
        objTargets[idx] = worst_value_objTargets

    return features, targets, objTargets


# def removeNans(features, targets, objTargets):
#     """
#     Removes NaN values from the outputs array and corresponding entries in the inputs array.

#     Parameters:
#         features (np.ndarray): Input array with function inputs.
#         targets (np.ndarray): Output array with function outputs.

#     Returns:
#         tuple: A tuple containing cleaned inputs and outputs arrays.
#     """
#     # Create a boolean mask for non-NaN values in the outputs array
#     mask = ~np.isnan(targets)

#     # Log the indices of NaN values
#     nan_indices = np.where(~mask)[0]
#     if nan_indices.size > 0:
#         print(f"NaN values found at indices: {nan_indices}")
#     else:
#         print("No NaN values found.")

#     # Return the cleaned arrays
#     return features[mask], targets[mask], objTargets[mask]


# plt.style.available
# plt.style.use(['science', 'notebook'])
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

    training_iter = 250
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


def GPEval(model, newFeatures):
    model.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = model(torch.from_numpy(newFeatures))

    mean_pred = observed_pred.mean.numpy()

    return mean_pred


class DifferentialEvolution:
    def __init__(
        self,
        bounds,
        objective_function,
        pop_size=50,
        mutation_factor=0.8,
        crossover_prob=0.7,
        max_generations=200,
        method="random",
    ):
        """
        Initialize the Differential Evolution (DE) optimizer.

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

        # Initialize population
        self.population = self.initialize_population()
        self.best_solution = None
        self.best_fitness = np.inf
        self.objective_function = objective_function

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

        # print(population.shape)
        return population

    def mutate(self, target_idx):
        """Mutation using DE/best/1 strategy."""
        # Choose three random and distinct individuals different from target_idx
        indices = [idx for idx in range(self.pop_size) if idx != target_idx]
        np.random.shuffle(indices)
        r1, r2, r3 = indices[:3]

        # Best individual in current population

        # print(self.population.shape)

        # TODO  instead of this list comprehension bollocks just evaluate them all at once
        # as thats what i think it wants, then find the minimum of the results.

        predictedValues = GPEval(self.objective_function, self.population)

        best_idx = np.argsort(predictedValues)[:1]

        best = self.population[best_idx]

        # best_idx = np.argmin([self.objective_function.predict(ind) for ind in self.population])
        # best = self.population[best_idx]

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
        # print(trial.shape)
        # print(mutant.shape)
        for i in range(self.dimensions):
            if np.random.rand() < self.crossover_prob or i == np.random.randint(
                self.dimensions
            ):
                # print(trial[i], mutant[i])
                trial[i] = mutant[i]
        return trial

    def select(self, target, trial):
        """Selection: Return the individual with the better fitness."""
        if self.objective_function.predict(trial) < self.objective_function.predict(
            target
        ):
            return trial
        return target

    def select(self, target, trial):
        """Selection: Return the individual with the better fitness."""
        if GPEval(self.objective_function, trial) < GPEval(
            self.objective_function, target
        ):
            return trial
        return target

    def optimize(self):
        """Run the Differential Evolution optimization."""
        # x_range = np.linspace(-5, 5, 100)
        # y_range = np.linspace(-5, 5, 100)
        # X, Y = np.meshgrid(x_range, y_range)
        # Z = ackley_function(X, Y)
        # x_range = np.linspace(self.bounds[0, 0], self.bounds[0, 1], 50)
        # y_range = np.linspace(self.bounds[1, 0], self.bounds[1, 1], 50)
        # fullRange = list(product(x_range, y_range))
        # fullRangeArray = np.array(fullRange)
        # # y_pred = self.objective_function.predict(fullRangeArray)
        # y_pred = GPEval(self.objective_function, fullRangeArray)

        for generation in range(self.max_generations):
            new_population = np.zeros_like(self.population)
            allTrials = np.zeros_like(self.population)
            allTargets = np.zeros_like(self.population)
            # print(self.population.shape)
            for i in range(self.pop_size):
                target = self.population[i]
                # print('break')
                # print(i)
                mutant = self.mutate(i)
                # print(mutant)
                mutant = np.reshape(mutant, (self.dimensions,))
                # print(mutant)
                trial = self.crossover(target, mutant)
                trial = np.reshape(trial, (1, -1))
                target = np.reshape(target, (1, -1))
                # print('for select', trial.shape, target.shape)
                new_population[i] = self.select(target, trial)

            # Update the population
            self.population = new_population

            # Track the best solution
            # best_idx = np.argmin([self.objective_function.predict(ind) for ind in self.population])
            # best_fitness = self.objective_function.predict(self.population[best_idx])

            # predictedValues = self.objective_function.predict(self.population)
            predictedValues = GPEval(self.objective_function, self.population)

            best_idx = np.argsort(predictedValues)[:1]

            # best_fitness = self.objective_function.predict(self.population[best_idx])
            best_fitness = GPEval(self.objective_function, self.population[best_idx])

            if best_fitness < self.best_fitness:
                self.best_fitness = best_fitness
                self.best_solution = self.population[best_idx]

            # plt.contourf(x_range, y_range, y_pred, levels=50, cmap='viridis')
        # plt.scatter(fullRangeArray[:,0], fullRangeArray[:,1], c = y_pred)

        # plt.scatter(self.population[:, 0], self.population[:, 1], color='red', label='Final Population', s=5)
        # plt.scatter(self.best_solution[0,0], self.best_solution[0,1], color='blue', label='Best Solution', s=100)
        # plt.legend()
        # plt.title("Local Surrogate")
        # plt.colorbar()
        # plt.clim(np.min(y_pred), np.max(y_pred))
        # plt.xlim(self.bounds[0,0], self.bounds[0,1])
        # plt.ylim(self.bounds[1,0], self.bounds[1,1])
        # plt.savefig('localGP.png')
        # plt.close()

        # plt.show()
        # # Debug information
        # print(f"Generation {generation + 1}: Best RBF Fitness = {self.best_fitness}")

        return self.best_solution, self.best_fitness


class BayesianDifferentialEvolution:
    def __init__(
        self,
        surrogateModel,
        bounds,
        bestTarget,
        pop_size=75,
        mutation_factor=0.8,
        crossover_prob=0.7,
        max_generations=40,
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
        # if expectedImprovement(self.surrogateModel, trial, self.bestTarget,  0.01) < expectedImprovement(self.surrogateModel, target, self.bestTarget,  0.01):
        #     return trial
        # return target

    def optimize(self):
        """Run the Differential Evolution optimization."""
        # x_range = np.linspace(self.bounds[0, 0], self.bounds[0, 1], 50)
        # y_range = np.linspace(self.bounds[1, 0], self.bounds[1, 1], 50)
        # fullRange = list(product(x_range, y_range))
        # fullRangeArray = np.array(fullRange)
        # Z = expectedImprovement(
        #     self.surrogateModel, fullRangeArray, self.bestTarget, 0.01
        # )

        for generation in range(self.max_generations):
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

            # print(best_fitness)

            # best_idx = np.argmin([expectedImprovement(self.surrogateModel, ind, self.bestTarget, 0.01) for ind in self.population])
            # best_fitness = expectedImprovement(self.surrogateModel, self.population[best_idx], self.bestTarget, 0.01)

            if best_fitness > self.best_fitness:
                self.best_fitness = best_fitness
                self.best_solution = self.population[best_idx]
                # print('"best solution"', self.best_solution)

            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            # plt.contourf(x_range, y_range, Z, levels=50, cmap='viridis')
        # plt.scatter(fullRangeArray[:,0], fullRangeArray[:,1], c = Z, alpha = 0.5)
        # plt.scatter(self.population[:, 0], self.population[:, 1], color='red', label='Population', s=10, marker='x')
        # plt.scatter(self.best_solution[0], self.best_solution[1], color='blue', label='Best Solution', s=10)
        # plt.legend()
        # plt.title("DE Optimisation of Expected Improvement")
        # plt.colorbar()
        # # plt.yscale('log')
        # plt.clim(np.min(Z), np.max(Z))
        # # plt.savefig('eiDE.png')
        # plt.show()
        # plt.close()
        # Debug information
        # print(f"Generation {generation + 1}: Best Fitness = {self.best_fitness}")

        return self.best_solution, self.best_fitness


class TS_DDEO:
    def __init__(
        self,
        bounds,
        pop_size,
        objFunction,
        scalarisingFunction,
        nObjectives,
        weights,
        useInitialPopulation,
        initialPopulation,
        initialObjvValues,
        c1=2.05,
        c2=2.05,
        PSOFE=50,
        BDDOFE=50,
        mutation_factor=0.8,
        crossover_prob=0.7,
    ):
        self.globalBounds = np.array(bounds)
        self.dimensions = len(bounds)
        self.pop_size = pop_size
        self.max_generations = PSOFE
        self.feFeatures = np.empty((0, self.dimensions))  # Consistent with dimensions
        self.globalBestFeature = None
        self.nObjectives = nObjectives
        self.objectiveTargets = np.empty((0, self.nObjectives))
        self.scalarisedTargets = np.empty(0)

        # self.feTargets = np.empty(0)
        self.globalBestTarget = np.inf  # Initialize as infinity
        self.popBestFeature = np.empty((self.pop_size, self.dimensions))
        self.popBestTargets = np.full(self.pop_size, np.inf)  # Initialize with inf
        self.generator = np.random.default_rng()
        self.BBDOIter = BDDOFE
        self.objFunction = objFunction

        self.scalarisingFunction = scalarisingFunction
        self.zbests = np.empty((0))
        self.weights = weights
        # Initialize population and velocities
        if useInitialPopulation == True:
            self.population = initialPopulation
            self.objectiveTargets = initialObjvValues
        else:
            self.population = self.initialisePopulation()
            self.evaluateInitialPopulation()
        self.scalariseInitialPopulation()

        self.velocities = self.initialiseVelocities()

        # PSO coefficients
        self.c1 = c1
        self.c2 = c2
        self.phi = c1 + c2
        self.chi = 2 / abs((2 - self.phi - np.sqrt(self.phi**2 - 4 * self.phi)))

        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob

    def initialisePopulation(self):
        """Initializes the population using Latin Hypercube Sampling."""
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

        self.feFeatures, self.scalarisedTargets, self.objectiveTargets = removeNans(
            self.feFeatures, self.scalarisedTargets, self.objectiveTargets
        )

        # this is a wastefull hack to prevent nans in the initial population
        # halting TSDDEO - as here the initial population is tracked throughout
        # TSDDEO stage 1, and has to have the same shape as the initial number of
        # features (after nans have been removed)

        # have also updated self.pop_size to the post-nan removal value

        self.population = self.feFeatures.copy()

        self.pop_size = len(self.feFeatures)

        # Initialize the personal bests
        self.popBestFeature = self.feFeatures.copy()
        # print(self.popBestFeature.shape)
        self.popBestTargets = self.scalarisedTargets.copy()

        # Initialize the global best
        bestIdx = np.argmin(self.scalarisedTargets)
        self.globalBestFeature = self.feFeatures[bestIdx]
        self.globalBestTarget = self.scalarisedTargets[bestIdx]

        # Plot initial population
        # plt.scatter(self.feFeatures[:, 0], self.feFeatures[:, 1], c=self.feTargets, cmap='viridis')
        # plt.title('Initial Population')
        # plt.colorbar()
        # plt.show()

    def initialiseVelocities(self):
        """Initializes particle velocities as a small fraction of the bounds range."""
        velocity_range = (self.globalBounds[:, 1] - self.globalBounds[:, 0]) * 0.001
        return self.generator.uniform(
            low=-velocity_range,
            high=velocity_range,
            size=(self.pop_size, self.dimensions),
        )

    def updateVelocity(self):
        """Updates the velocities of the particles."""
        r1 = self.generator.random(size=(self.pop_size, self.dimensions))
        r2 = self.generator.random(size=(self.pop_size, self.dimensions))
        # print(r1.shape, self.popBestFeature.shape, self.population.shape)
        cognitive = self.c1 * r1 * (self.popBestFeature - self.population)
        social = self.c2 * r2 * (self.globalBestFeature - self.population)
        self.velocities = self.chi * (self.velocities + cognitive + social)

        # print(self.velocities)

    def updatePosition(self):
        """Updates the positions of the particles."""
        self.population += self.velocities
        # print(self.population)

    def clipPositions(self):
        """Clips particle positions to stay within bounds."""
        self.population = np.clip(
            self.population, self.globalBounds[:, 0], self.globalBounds[:, 1]
        )

    def stage1(self):
        """Runs the PSO optimization loop."""
        # x_range = np.linspace(self.globalBounds[0, 0], self.globalBounds[0, 1], 100)
        # y_range = np.linspace(self.globalBounds[1, 0], self.globalBounds[1, 1], 100)
        # X, Y = np.meshgrid(x_range, y_range)
        # Z = ackley_function(X, Y)

        iteration = 1

        while len(self.feFeatures) < (self.pop_size + self.max_generations):
            # Train surrogate model and find a solution
            GPModel = GPTrain(self.feFeatures, self.scalarisedTargets, meanPrior="zero")
            globalDE = DifferentialEvolution(self.globalBounds, GPModel)
            bestGPSolution, bestGPFitness = globalDE.optimize()

            # Evaluate the found solution
            bestGPSolution = np.reshape(bestGPSolution, (self.dimensions,))
            # print(bestGPSolution.shape)
            newObjectiveTargets = MOobjective_function(
                bestGPSolution, self.objFunction, self.nObjectives
            )
            self.objectiveTargets = np.vstack(
                (self.objectiveTargets, newObjectiveTargets)
            )
            self.feFeatures = np.vstack((self.feFeatures, bestGPSolution))

            # find minimum in boths columns - new zbest values

            self.zbests = np.min(self.objectiveTargets, axis=0)

            self.scalarisedTargets = scalariseValues(
                self.scalarisingFunction,
                self.objectiveTargets,
                self.zbests,
                self.weights,
                iteration,
                self.max_generations,
            )

            self.feFeatures, self.scalarisedTargets, self.objectiveTargets = removeNans(
                self.feFeatures, self.scalarisedTargets, self.objectiveTargets
            )

            # Update global best if necessary
            bestIdx = np.argmin(self.scalarisedTargets)
            self.globalBestFeature = self.feFeatures[bestIdx]
            self.globalBestTarget = self.scalarisedTargets[bestIdx]

            # Update velocities and positions
            self.updateVelocity()
            self.updatePosition()
            self.clipPositions()

            # Evaluate swarm on surrogate model
            swarmOnGP = GPEval(GPModel, self.population)
            betterThanPBest = swarmOnGP < self.popBestTargets

            # Evaluate selected particles
            toEvaluate = np.where(betterThanPBest)[0]
            for idx in toEvaluate:
                particle = self.population[idx]
                newObjectiveTargets = MOobjective_function(
                    particle, self.objFunction, self.nObjectives
                )
                self.objectiveTargets = np.vstack(
                    (self.objectiveTargets, newObjectiveTargets)
                )
                self.feFeatures = np.vstack((self.feFeatures, particle))

                # find minimum in boths columns - new zbest values

                self.zbests = np.min(self.objectiveTargets, axis=0)

                self.scalarisedTargets = scalariseValues(
                    self.scalarisingFunction,
                    self.objectiveTargets,
                    self.zbests,
                    self.weights,
                    iteration,
                    self.max_generations,
                )

                # Update personal best if necessary
                if self.scalarisedTargets[-1] < self.popBestTargets[idx]:
                    self.popBestTargets[idx] = self.scalarisedTargets[-1]
                    self.popBestFeature[idx] = particle
            self.feFeatures, self.scalarisedTargets, self.objectiveTargets = removeNans(
                self.feFeatures, self.scalarisedTargets, self.objectiveTargets
            )

            # refind best values
            bestIdx = np.argmin(self.scalarisedTargets)
            self.globalBestFeature = self.feFeatures[bestIdx]
            self.globalBestTarget = self.scalarisedTargets[bestIdx]
            iteration += 1
            # Plot the optimization progress
            # plt.contourf(X, Y, Z, levels=50, cmap='viridis')
            # plt.scatter(self.population[:, 0], self.population[:, 1], color='red', label='Swarm', s=10)
            # plt.legend()
            # plt.title(f"PSO Optimization - Iteration {iteration + 1}")
            # plt.colorbar()
            # plt.clim(0, 14)
            # plt.show()

            # plt.scatter(self.feFeatures[:, 0], self.feFeatures[:, 1], c=self.feTargets, cmap='viridis')
            # plt.title('Initial Population')
            # plt.colorbar()
            # plt.show()
            np.savetxt("TSDDEOFeatures.txt", self.feFeatures)
            np.savetxt("TSDDEOTargets.txt", self.scalarisedTargets)
            np.savetxt("TSDDEOObjectiveTargets.txt", self.objectiveTargets)

            # Debug information
            print(
                f"PSO Iteration {iteration}: Best Fitness = {self.scalarisedTargets[bestIdx]}"
            )
            print(f"Evaluated points = {len(self.feFeatures)}")

    def mutate(self, target_idx, currentGP):
        """Mutation using DE/best/1 strategy."""
        # Choose three random and distinct individuals different from target_idx
        indices = [idx for idx in range(self.pop_size) if idx != target_idx]
        np.random.shuffle(indices)
        r1, r2, r3 = indices[:3]

        predictedValues = GPEval(currentGP, self.population)

        best_idx = np.argsort(predictedValues)[:1]

        best = self.population[best_idx]

        # Mutant vector: v = best + F * (r1 - r2)
        mutant = best + self.mutation_factor * (
            self.population[r1] - self.population[r2]
        )

        # Ensure mutant vector is within bounds
        mutant = np.clip(mutant, self.globalBounds[:, 0], self.globalBounds[:, 1])

        return mutant

    def crossover(self, target, mutant):
        """Crossover to create trial vector."""
        trial = np.copy(target)
        # print(trial.shape)
        # print(mutant.shape)
        for i in range(self.dimensions):
            if np.random.rand() < self.crossover_prob or i == np.random.randint(
                self.dimensions
            ):
                # print(trial[i], mutant[i])
                trial[i] = mutant[i]
        return trial

    def localRBF(self, numSolutions):
        bestFeatures = np.empty((numSolutions, self.dimensions))
        bestTargets = np.empty(numSolutions)

        # find c best solutions
        bestIndices = np.argsort(self.scalarisedTargets)[:numSolutions]

        for i in range(numSolutions):
            bestFeatures[i] = self.feFeatures[bestIndices[i]]
            bestTargets[i] = self.scalarisedTargets[bestIndices[i]]

        # x_min, x_max = np.min(bestFeatures[:, 0]), np.max(bestFeatures[:, 0])
        # y_min, y_max = np.min(bestFeatures[:, 1]), np.max(bestFeatures[:, 1])

        # bounds = [(x_min, x_max), (y_min, y_max)]

        bounds = [(np.min(bestFeatures[:, d]), np.max(bestFeatures[:, d])) for d in range(bestFeatures.shape[1])]

        # pairwiseDistancesLocal = np.linalg.norm(bestFeatures[:, np.newaxis] - bestFeatures, axis=2)
        # avgDistanceLocal = np.mean(pairwiseDistancesLocal)

        localGP = GPTrain(bestFeatures, bestTargets, meanPrior="max")

        # localRBF = RBFSurrogateModel(epsilon=1.0)
        # localRBF.fit(bestFeatures, bestTargets)

        # functionEval = localRBF.predict()
        localDE = DifferentialEvolution(bounds, localGP)
        bestLocalSolution, bestLocalFitness = localDE.optimize()

        return bestLocalSolution

    def fullCrossover(self, iteration):
        # build surrogate using all points in population
        crossoverGP = GPTrain(self.feFeatures, self.scalarisedTargets, meanPrior="max")

        best_idx = np.argmin(self.scalarisedTargets)
        bestFeature = self.feFeatures[best_idx]
        bestTarget = self.scalarisedTargets[best_idx]

        RVS = random.sample(range(0, self.dimensions), self.dimensions)

        # print(self.feFeatures.shape)
        # print(bestFeature)

        # x_range = np.linspace(self.globalBounds[0, 0], self.globalBounds[0, 1], 100)
        # y_range = np.linspace(self.globalBounds[1, 0], self.globalBounds[1, 1], 100)
        # fullRange = list(product(x_range, y_range))
        # fullRangeArray = np.array(fullRange)
        # # y_pred = self.objective_function.predict(fullRangeArray)
        # y_pred = GPEval(crossoverGP, fullRangeArray)

        tempPop = np.full_like((self.feFeatures), bestFeature)

        # print(tempPop)

        for i in RVS:
            # print(i)

            tempPop[:, i] = self.feFeatures[:, i]

            tempPopEval = GPEval(crossoverGP, tempPop)

            # plt.scatter(fullRangeArray[:,0], fullRangeArray[:,1], c = y_pred, alpha = 0.5)
            # plt.scatter(tempPop[:, 0], tempPop[:, 1], color='red', label='crossover', s=10, marker='x')
            # plt.scatter(bestFeature[0], bestFeature[1], color='blue', label='Best Solution', s=10)
            # plt.legend()
            # plt.title("Full-Crossover")
            # plt.colorbar()
            # # plt.clim(0, 14)
            # # plt.savefig(f'DEPlots/{generation}.png')
            # plt.show()

            tempPopBestIdx = np.argmin(tempPopEval)
            if tempPopEval[tempPopBestIdx] > bestTarget:
                tempPop = np.full_like((self.feFeatures), tempPop[tempPopBestIdx])
                # print("new best value!")

        # take final best predicted value and explicitely evaluate
        newObjectiveTargets = MOobjective_function(
            tempPop[tempPopBestIdx], self.objFunction, self.nObjectives
        )
        self.objectiveTargets = np.vstack((self.objectiveTargets, newObjectiveTargets))
        self.feFeatures = np.vstack((self.feFeatures, tempPop[tempPopBestIdx]))

        # find minimum in boths columns - new zbest values

        self.zbests = np.min(self.objectiveTargets, axis=0)

        self.scalarisedTargets = scalariseValues(
            self.scalarisingFunction,
            self.objectiveTargets,
            self.zbests,
            self.weights,
            iteration,
            self.BBDOIter,
        )
        self.feFeatures, self.scalarisedTargets, self.objectiveTargets = removeNans(
            self.feFeatures, self.scalarisedTargets, self.objectiveTargets
        )

    def stage2(self):
        iteration = 0

        while len(self.feFeatures) < (self.BBDOIter + self.max_generations + self.pop_size):
            # DE screening stage
            GPModel = GPTrain(self.feFeatures, self.scalarisedTargets, meanPrior="max")

            new_population = np.zeros_like(self.population)

            for i in range(self.pop_size):
                # print(i)
                target = self.population[i]
                mutant = self.mutate(i, GPModel)
                mutant = np.reshape(mutant, (self.dimensions,))

                trial = self.crossover(target, mutant)
                trial = np.reshape(trial, (1, -1))
                target = np.reshape(target, (1, -1))
                new_population[i] = trial

                # Update the population
            self.population = new_population

            popOnGP = GPEval(GPModel, self.population)

            # evaluating whole landscape on RBF for plotting reasons:
            # x_range = np.linspace(-5, 5, 50)
            # y_range = np.linspace(-5, 5, 50)
            # fullRange = list(product(x_range, y_range))
            # fullRangeArray = np.array(fullRange)
            # y_pred = GPEval(GPModel, fullRangeArray)

            # function evaluation of best predicted child
            best_idx = np.argmin(popOnGP)

            bestFeature = self.population[best_idx]
            # self.population[best_idx]

            # print('best index =', best_idx)
            # evaluate best child and add results to global stores of FE features and targets
            newObjectiveTargets = MOobjective_function(
                self.population[best_idx], self.objFunction, self.nObjectives
            )
            self.objectiveTargets = np.vstack(
                (self.objectiveTargets, newObjectiveTargets)
            )
            self.feFeatures = np.vstack((self.feFeatures, self.population[best_idx]))

            # find minimum in boths columns - new zbest values

            self.zbests = np.min(self.objectiveTargets, axis=0)

            self.scalarisedTargets = scalariseValues(
                self.scalarisingFunction,
                self.objectiveTargets,
                self.zbests,
                self.weights,
                iteration,
                self.BBDOIter,
            )
            self.feFeatures, self.scalarisedTargets, self.objectiveTargets = removeNans(
                self.feFeatures, self.scalarisedTargets, self.objectiveTargets
            )

            # plt.scatter(fullRangeArray[:,0], fullRangeArray[:,1], c = y_pred)

            # plt.scatter(self.population[:, 0], self.population[:, 1], color='red', label='Final Population', s=5)
            # plt.scatter(bestFeature[0], bestFeature[1], color='blue', label='Best Solution', s=10)
            # # plt.legend()
            # plt.title("Global Surrogate")
            # plt.colorbar()
            # plt.clim(0,14)
            # plt.show()

            # construct local RBF using c best solutions, find minima using DE, and evaluate at that minima

            bestLocalSolution = self.localRBF(15)

            bestLocalSolution = np.reshape(bestLocalSolution, (self.dimensions,))

            # print("best local Solution", bestLocalSolution)

            newObjectiveTargets = MOobjective_function(
                bestLocalSolution, self.objFunction, self.nObjectives
            )
            self.objectiveTargets = np.vstack(
                (self.objectiveTargets, newObjectiveTargets)
            )
            self.feFeatures = np.vstack((self.feFeatures, bestLocalSolution))

            # find minimum in boths columns - new zbest values

            self.zbests = np.min(self.objectiveTargets, axis=0)

            self.scalarisedTargets = scalariseValues(
                self.scalarisingFunction,
                self.objectiveTargets,
                self.zbests,
                self.weights,
                iteration,
                self.BBDOIter,
            )
            self.feFeatures, self.scalarisedTargets, self.objectiveTargets = removeNans(
                self.feFeatures, self.scalarisedTargets, self.objectiveTargets
            )

            self.fullCrossover(iteration)

            iteration += 1
            bestIdx = np.argmin(self.scalarisedTargets)

            np.savetxt("TSDDEOFeatures.txt", self.feFeatures)
            np.savetxt("TSDDEOTargets.txt", self.scalarisedTargets)
            np.savetxt("TSDDEOObjectiveTargets.txt", self.objectiveTargets)

            print(
                f"BDDO Iteration {iteration}: Best Fitness = {self.scalarisedTargets[bestIdx]}"
            )
            print(f"Evaluated points = {len(self.feFeatures)}")


# def lipschitz_global_underestimate(f_values, samplesXY, L, test_points):
#     """
#     Compute the Lipschitz-based global underestimate at specific points.

#     Parameters:
#     f_values: np.ndarray of shape (n_samples,)
#         The function values at the sampled points.
#     samplesXY: np.ndarray of shape (n_samples, 2)
#         Array of the sampled (x, y) points.
#     L: float
#         The Lipschitz constant.
#     test_points: np.ndarray of shape (n, 2)
#         Array of (x, y) points at which to compute the underestimate.

#     Returns:
#     Z_under: np.ndarray of shape (n,)
#         The global Lipschitz-based underestimate values at the test points.
#     """
#     n_test_points = test_points.shape[0]
#     Z_under = np.full(n_test_points, -np.inf)  # Initialize with very low values

#     # Loop over all sample points to compute their individual underestimates
#     for i, (x_i, y_i) in enumerate(samplesXY):
#         f_x_i_y_i = f_values[i]

#         # Compute the distance from each test point to the sample point (x_i, y_i)
#         distances = np.sqrt(
#             (test_points[:, 0] - x_i) ** 2 + (test_points[:, 1] - y_i) ** 2
#         )

#         # Compute the local Lipschitz underestimate for this sample
#         Z_local_under = f_x_i_y_i - L * distances

#         # Update the global underestimate by taking the maximum across samples
#         Z_under = np.maximum(Z_under, Z_local_under)

#     return Z_under

def lipschitz_global_underestimate(f_values, samples, L, test_points):
    """
    Compute the Lipschitz-based global underestimate at specific points in any-dimensional space.

    Parameters:
    f_values: np.ndarray of shape (n_samples,)
        The function values at the sampled points.
    samples: np.ndarray of shape (n_samples, d)
        Array of the sampled points in d-dimensional space.
    L: float
        The Lipschitz constant.
    test_points: np.ndarray of shape (n, d)
        Array of points in d-dimensional space at which to compute the underestimate.

    Returns:
    Z_under: np.ndarray of shape (n,)
        The global Lipschitz-based underestimate values at the test points.
    """
    n_test_points = test_points.shape[0]
    Z_under = np.full(n_test_points, -np.inf)  # Initialise with very low values

    # Loop over all sample points to compute their individual underestimates
    for i, sample_point in enumerate(samples):
        f_sample = f_values[i]

        # Compute the distance from each test point to the current sample point
        distances = np.linalg.norm(test_points - sample_point, axis=1)

        # Compute the local Lipschitz underestimate for this sample
        Z_local_under = f_sample - L * distances

        # Update the global underestimate by taking the maximum across samples
        Z_under = np.maximum(Z_under, Z_local_under)

    return Z_under


def estimate_lipschitz_constant(samplesXY, f_values):
    """
    Estimate the Lipschitz constant based on known sample points and their function values.

    Parameters:
    samplesXY: np.ndarray of shape (n_samples, 2)
        Array of the sampled (x, y) points.
    f_values: np.ndarray of shape (n_samples,)
        Array of function values at the sampled points.

    Returns:
    L_est: float
        The estimated Lipschitz constant.
    """
    n_samples = samplesXY.shape[0]
    # print(samplesXY.shape)
    # print(n_samples)
    L_est = 0.0

    # Loop over all pairs of points to estimate L
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            # Compute the Euclidean distance between points i and j
            dist = np.linalg.norm(samplesXY[i] - samplesXY[j])
            if dist > 0:
                # Compute the absolute difference in function values
                f_diff = np.abs(f_values[i] - f_values[j])
                # Compute the slope and update the maximum
                # print(L_est, f_diff/dist)
                L_est = max(L_est, f_diff / dist)

    return L_est


class LSADE:
    def __init__(
        self,
        bounds,
        pop_size,
        objFunction,
        scalarisingFunction,
        nObjectives,
        weights,
        useInitialPopulation,
        initialPopulation,
        initialObjvValues,
        DEPop=50,
        mutation_factor=0.8,
        crossover_prob=0.7,
        method="lhs",
        max_generations=100,
    ):
        """
        Initialize the Differential Evolution (DE) optimizer.

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
        self.nObjectives = nObjectives
        self.objectiveTargets = np.empty((0, self.nObjectives))
        self.scalarisedTargets = np.empty(0)
        self.feFeatures = np.empty((0, self.dimensions))
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.DEPop = np.empty((DEPop, self.dimensions))
        self.max_generations = max_generations
        self.method = method
        self.feFeatures = np.empty((0, self.dimensions))
        # self.feTargets = np.empty(0)
        # Initialize population
        self.objFunction = objFunction

        self.scalarisingFunction = scalarisingFunction
        self.zbests = np.full((1, self.nObjectives), np.inf)
        self.weights = weights
        if useInitialPopulation == True:
            self.population = initialPopulation
            self.objectiveTargets = initialObjvValues

        else:
            self.population = self.initialisePopulation()
            self.evaluateInitialPopulation()
        self.scalariseInitialPopulation()


        self.best_solution = None
        self.best_fitness = np.inf

    def initialisePopulation(self):
        """Initialize population using random sampling or Latin Hypercube Sampling."""

        # print('initial shape', self.feFeatures.shape)
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
        self.feFeatures, self.scalarisedTargets, self.objectiveTargets = removeNans(
            self.feFeatures, self.scalarisedTargets, self.objectiveTargets
        )

        # for i in range(len(self.objectiveTargets)):
        #     print(self.objectiveTargets[i], self.scalarisedTargets[i])

        # plt.scatter(self.feFeatures[:,0], self.feFeatures[:,1], c = self.feTargets)
        # plt.title('Initial Population')
        # plt.colorbar()
        # plt.show()

    def mutate(self, target_idx, currentGP):
        """Mutation using DE/best/1 strategy."""
        # Choose three random and distinct individuals different from target_idx
        indices = [idx for idx in range(self.pop_size) if idx != target_idx]
        np.random.shuffle(indices)
        r1, r2, r3 = indices[:3]

        predictedValues = GPEval(currentGP, self.population)

        best_idx = np.argsort(predictedValues)[:1]

        best = self.population[best_idx]

        # Mutant vector: v = best + F * (r1 - r2)
        mutant = best + self.mutation_factor * (
            self.population[r1] - self.population[r2]
        )

        # Ensure mutant vector is within bounds
        mutant = np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])

        return mutant

    def crossover(self, target, mutant):
        """Crossover to create trial vector.
        This loops through the features in the vector (dimensions) and sees if any of them crossover.
        allows the retention of some original vector features but not others.
        """

        trial = np.copy(target)
        # print(trial.shape)
        # print(mutant.shape)
        for i in range(self.dimensions):
            if np.random.rand() < self.crossover_prob or i == np.random.randint(
                self.dimensions
            ):
                # print(trial[i], mutant[i])

                trial[i] = mutant[i]
        return trial

    # def select(self, target, trial):
    #     """Selection: Return the individual with the better fitness."""
    #     if objective_function(trial) < objective_function(target):
    #         return trial
    #     return target

    def localRBF(self, numSolutions):
        bestFeatures = np.empty((numSolutions, self.dimensions))
        bestTargets = np.empty(numSolutions)

        # find c best solutions
        bestIndices = np.argsort(self.scalarisedTargets)[:numSolutions]

        for i in range(numSolutions):
            bestFeatures[i] = self.feFeatures[bestIndices[i]]
            bestTargets[i] = self.scalarisedTargets[bestIndices[i]]

        # x_min, x_max = np.min(bestFeatures[:, 0]), np.max(bestFeatures[:, 0])
        # y_min, y_max = np.min(bestFeatures[:, 1]), np.max(bestFeatures[:, 1])

        # bounds = [(x_min, x_max), (y_min, y_max)]

        bounds = [(np.min(bestFeatures[:, d]), np.max(bestFeatures[:, d])) for d in range(bestFeatures.shape[1])]

        # pairwiseDistancesLocal = np.linalg.norm(bestFeatures[:, np.newaxis] - bestFeatures, axis=2)
        # avgDistanceLocal = np.mean(pairwiseDistancesLocal)

        localGP = GPTrain(bestFeatures, bestTargets, meanPrior="max")

        # localRBF = RBFSurrogateModel(epsilon=1.0)
        # localRBF.fit(bestFeatures, bestTargets)

        # functionEval = localRBF.predict()
        localDE = DifferentialEvolution(bounds, localGP)
        bestLocalSolution, bestLocalFitness = localDE.optimize()

        return bestLocalSolution

    def optimizerStep(self):
        """Run the Differential Evolution optimization."""
        # x_range = np.linspace(-5, 5, 100)
        # y_range = np.linspace(-5, 5, 100)
        # X, Y = np.meshgrid(x_range, y_range)
        # Z = ackley_function(X, Y)
        iteration = 0
        while iteration < (self.max_generations/3):
            GPModel = GPTrain(self.feFeatures, self.scalarisedTargets, meanPrior="max")

            new_population = np.zeros_like(self.population)

            for i in range(self.pop_size):
                # print(i)
                target = self.population[i]
                mutant = self.mutate(i, GPModel)
                mutant = np.reshape(mutant, (self.dimensions,))

                trial = self.crossover(target, mutant)
                trial = np.reshape(trial, (1, -1))
                target = np.reshape(target, (1, -1))
                new_population[i] = trial

                # Update the population
            self.population = new_population
            popOnGP = GPEval(GPModel, self.population)

            # evaluating whole landscape on RBF for plotting reasons:
            # x_range = np.linspace(self.bounds[0, 0], self.bounds[0, 1], 100)
            # y_range = np.linspace(self.bounds[1, 0], self.bounds[1, 1], 100)
            # fullRange = list(product(x_range, y_range))
            # fullRangeArray = np.array(fullRange)
            # y_pred = GPEval(GPModel, fullRangeArray)

            # evaluate current population (children) on RBF
            # popOnRBF = globalRBF.predict(self.population)

            # function evaluation of best predicted child
            best_idx = np.argmin(popOnGP)

            bestFeature = self.population[best_idx]

            # print('best index =', best_idx)
            # evaluate best child and add results to global stores of FE features and targets
            newObjectiveTargets = MOobjective_function(
                self.population[best_idx], self.objFunction, self.nObjectives
            )
            self.objectiveTargets = np.vstack(
                (self.objectiveTargets, newObjectiveTargets)
            )
            self.feFeatures = np.vstack((self.feFeatures, self.population[best_idx]))

            # find minimum in boths columns - new zbest values

            self.zbests = np.min(self.objectiveTargets, axis=0)

            self.scalarisedTargets = scalariseValues(
                self.scalarisingFunction,
                self.objectiveTargets,
                self.zbests,
                self.weights,
                iteration,
                self.max_generations,
            )
            self.feFeatures, self.scalarisedTargets, self.objectiveTargets = removeNans(
                self.feFeatures, self.scalarisedTargets, self.objectiveTargets
            )

            # plt.scatter(fullRangeArray[:,0], fullRangeArray[:,1], c = y_pred)

            # plt.scatter(self.population[:, 0], self.population[:, 1], color='red', label='Final Population', s=5)
            # plt.scatter(bestFeature[0], bestFeature[1], color='blue', label='Best Solution', s=10)
            # # plt.legend()
            # plt.title("Global Surrogate")
            # plt.colorbar()
            # plt.clim(np.min(y_pred), np.max(y_pred))
            # plt.savefig('globalGP.png')
            # plt.close()
            # generate Lipschitz surrogate, evaluate children, FE evaluate best potential child and add to bank

            # print(self.feTargets, self.feFeatures)
            # print('final shape', self.feFeatures.shape)

            L_est = estimate_lipschitz_constant(self.feFeatures, self.scalarisedTargets)

            popOnLipschitz = lipschitz_global_underestimate(
                self.scalarisedTargets, self.feFeatures, L_est, self.population
            )
            best_idx = np.argmin(popOnLipschitz)
            bestFeature = self.population[best_idx]

            newObjectiveTargets = MOobjective_function(
                self.population[best_idx], self.objFunction, self.nObjectives
            )
            self.objectiveTargets = np.vstack(
                (self.objectiveTargets, newObjectiveTargets)
            )
            self.feFeatures = np.vstack((self.feFeatures, self.population[best_idx]))

            # find minimum in boths columns - new zbest values

            self.zbests = np.min(self.objectiveTargets, axis=0)

            self.scalarisedTargets = scalariseValues(
                self.scalarisingFunction,
                self.objectiveTargets,
                self.zbests,
                self.weights,
                iteration,
                self.max_generations,
            )
            self.feFeatures, self.scalarisedTargets, self.objectiveTargets = removeNans(
                self.feFeatures, self.scalarisedTargets, self.objectiveTargets
            )

            # # evaluating all points in function on Lipschitz for plotting purposes
            # Z_under = lipschitz_global_underestimate(
            #     self.scalarisedTargets, self.feFeatures, L_est, fullRangeArray
            # )

            # plt.scatter(fullRangeArray[:,0], fullRangeArray[:,1], c = Z_under)

            # plt.scatter(self.population[:, 0], self.population[:, 1], color='red', label='Final Population', s=5)
            # plt.scatter(bestFeature[0], bestFeature[1], color='blue', label='Best Solution', s=10)
            # # plt.legend()
            # plt.title("Lipschitz Underestimation")
            # plt.colorbar()
            # plt.clim(np.min(Z_under), np.max(Z_under))
            # plt.savefig('lipschitz.png')
            # plt.close()

            # construct local RBF using c best solutions, find minima using DE, and evaluate at that minima

            bestLocalSolution = self.localRBF(15)

            bestLocalSolution = np.reshape(bestLocalSolution, (self.dimensions,))

            # print("best local Solution", bestLocalSolution)

            newObjectiveTargets = MOobjective_function(
                bestLocalSolution, self.objFunction, self.nObjectives
            )
            self.objectiveTargets = np.vstack(
                (self.objectiveTargets, newObjectiveTargets)
            )
            self.feFeatures = np.vstack((self.feFeatures, bestLocalSolution))

            # find minimum in boths columns - new zbest values

            self.zbests = np.min(self.objectiveTargets, axis=0)

            self.scalarisedTargets = scalariseValues(
                self.scalarisingFunction,
                self.objectiveTargets,
                self.zbests,
                self.weights,
                iteration,
                self.max_generations,
            )
            self.feFeatures, self.scalarisedTargets, self.objectiveTargets = removeNans(
                self.feFeatures, self.scalarisedTargets, self.objectiveTargets
            )

            # plt.scatter(self.objectiveTargets[:,0], self.objectiveTargets[:,1], c = self.scalarisedTargets)
            # plt.title('Evaluated Population')
            # plt.colorbar()
            # plt.clim(np.min(self.scalarisedTargets), np.max(self.scalarisedTargets))
            # plt.savefig('population.png')
            # plt.close()

            # Track the best solution
            # best_idx = np.argmin([objective_function(ind) for ind in self.population])
            # best_fitness = objective_function(self.population[best_idx])

            # if best_fitness < self.best_fitness:
            #     self.best_fitness = best_fitness
            #     self.best_solution = self.population[best_idx]

            # plt.contourf(X, Y, Z, levels=50, cmap='viridis')
            # plt.scatter(de.population[:, 0], de.population[:, 1], color='red', label='Final Population', s=5)
            # # plt.scatter(best_solution[0], best_solution[1], color='blue', label='Best Solution', s=100)
            # plt.legend()
            # plt.title("Ackley Function with Final Population and Best Solution")
            # plt.colorbar()
            # plt.show()
            # Debug information
            # print(f"Generation {generation + 1}: Best Fitness = {self.best_fitness}")
            print(
                f"LSADE Iteration {iteration}, Best found solution = ",
                min(self.scalarisedTargets),
            )
            print(f"Evaluated points = {len(self.feFeatures)}")

            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            # localGP = Image.open('localGP.png')
            # globalGP = Image.open('globalGP.png')
            # lipschitz = Image.open('lipschitz.png')
            # population = Image.open('population.png')

            # width, height = localGP.size
            # combinedImage = Image.new('RGB', (2 * width, 2 * height))
            # combinedImage.paste(localGP, (0, 0))
            # combinedImage.paste(lipschitz, (width, 0))
            # combinedImage.paste(globalGP, (0, height))
            # combinedImage.paste(population, (width, height))

            # combinedImage.save(f'surrogatePlots/{iteration}.png')

            np.savetxt("LSADEFeatures.txt", self.feFeatures)
            np.savetxt("LSADEScalarisedTargets.txt", self.scalarisedTargets)
            np.savetxt("LSADEObjectiveTargets.txt", self.objectiveTargets)
            iteration += 1

        return self.best_solution, self.best_fitness


class ESA:
    def __init__(
        self,
        bounds,
        pop_size,
        localPopSize,
        alpha,
        objFunction,
        scalarisingFunction,
        nObjectives,
        weights,
        gamma,
        maxFE,
        useInitialPopulation,
        initialPopulation,
        initialObjvValues,
        mutation_factor=0.8,
        crossover_prob=0.7,
    ):
        self.globalBounds = np.array(bounds)
        self.dimensions = len(bounds)
        self.pop_size = pop_size
        self.nObjectives = nObjectives
        self.objectiveTargets = np.empty((0, self.nObjectives))
        self.scalarisedTargets = np.empty(0)
        self.feFeatures = np.empty((0, self.dimensions))
        # self.feTargets = np.empty(0)
        # self.k = k
        self.objFunction = objFunction

        self.scalarisingFunction = scalarisingFunction
        self.zbests = np.full((1, self.nObjectives), np.inf)
        self.weights = weights
        if useInitialPopulation == True:
            self.population = initialPopulation
            self.objectiveTargets = initialObjvValues
        else:
            self.population = self.initialisePopulation()
            self.evaluateInitialPopulation()
        self.scalariseInitialPopulation()
        self.localPopSize = localPopSize

        # initialise 4actions x 8states array, all entries initialised to 0.25
        self.qTable = np.full((8, 4), 0.1)
        self.x_best = 0
        self.x_bestSolution = 0
        self.alpha = alpha
        self.gamma = gamma
        self.maxFE = maxFE
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob

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
        self.feFeatures, self.scalarisedTargets, self.objectiveTargets = removeNans(
            self.feFeatures, self.scalarisedTargets, self.objectiveTargets
        )

        # plt.scatter(self.feFeatures[:,0], self.feFeatures[:,1], c = self.feTargets)
        # plt.title('Initial Population')
        # plt.colorbar()
        # plt.show()

    def mutate(self, target_idx, currentGP):
        """Mutation using DE/best/1 strategy."""
        # Choose three random and distinct individuals different from target_idx
        indices = [idx for idx in range(self.pop_size) if idx != target_idx]
        np.random.shuffle(indices)
        r1, r2, r3 = indices[:3]

        predictedValues = GPEval(currentGP, self.population)

        best_idx = np.argsort(predictedValues)[:1]

        best = self.population[best_idx]

        # Mutant vector: v = best + F * (r1 - r2)
        mutant = best + self.mutation_factor * (
            self.population[r1] - self.population[r2]
        )

        # Ensure mutant vector is within bounds
        mutant = np.clip(mutant, self.globalBounds[:, 0], self.globalBounds[:, 1])

        return mutant

    def crossover(self, target, mutant):
        """Crossover to create trial vector.
        This loops through the features in the vector (dimensions) and sees if any of them crossover.
        allows the retention of some original vector features but not others.
        """

        trial = np.copy(target)
        # print(trial.shape)
        # print(mutant.shape)
        for i in range(self.dimensions):
            if np.random.rand() < self.crossover_prob or i == np.random.randint(
                self.dimensions
            ):
                # print(trial[i], mutant[i])

                trial[i] = mutant[i]
        return trial

    def runNextAction(self, currentAction, iteration):
        if currentAction == 0:
            print("Running a1!")
            newBestFeature, newBestTarget = self.a1(iteration)

            if newBestTarget < self.x_bestSolution:
                self.x_bestSolution = newBestTarget
                currentState = 1
                r = 1

            else:
                currentState = 0
                r = 0

        if currentAction == 1:
            print("Running a2!")

            newBestFeature, newBestTarget = self.a2(iteration)

            if newBestTarget < self.x_bestSolution:
                self.x_bestSolution = newBestTarget
                currentState = 3
                r = 1

            else:
                currentState = 2
                r = 0

        if currentAction == 2:
            print("Running a3!")

            newBestFeature, newBestTarget = self.a3(iteration)

            if newBestTarget < self.x_bestSolution:
                self.x_bestSolution = newBestTarget
                currentState = 5
                r = 1

            else:
                currentState = 4
                r = 0

        if currentAction == 3:
            print("Running a4!")

            returnedFeatures, returnedTargets = self.a4(iteration)
            # print('returned', returnedTargets, returnedTargets.shape)

            for i in range(0, 3):
                if returnedTargets[i] < self.x_bestSolution:
                    self.x_bestSolution = returnedTargets[i]
                    currentState = 7
                    r = 1

                else:
                    currentState = 6
                    r = 0

            newBestTarget = np.min(returnedTargets)

        return currentState, r, newBestTarget

    def mainMenu(self, initialAction):
        """
        handle all the q-value and q-table stuff here
        such as calculating rewards, probabilities of selecting next states,
        and running the next algorithm

        set initialAction to be a random number (from 0-3) to select initial algorithm to use
        """

        bestIndex = np.argmin(self.scalarisedTargets)
        self.x_best = self.feFeatures[bestIndex]

        self.x_bestSolution = self.scalarisedTargets[bestIndex]

        currentAction = initialAction

        iteration = 0

        currentState, r, mostRecentValue = self.runNextAction(currentAction, iteration)

        # calculate softmax selection strategy

        actions = np.arange(4)

        # loop from here?

        while len(self.feFeatures) < (self.pop_size + self.maxFE):
            actionProb = np.empty((1, 4))

            for i in range(0, 4):
                actionProb[0, i] = (np.exp(self.qTable[currentState, i])) / np.sum(
                    np.exp(self.qTable[currentState])
                )

            previousState = currentState

            # Select an action based on the probabilities in actionProb[1]
            next_action = np.random.choice(actions, p=actionProb[0])

            currentState, r, mostRecentValue = self.runNextAction(
                next_action, iteration
            )
            # print("At iteration: ", iteration, " current state: ", currentState)

            # calculate the new q value for the action taken in the PREVIOUS state

            self.qTable[previousState, next_action] += self.alpha * (
                r
                + (self.gamma * np.max(self.qTable[currentState, :]))
                - self.qTable[previousState, next_action]
            )

            print("QTable at iteration ", iteration)
            print(self.qTable)

            # if iteration % 5 == 0:
            # plt.scatter(self.feFeatures[:,0], self.feFeatures[:,1], c=self.scalarisedTargets)
            # plt.scatter(self.feFeatures[-1,0], self.feFeatures[-1,1], color='black', marker='x')
            # plt.title(f'Population at iteration {iteration}')
            # plt.colorbar()
            # plt.show()
            # plt.savefig(f'ESAIteration{iteration}.png')

            # plt.close()

            print(f"ESA Best result at iteration {iteration}", self.x_bestSolution)
            print(f"Evaluated points = {len(self.feFeatures)}")

            # fig, ax = plt.subplots()
            # cax = ax.matshow(np.ndarray.transpose(self.qTable), cmap="binary", vmin = 0, vmax = 1, aspect=1)

            # # Add color bar to show scale
            # fig.colorbar(cax)

            # # Set custom tick labels for x and y axes
            # x_labels = ['1', '2', '3', '4', '5', '6', '7', '8']  # 8 columns
            # y_labels = ['A1', 'A2', 'A3', 'A4']       # 4 rows

            # ax.set_xticks(np.arange(len(x_labels)))  # Set x-ticks to match number of columns
            # ax.set_yticks(np.arange(len(y_labels)))  # Set y-ticks to match number of rows
            # ax.tick_params(top=True, labeltop=True, labelbottom=False, bottom=False)

            # ax.set_xticklabels(x_labels)  # Set x-tick labels
            # ax.set_yticklabels(y_labels)  # Set y-tick labels

            # ax.set_xlabel('State' )
            # ax.xaxis.set_label_position('top')
            # ax.set_ylabel("Action")
            # # Add grid
            # ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
            # ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)
            # ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
            # ax.tick_params(which="minor", size=0)  # Hide minor ticks but keep grid
            # plt.text(-0.5,4,f'Previous State = {previousState + 1}')
            # plt.text(-0.5,4.5,f'Previous Action = {next_action + 1}')
            # plt.text(-0.5,5,f'Reward = {r}')
            # plt.text(3,4,f'Current Best Value = {round(self.x_bestSolution, 5)}')
            # plt.text(3,5,f'Last Found Value = {round(mostRecentValue, 5)}')

            # plt.savefig(f"ESAQTablePlots/{iteration}.png")
            # plt.close()
            np.savetxt("ESAFeatures.txt", self.feFeatures)
            np.savetxt("ESAScalarisedTargets.txt", self.scalarisedTargets)
            np.savetxt("ESAObjectiveTargets.txt", self.objectiveTargets)

            iteration += 1

        # algorithm:
        # select initial action randomly
        # run action to determine initial state, updating no values yet
        # choose new action and run algorithm. This gives FIRST state-action pair
        # depending on result, update q value for state action pair.
        # repeat.

    def a1(self, globalIteration):
        GPModel = GPTrain(self.feFeatures, self.scalarisedTargets, meanPrior="max")

        new_population = np.zeros_like(self.population)

        for i in range(self.pop_size):
            # print(i)
            target = self.population[i]
            mutant = self.mutate(i, GPModel)
            mutant = np.reshape(mutant, (self.dimensions,))

            trial = self.crossover(target, mutant)
            trial = np.reshape(trial, (1, -1))
            target = np.reshape(target, (1, -1))
            new_population[i] = trial

            # Update the population
        self.population = new_population
        popOnGP = GPEval(GPModel, self.population)

        # evaluating whole landscape on RBF for plotting reasons:
        # x_range = np.linspace(-5, 5, 50)
        # y_range = np.linspace(-5, 5, 50)
        # fullRange = list(product(x_range, y_range))
        # fullRangeArray = np.array(fullRange)
        # y_pred = GPEval(GPModel, fullRangeArray)

        # evaluate current population (children) on RBF
        # popOnRBF = globalRBF.predict(self.population)

        # function evaluation of best predicted child
        best_idx = np.argmin(popOnGP)

        bestFeature = self.population[best_idx]

        # print('best index =', best_idx)
        # evaluate best child and add results to global stores of FE features and targets
        newObjectiveTargets = MOobjective_function(
            bestFeature, self.objFunction, self.nObjectives
        )
        self.objectiveTargets = np.vstack((self.objectiveTargets, newObjectiveTargets))
        self.feFeatures = np.vstack((self.feFeatures, bestFeature))

        # find minimum in boths columns - new zbest values

        self.zbests = np.min(self.objectiveTargets, axis=0)

        self.scalarisedTargets = scalariseValues(
            self.scalarisingFunction,
            self.objectiveTargets,
            self.zbests,
            self.weights,
            globalIteration,
            self.maxFE,
        )
        self.feFeatures, self.scalarisedTargets, self.objectiveTargets = removeNans(
            self.feFeatures, self.scalarisedTargets, self.objectiveTargets
        )

        return self.feFeatures[-1], self.scalarisedTargets[-1]

    def a2(self, globalIteration):
        bestFeatures = np.empty((self.localPopSize, self.dimensions))
        bestTargets = np.empty(self.localPopSize)

        # find c best solutions
        bestIndices = np.argsort(self.scalarisedTargets)[: self.localPopSize]

        for i in range(self.localPopSize):
            bestFeatures[i] = self.feFeatures[bestIndices[i]]
            bestTargets[i] = self.scalarisedTargets[bestIndices[i]]

        # x_min, x_max = np.min(bestFeatures[:, 0]), np.max(bestFeatures[:, 0])
        # y_min, y_max = np.min(bestFeatures[:, 1]), np.max(bestFeatures[:, 1])

        # bounds = [(x_min, x_max), (y_min, y_max)]

        bounds = [(np.min(bestFeatures[:, d]), np.max(bestFeatures[:, d])) for d in range(bestFeatures.shape[1])]

        # pairwiseDistancesLocal = np.linalg.norm(bestFeatures[:, np.newaxis] - bestFeatures, axis=2)
        # avgDistanceLocal = np.mean(pairwiseDistancesLocal)

        localGP = GPTrain(bestFeatures, bestTargets, meanPrior="max")

        # localRBF = RBFSurrogateModel(epsilon=1.0)
        # localRBF.fit(bestFeatures, bestTargets)

        # functionEval = localRBF.predict()
        localDE = DifferentialEvolution(bounds, localGP)
        bestLocalSolution, bestLocalFitness = localDE.optimize()
        bestLocalSolution = np.reshape(bestLocalSolution, (self.dimensions,))

        newObjectiveTargets = MOobjective_function(
            bestLocalSolution, self.objFunction, self.nObjectives
        )
        self.objectiveTargets = np.vstack((self.objectiveTargets, newObjectiveTargets))
        self.feFeatures = np.vstack((self.feFeatures, bestLocalSolution))

        # find minimum in boths columns - new zbest values

        self.zbests = np.min(self.objectiveTargets, axis=0)

        self.scalarisedTargets = scalariseValues(
            self.scalarisingFunction,
            self.objectiveTargets,
            self.zbests,
            self.weights,
            globalIteration,
            self.maxFE,
        )
        self.feFeatures, self.scalarisedTargets, self.objectiveTargets = removeNans(
            self.feFeatures, self.scalarisedTargets, self.objectiveTargets
        )

        return self.feFeatures[-1], self.scalarisedTargets[-1]

    def a3(self, globalIteration):
        # build surrogate using all points in population
        crossoverGP = GPTrain(self.feFeatures, self.scalarisedTargets, meanPrior="zero")

        best_idx = np.argmin(self.scalarisedTargets)
        bestFeature = self.feFeatures[best_idx]
        bestTarget = self.scalarisedTargets[best_idx]

        RVS = random.sample(range(0, self.dimensions), self.dimensions)

        # print(self.feFeatures.shape)
        # print(bestFeature)

        tempPop = np.full_like((self.feFeatures), bestFeature)

        # print(tempPop)

        for i in RVS:
            # print(i)

            tempPop[:, i] = self.feFeatures[:, i]

            tempPopEval = GPEval(crossoverGP, tempPop)

            tempPopBestIdx = np.argmin(tempPopEval)
            if tempPopEval[tempPopBestIdx] > bestTarget:
                tempPop = np.full_like((self.feFeatures), tempPop[tempPopBestIdx])
                # print('new best value!')

        # take final best predicted value and explicitely evaluate
        newObjectiveTargets = MOobjective_function(
            tempPop[tempPopBestIdx], self.objFunction, self.nObjectives
        )
        self.objectiveTargets = np.vstack((self.objectiveTargets, newObjectiveTargets))
        self.feFeatures = np.vstack((self.feFeatures, tempPop[tempPopBestIdx]))

        # find minimum in boths columns - new zbest values

        self.zbests = np.min(self.objectiveTargets, axis=0)

        self.scalarisedTargets = scalariseValues(
            self.scalarisingFunction,
            self.objectiveTargets,
            self.zbests,
            self.weights,
            globalIteration,
            self.maxFE,
        )
        self.feFeatures, self.scalarisedTargets, self.objectiveTargets = removeNans(
            self.feFeatures, self.scalarisedTargets, self.objectiveTargets
        )

        return self.feFeatures[-1], self.scalarisedTargets[-1]

    def find_closest_points(self, points, selected_point_index, n):
        # Convert points to a NumPy array for easier manipulation
        # points = np.array(points)

        # Get the selected point from the array
        selected_point = points[selected_point_index]

        # Calculate the Euclidean distance from the selected point to each other point
        distances = np.linalg.norm(points - selected_point, axis=1)

        # Exclude the selected point itself by setting its distance to infinity
        distances[selected_point_index] = np.inf

        # Find the indices of the n smallest distances
        closest_indices = np.argpartition(distances, n)[:n]

        # Get the n closest points
        closest_points = points[closest_indices]

        return closest_points

    def a4(self, globalIteration):
        iteration = 0

        while iteration < 3:
            print("Trust Region iteration = ", iteration + 1)

            # This handles x_best updating
            bestIndex = np.argmin(self.scalarisedTargets)
            x_best = self.feFeatures[bestIndex]

            x_bestSolution = self.scalarisedTargets[bestIndex]

            if iteration == 0:
                nPoints = self.dimensions * 5

                closestPoints = self.find_closest_points(
                    self.feFeatures, bestIndex, nPoints
                )

                bounds_min = np.min(closestPoints, axis=0)
                bounds_max = np.max(closestPoints, axis=0)

                # Calculate sigma for each dimension
                sigma = (bounds_max - bounds_min) / 2

            # Compute trust region bounds for all dimensions
            lower_bound_trust = x_best - sigma
            upper_bound_trust = x_best + sigma

            # Create trust region bounds as a list of tuples
            trustRegionBounds = [
                (lower_bound_trust[i], upper_bound_trust[i])
                for i in range(self.dimensions)
            ]

            # Check if points fall within the trust region for all dimensions
            in_area = np.all(
                (self.feFeatures >= lower_bound_trust)
                & (self.feFeatures <= upper_bound_trust),
                axis=1,
            )

            # Filter features and targets within the trust region
            trustRegionFeatures = self.feFeatures[in_area]
            trustRegionTargets = self.scalarisedTargets[in_area]

            try:
                # build surrogate on points in the trust region
                trustRegionGP = GPTrain(
                    trustRegionFeatures, trustRegionTargets, meanPrior="max"
                )

                trustRegionDE = DifferentialEvolution(trustRegionBounds, trustRegionGP)
                trustBestSolution, trustBestFitness = trustRegionDE.optimize()

                trustBestSolution = np.reshape(trustBestSolution, (self.dimensions,))

                trustBestSolution = np.clip(
                    trustBestSolution, self.globalBounds[:, 0], self.globalBounds[:, 1]
                )

                # print('trust region best Solution', trustBestSolution)

                newObjectiveTargets = MOobjective_function(
                    trustBestSolution, self.objFunction, self.nObjectives
                )
                self.objectiveTargets = np.vstack(
                    (self.objectiveTargets, newObjectiveTargets)
                )
                self.feFeatures = np.vstack((self.feFeatures, trustBestSolution))

                # find minimum in boths columns - new zbest values

                self.zbests = np.min(self.objectiveTargets, axis=0)

                self.scalarisedTargets = scalariseValues(
                    self.scalarisingFunction,
                    self.objectiveTargets,
                    self.zbests,
                    self.weights,
                    globalIteration,
                    self.maxFE,
                )
                self.feFeatures, self.scalarisedTargets, self.objectiveTargets = (
                    removeNans(
                        self.feFeatures, self.scalarisedTargets, self.objectiveTargets
                    )
                )

                # calculate trust ratio

                rho_k = (x_bestSolution - self.scalarisedTargets[-1] + 1e-6) / (
                    x_bestSolution - trustBestFitness + 1e-6
                )

                # print('x_bestSolution =', x_bestSolution)
                # print('last solution =', self.scalarisedTargets[-1])
                # print('trustBestFitness =', trustBestFitness)
                # print('top =', (x_bestSolution - self.scalarisedTargets[-1]))
                # print('bottom =', (
                #     x_bestSolution - trustBestFitness
                # ))
                # print('rho_k = ', rho_k)
                # print('old sigma =', sigma)

                epsilon = 1.5

                if rho_k < 0.25:
                    sigma = 0.25 * sigma

                elif rho_k > 0.25 and rho_k < 0.75:
                    sigma = sigma  # is there something smarter i can do?

                elif rho_k > 0.75:
                    sigma = epsilon * sigma

                # print('new sigma = ', sigma)

                # plt.scatter(self.feFeatures[:,0], self.feFeatures[:,1], c = self.feTargets)
                # plt.title('Evaluated Population')
                # plt.colorbar()
                # plt.show()

                iteration += 1
            except RuntimeError:
                print("Error in training GP Surrogate, skipping...")
                iteration += 1

            # return however many iterations worth of features/targets
        return self.feFeatures[-3:], self.scalarisedTargets[-3:]


def BOGPEval(model, newFeatures):
    model.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = model(torch.from_numpy(newFeatures))

    mean_pred = observed_pred.mean.numpy()
    stdDev = observed_pred.stddev.numpy()

    return mean_pred, stdDev


def expectedImprovement(currentGP, feature, bestY, epsilon):
    yPred, yStd = BOGPEval(currentGP, feature)

    # TODO check that signs are the correct way round in ei and z equations.

    z = (bestY - yPred - epsilon) / yStd
    ei = ((bestY - yPred - epsilon) * norm.cdf(z)) + yStd * norm.pdf(z)
    return ei


def upperConfidenceBounds(currentGP, feature, bestY, Lambda):
    yPred, yStd = BOGPEval(currentGP, feature)

    a = yPred + (Lambda * yStd)

    return a


def probabilityOfImprovement(currentGP, feature, bestY, epsilon):
    # NO EPSILON IN THIS EQUATION. THATS ONLY FOR EI
    # THIS Z EQUATION IS FOR MAXIMISATION.
    yPred, yStd = BOGPEval(currentGP, feature)
    z = (bestY - yPred) / yStd
    pi = norm.cdf(z)

    return pi


class bayesianOptimiser:
    def __init__(
        self,
        bounds,
        pop_size,
        objFunction,
        scalarisingFunction,
        nObjectives,
        weights,
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
        self.feFeatures, self.scalarisedTargets, self.objectiveTargets = removeNans(
            self.feFeatures, self.scalarisedTargets, self.objectiveTargets
        )

        # for i in range(len(self.objectiveTargets)):
        #     print(self.objectiveTargets[i], self.scalarisedTargets[i])

        # plt.scatter(self.feFeatures[:,0], self.feFeatures[:,1], c = self.feTargets)
        # plt.title('Initial Population')
        # plt.colorbar()
        # plt.show()

    def runOptimiser(self):
        iteration = 0

        # while self.bestEI > 1e-7:
        # while iteration < 80:
        while len(self.feFeatures) < self.maxFE:
            best_idx = np.argmin(self.scalarisedTargets)
            bestFeature = self.feFeatures[best_idx]
            bestTarget = self.scalarisedTargets[best_idx]
            # print(bestTarget)

            # numSolutions = self.pop_size

            # bestFeatures = np.empty((numSolutions, self.dimensions))
            # bestTargets = np.empty(numSolutions)

            # find c best solutions
            # bestIndices = np.argsort(self.scalarisedTargets)[:numSolutions]

            # for i in range(numSolutions):
            #     bestFeatures[i] = self.feFeatures[bestIndices[i]]
            #     bestTargets[i] = self.scalarisedTargets[bestIndices[i]]

            # x_min, x_max = np.min(bestFeatures[:, 0]), np.max(bestFeatures[:, 0])
            # y_min, y_max = np.min(bestFeatures[:, 1]), np.max(bestFeatures[:, 1])

            # localBounds = [(x_min, x_max), (y_min, y_max)]

            # localBounds = [(np.min(bestFeatures[:, d]), np.max(bestFeatures[:, d])) for d in range(bestFeatures.shape[1])]

            # pairwiseDistancesLocal = np.linalg.norm(bestFeatures[:, np.newaxis] - bestFeatures, axis=2)
            # avgDistanceLocal = np.mean(pairwiseDistancesLocal)

            # localGP = GPTrain(bestFeatures, bestTargets, meanPrior="max")

            # localRBF = RBFSurrogateModel(epsilon=1.0)
            # localRBF.fit(bestFeatures, bestTargets)

            # functionEval = localRBF.predict()
            # localDE = DifferentialEvolution(bounds, localGP)

            # this is the original training call
            globalGP = GPTrain(
                self.feFeatures, self.scalarisedTargets, meanPrior="zero"
            )

            # evaluating whole landscape on RBF for plotting reasons:
            # x_range = np.linspace(self.globalBounds[0, 0], self.globalBounds[0, 1], 100)
            # y_range = np.linspace(self.globalBounds[1, 0], self.globalBounds[1, 1], 100)
            # fullRange = list(product(x_range, y_range))
            # fullRangeArray = np.array(fullRange)
            # y_pred, ystd = BOGPEval(localGP, fullRangeArray)

            # print(fullRangeArray.shape, y_pred.shape)

            # plt.scatter(fullRangeArray[:,0], fullRangeArray[:,1], c=y_pred)
            # plt.title("Global Surrogate")
            # plt.colorbar()
            # plt.clim(1e-5, 1e2)
            # # plt.yscale('log')

            # plt.savefig('eiGS.png')
            # plt.close()

            eiDEGlobal = BayesianDifferentialEvolution(globalGP, self.globalBounds, bestTarget)
            newSolution, newFitness = eiDEGlobal.optimize()

            # print("newsol", newSolution.shape)

            # plt.scatter(fullRangeArray[:,0], fullRangeArray[:,1], c = y_pred, alpha = 0.5)
            # plt.scatter(newSolution[0], newSolution[1], color='blue', label='Best Solution', s=10)
            # plt.legend()
            # plt.title("DE Optimisation of Expected Improvement")
            # plt.colorbar()
            # # plt.yscale('log')
            # plt.clim(np.min(y_pred), np.max(y_pred))
            # # plt.savefig('eiDE.png')
            # plt.show()

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
            self.feFeatures, self.scalarisedTargets, self.objectiveTargets = removeNans(
                self.feFeatures, self.scalarisedTargets, self.objectiveTargets
            )

            # for i in range(len(self.objectiveTargets)):
            #     print(self.objectiveTargets[i], self.scalarisedTargets[i])

            print(f"BO Iteration {iteration}, Best found solution = ", bestTarget)
            print(f"Evaluated points = {len(self.feFeatures)}")

            # surrogate = Image.open('eiGS.png')
            # population = Image.open('eiDE.png')

            # width, height = population.size
            # combinedImage = Image.new('RGB', (2 * width, height), "WHITE")
            # combinedImage.paste(population, (0, 0))
            # combinedImage.paste(surrogate, (width, 0))

            # combinedImage.save(f'{iteration}.png')

            self.bestEI = newFitness

            positions = np.arange(len(self.scalarisedTargets))

            # plt.scatter(
            #     self.objectiveTargets[:, 0], self.objectiveTargets[:, 1], c=positions
            # )
            # plt.title(f"Pareto Front, iteration {iteration}")
            # plt.colorbar()
            # plt.clim(0, len(self.scalarisedTargets))
            # plt.xlabel("f2(x)")
            # plt.ylabel("f1(x)")
            # # plt.savefig("BOPareto.png")
            # # plt.close()
            # plt.show()

            np.savetxt("BOFeatures.txt", self.feFeatures)
            np.savetxt("BOScalarisedTargets.txt", self.scalarisedTargets)
            np.savetxt("BOObjectiveTargets.txt", self.objectiveTargets)

            iteration += 1




class BOZeroMax:
    def __init__(
        self,
        bounds,
        pop_size,
        objFunction,
        scalarisingFunction,
        nObjectives,
        weights,
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
        self.feFeatures, self.scalarisedTargets, self.objectiveTargets = removeNans(
            self.feFeatures, self.scalarisedTargets, self.objectiveTargets
        )

        # for i in range(len(self.objectiveTargets)):
        #     print(self.objectiveTargets[i], self.scalarisedTargets[i])

        # plt.scatter(self.feFeatures[:,0], self.feFeatures[:,1], c = self.feTargets)
        # plt.title('Initial Population')
        # plt.colorbar()
        # plt.show()

    def runOptimiser(self):
        iteration = 0

        # while self.bestEI > 1e-7:
        # while iteration < 40:
        while len(self.feFeatures) < (self.pop_size + self.maxFE):

            best_idx = np.argmin(self.scalarisedTargets)
            bestFeature = self.feFeatures[best_idx]
            bestTarget = self.scalarisedTargets[best_idx]
            # print(bestTarget)

            numSolutions = self.pop_size

            bestFeatures = np.empty((numSolutions, self.dimensions))
            bestTargets = np.empty(numSolutions)

            # find c best solutions
            bestIndices = np.argsort(self.scalarisedTargets)[:numSolutions]

            for i in range(numSolutions):
                bestFeatures[i] = self.feFeatures[bestIndices[i]]
                bestTargets[i] = self.scalarisedTargets[bestIndices[i]]

            # x_min, x_max = np.min(bestFeatures[:, 0]), np.max(bestFeatures[:, 0])
            # y_min, y_max = np.min(bestFeatures[:, 1]), np.max(bestFeatures[:, 1])

            # localBounds = [(x_min, x_max), (y_min, y_max)]

            localBounds = [(np.min(bestFeatures[:, d]), np.max(bestFeatures[:, d])) for d in range(bestFeatures.shape[1])]

            # pairwiseDistancesLocal = np.linalg.norm(bestFeatures[:, np.newaxis] - bestFeatures, axis=2)
            # avgDistanceLocal = np.mean(pairwiseDistancesLocal)

            localGP = GPTrain(bestFeatures, bestTargets, meanPrior="max")

            # localRBF = RBFSurrogateModel(epsilon=1.0)
            # localRBF.fit(bestFeatures, bestTargets)

            # functionEval = localRBF.predict()
            # localDE = DifferentialEvolution(bounds, localGP)

            #now perform a global zero mean search

            # this is the original training call
            globalGP = GPTrain(
                self.feFeatures, self.scalarisedTargets, meanPrior="zero"
            )

            # evaluating whole landscape on RBF for plotting reasons:
            # x_range = np.linspace(self.globalBounds[0, 0], self.globalBounds[0, 1], 100)
            # y_range = np.linspace(self.globalBounds[1, 0], self.globalBounds[1, 1], 100)
            # fullRange = list(product(x_range, y_range))
            # fullRangeArray = np.array(fullRange)
            # y_predG, ystdG = BOGPEval(globalGP, fullRangeArray)
            # y_predM, ystdM = BOGPEval(localGP, fullRangeArray)

            # # print(fullRangeArray.shape, y_pred.shape)

            # plt.scatter(fullRangeArray[:,0], fullRangeArray[:,1], c=y_predG)
            # plt.scatter(self.feFeatures[:,0], self.feFeatures[:,1], c = self.objectiveTargets[:,2])
            # plt.title("Global Surrogate")
            # plt.colorbar()
            # plt.show()

            # plt.scatter(fullRangeArray[:,0], fullRangeArray[:,1], c=y_predM)
            # plt.scatter(self.feFeatures[:,0], self.feFeatures[:,1], c = self.objectiveTargets[:,2])
            # plt.title("Local Surrogate")
            # plt.colorbar()
            # plt.show()


            # plt.clim(1e-5, 1e2)
            # # plt.yscale('log')

            # plt.savefig('eiGS.png')
            # plt.close()

            eiDELocal = BayesianDifferentialEvolution(localGP, localBounds, bestTarget)
            newSolutionLocal, newFitnessLocal = eiDELocal.optimize()

            eiDEGlobal = BayesianDifferentialEvolution(globalGP, self.globalBounds, bestTarget)
            newSolutionGlobal, newFitnessGlobal = eiDEGlobal.optimize()
            # print("newsol", newSolution.shape)

            # plt.scatter(fullRangeArray[:,0], fullRangeArray[:,1], c = y_pred, alpha = 0.5)
            # plt.scatter(newSolution[0], newSolution[1], color='blue', label='Best Solution', s=10)
            # plt.legend()
            # plt.title("DE Optimisation of Expected Improvement")
            # plt.colorbar()
            # # plt.yscale('log')
            # plt.clim(np.min(y_pred), np.max(y_pred))
            # # plt.savefig('eiDE.png')
            # plt.show()

            newObjectiveTargets = MOobjective_function(
                newSolutionLocal, self.objFunction, self.nObjectives
            )
            self.objectiveTargets = np.vstack(
                (self.objectiveTargets, newObjectiveTargets)
            )
            self.feFeatures = np.vstack((self.feFeatures, newSolutionLocal))

            newObjectiveTargets = MOobjective_function(
                newSolutionGlobal, self.objFunction, self.nObjectives
            )
            self.objectiveTargets = np.vstack(
                (self.objectiveTargets, newObjectiveTargets)
            )
            self.feFeatures = np.vstack((self.feFeatures, newSolutionGlobal))

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
            self.feFeatures, self.scalarisedTargets, self.objectiveTargets = removeNans(
                self.feFeatures, self.scalarisedTargets, self.objectiveTargets
            )

            # for i in range(len(self.objectiveTargets)):
            #     print(self.objectiveTargets[i], self.scalarisedTargets[i])

            print(f"BO Iteration {iteration}, Best found solution = ", bestTarget)
            print(f"Evaluated points = {len(self.feFeatures)}")

            # surrogate = Image.open('eiGS.png')
            # population = Image.open('eiDE.png')

            # width, height = population.size
            # combinedImage = Image.new('RGB', (2 * width, height), "WHITE")
            # combinedImage.paste(population, (0, 0))
            # combinedImage.paste(surrogate, (width, 0))

            # combinedImage.save(f'{iteration}.png')

            # self.bestEI = newFitness

            # positions = np.arange(len(self.scalarisedTargets))

            # plt.scatter(
            #     self.objectiveTargets[:, 0], self.objectiveTargets[:, 1], c=positions
            # )
            # plt.title(f"Pareto Front, iteration {iteration}")
            # plt.colorbar()
            # plt.clim(0, len(self.scalarisedTargets))
            # plt.xlabel("f2(x)")
            # plt.ylabel("f1(x)")
            # # plt.savefig("BOPareto.png")
            # # plt.close()
            # plt.show()

            np.savetxt("BOMinMaxFeatures.txt", self.feFeatures)
            np.savetxt("BOMinMaxScalarisedTargets.txt", self.scalarisedTargets)
            np.savetxt("BOMinMaxObjectiveTargets.txt", self.objectiveTargets)

            iteration += 1