MOOptimisers

This is a test suite for multi-objective optimisation. The suite includes:
- Three Surrogate-Assisted Evolutionary Algorithms
- A Bayesian Optimiser with choice of infill criterion
- 10 Scalarising functions for use with multi-objective optimisation cases
- 6 two-objective benchmarking functions
- A novel real-world test case based on the hyperparemeter tuning for the OpenFOAM simulation of a sand trap geometry
- Scripts for running the combinations of the above optimisers, functions, and scalarising functions
- Scripts for data handling and plotting of results.


Important Files

- optimiserBank.py - contains the code for the four optimisers
- functionBank.py - contains benchmarking functions, scalarising functions, and code for handling
                  the sand trap OpenFOAM simulations
- allrun.py - cath-all script for running benchmarking tests for optimisers
- sandTrapCaseDir - contains files read by OpenFOAM when running sand trap simulations. 
