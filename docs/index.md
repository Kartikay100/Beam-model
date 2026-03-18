# Finite Element solvers for coupled Drillstring models

This repo contains code for open source solvers for modelling drillstring using finite element method.

## Features
- Solvers in this code are generic in nature which can be used to develop drillstring models.
- The spreadsheet in the verification folder provides guidelines for validating any drillstring model.
- The code shows the implementation of a finite strain beam model which can capture large deformation of a drill string or BHA with six degrees of freedom.

## Repository Structure
- `gen`: contains generic python functions like utilities, mesh1D, interpolation function etc. These functions are useful in the preprocessing and processing stage. Utilities.py contains a variety of different function from handling the data structures to calculation and interpolation of rotation pseudovectors. Details on each functions can be found in the documentation.

- `input`: contains files that define the input condition for different verfication cases. A python file has been created for each verification case which generates a JSON file. This JSON file is read by a python function to define the input. 

- `output`: folder for storing the calculations in JSON files.

- `Testing`: contains python notebooks with test cases for different functions present in gen folder. Various functions present in gen folder are tested for various cases to check if the output of the functions are as intended, before being used in the full model. This initial testing also helps understand the edge case behavior of the algorithm. 

- `Verification`: contains spreadsheet and powerpoint with verification cases. The spreadsheet contains inputs for the different cases, result of the model, reference of the problem and comparison with analytical or commercial model solution. The powerpoint contains plots of response of beam with analytical plots.

- `boundary.py`: reads the boundary condition defined in the input and applies them to element level equations.

- `main.py`: main python file which is run to solve.

- `solver.py`: file containing primary functions of finite element solver.

- `postprocess.py`: function for postprocessing of results. This function handles the output of the solver.py and plots them.

## Installation
```bash
# create/activate your environment (example)
## Installation
pip install -r requirements.txt

## Usage
 
### 1) Generate the input JSON file
This script writes `inputJSON.json` from predefined dictionaries or lists (initial conditions, geometry, material, boundary conditions, loads, distributed force and moment).

```bash
python inputJSON.py

### 2) Run main file
This script runs the main finite element solver. It reads the input from the `inputJSON.json` and calculates the response of the beam based on input conditions.

```bash
python main.py