# <Finite Element solvers for coupled Drillstring models>

<This repo contains code for open source solvers for modelling drillstring using finite element model.>

## Features
- <Solvers in this code are generic in nature which can be used to develop drillstring models.>
- <The spreadsheet in the verification folder provides guidelines for validating any drillstring model.>
- <>

## Repository Structure
- `gen`: contains generic python functions 
- `input`: contains files that define the input condition for different verfication cases.
- `output`: folder for storing the calculations in JSON files.
- `Testing`: contains python notebooks with test cases for different functions present in gen folder.
- `Verification`: contains spreadsheet and powerpoint with verification cases. 
- `boundary.py`: reads the boundary condition defined in the input and applies them to element level equations.
- `main.py`: main python file which is run to solve.
- `solver.py`: file containing primary functions of finite element solver.
- `postprocess.py`: function for postprocessing of results.

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