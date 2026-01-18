# Parallel Cyclic Reduction (PCR) Solver

A C implementation of the Parallel Cyclic Reduction algorithm for solving
tridiagonal systems of linear equations.

## Overview

This project implements the Parallel Cyclic Reduction (PCR) algorithm, a
numerical method for solving tridiagonal systems of the form:

```
Ax = d
```

where A is a tridiagonal coefficient matrix, x is the solution vector, and d is
the right-hand side vector. PCR is particularly well-suited for parallel
computing due to its logarithmic number of stages and high degree of
parallelism.

## Project Structure

```
c/
├── README.md              # This file
├── Makefile               # Makefile for direct compilation
├── diagonal.h/c           # Diagonal matrix data structure
├── sle.h/c                # Tridiagonal system structure and utilities
├── solver.h               # PCR solver interface
├── util.h                 # Timing and utility macros
├── pcr.c                  # PCR algorithm implementation
└── main.c                 # Example program entry point
```

## Prerequisites

You need a C compiler:

```bash
module load gcc
```

## Compilation

```bash
# Compile all targets
make 

# Or compile specific targets
make pcrsolve
```

**Available Make targets:**

| Target      | Description                         |
| ----------- | ----------------------------------- |
| `all`       | Build all executables (default)     |
| `pcrsolve`  | Build PCR solver executable         |
| `clean`     | Remove object files and executables |
| `distclean` | Clean all generated files           |
| `help`      | Display help information            |

## Running the Code

### Basic Usage

The solver requires the system size (number of equations) as a command-line
argument:

```bash
srun -N 1 --exclusive -c 24 ./pcrsolve <number_of_equations>
```

### Program Output

The program produces the following output:

```
PCR solve time: X.XXXXXX sec
Max relative error: X.XXXXXXe+00
MAPE value: X.XXXXXXe+00%
```

**Output Explanation:**

| Line                 | Description                                                                          |
| -------------------- | ------------------------------------------------------------------------------------ |
| `PCR solve time`     | Total execution time for solving the system (in seconds)                             |
| `Max relative error` | Maximum relative error between computed and reference solutions, indicating accuracy |
| `MAPE value`         | Mean Absolute Percentage Error (%) - average relative error across all solutions     |
