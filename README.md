Airline Crew Scheduling with Metaheuristic Algorithms
This repository contains a Python implementation of three metaheuristic algorithms designed to solve the airline crew scheduling problem. The algorithms are built from scratch and include:

Simulated Annealing (SA)
Standard Binary Genetic Algorithm (BGA)
Improved Binary Genetic Algorithm (Improved BGA)
The improved BGA integrates advanced features:

Pseudo-Random Initialization (based on Algorithm 2, p341 in [1])
Stochastic Ranking for Constraint Handling (as described in [2])
Heuristic Improvement Operator (based on Algorithm 1, p331 in [1])
Project Overview
The goal of this project is to efficiently cover all flights with the available crews while minimizing the overall cost. The algorithms have been designed to handle constraints such as ensuring every flight is covered exactly once. The code also plots the cost history over iterations and writes detailed results to text files.

Features
Three Algorithms: SA, Standard BGA, and Improved BGA.
Constraint Handling: Ensures each flight is covered exactly once.
Advanced Improvements: For the improved BGA, including pseudo-random initialization, stochastic ranking, and heuristic improvement.
Command-Line Interface: Easily select which algorithm to run and specify the dataset.
Visualization: Generates plots to show the evolution of cost over iterations.
Output Files: Results are saved to the output folder.
Requirements
Python 3.x
Required Python packages:
numpy
matplotlib
argparse
concurrent.futures (standard library)
os (standard library)
Other standard libraries: random, math
Getting Started
Clone the Repository:

bash
Copy
Edit
git clone https://github.com/yourusername/airline-crew-scheduling.git
cd airline-crew-scheduling
Install Dependencies:

If using pip, install the necessary packages:

bash
Copy
Edit
pip install numpy matplotlib
Prepare Your Dataset:

Place your dataset (e.g., sppnw41.txt) in the project directory or specify the correct path when running the program.

Usage
Run the program using the command line and choose the desired algorithm:

Simulated Annealing (SA):
python3 main.py --algorithm sa --dataset sppnw41.txt
Standard BGA:
python3 main.py --algorithm bga --dataset sppnw41.txt
Improved BGA:
python3 main.py --algorithm improved_bga --dataset sppnw41.txt
The program will execute the selected algorithm, display cost evolution plots, and write the detailed results to the output folder.

Implementation Details
Simulated Annealing (SA):
Randomly initializes a solution represented by a binary vector. It iteratively flips bits, calculates costs and coverage, and cools down the temperature while tracking the best solution.

Standard BGA:
Generates an initial population of binary vectors, evaluates fitness using a cost function, and applies tournament selection, crossover, and mutation operators.

Improved BGA:
Incorporates:

Pseudo-Random Initialization: Similar to stochastic local search for the set cover problem.
Stochastic Ranking: For constraint handling, balancing cost minimization with constraint violations.
Heuristic Improvement: To refine solutions by dropping redundant selections and adding missing coverage.
