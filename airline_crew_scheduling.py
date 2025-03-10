#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import os

os.makedirs("output", exist_ok=True)
def main():
    parser = argparse.ArgumentParser(description="Solve airline crew scheduling with SA/BGA/Improved BGA.")
    parser.add_argument("--algorithm", type=str, required=True, help="Algorithm (sa, bga, improved_bga)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset (e.g., sppnw41.txt)")
    args = parser.parse_args()

    # Load dataset
    data = load_dataset(args.dataset)

    if args.algorithm == "sa":
        result = simulated_annealing(data)
         # List to store costs and a list for writing results to file
        costs = []
        results_lines = []
        
        # Run bga 30 times
        for i in range(1):
            solution, cost, flight_covers = simulated_annealing(data)
            costs.append(cost)
            line = f"Run {i+1}: Cost: {cost}, Flight covers: {flight_covers}, Solution: {solution}"
            results_lines.append(line)
            if flight_covers != [1] * len(flight_covers):
                results_lines.append("No feasible solution found for SA")
            else:
                results_lines.append("Feasible solution found for SA")
        # Calculate average and standard deviation of the costs
        avg_cost = np.mean(costs)
        std_dev_cost = np.std(costs)
        
        # Append final statistics to the results
        results_lines.append("\nFinal Statistics:")
        results_lines.append(f"Average cost over 1 runs: {avg_cost:.2f}")
        results_lines.append(f"Standard deviation of cost: {std_dev_cost:.2f}")

        # Write the results to a text file
        with open("output/SA_results_43_oneshot.txt", "w") as file:
            for line in results_lines:
                file.write(line + "\n")
                
        print("Results saved to SA_results_43.txt")
        plt.show()

    elif args.algorithm == "bga":
        # List to store costs and a list for writing results to file
        costs = []
        results_lines = []
        
        # Run bga 30 times
        for i in range(1):
            solution, cost, flight_covers = standard_bga(data)
            costs.append(cost)
            line = f"Run {i+1}: Cost: {cost}, Flight covers: {flight_covers}, Solution: {solution}"
            results_lines.append(line)
            if flight_covers != [1] * len(flight_covers):
                results_lines.append("No feasible solution found for ")
        # Calculate average and standard deviation of the costs
        avg_cost = np.mean(costs)
        std_dev_cost = np.std(costs)
        
        # Append final statistics to the results
        results_lines.append("\nFinal Statistics:")
        results_lines.append(f"Average cost over 30 runs: {avg_cost:.2f}")
        results_lines.append(f"Standard deviation of cost: {std_dev_cost:.2f}")

        # Write the results to a text file
        with open("output/bga_results_41_oneshot.txt", "w") as file:
            for line in results_lines:
                file.write(line + "\n")
                
        print("Results saved to bga_results_41.txt")
        plt.show()
        return
          
    elif args.algorithm == "improved_bga":
        # List to store costs and a list for writing results to file
        costs = []
        results_lines = []
        
        # Run improved_bga 1 times
        for i in range(1):
            solution, cost, flight_covers = improved_bga(data)
            costs.append(cost)
            line = f"Run {i+1}: Cost: {cost}, Flight covers: {flight_covers}, Solution: {solution}"
            results_lines.append(line)
        
        # Calculate average and standard deviation of the costs
        avg_cost = np.mean(costs)
        std_dev_cost = np.std(costs)
        
        # Append final statistics to the results
        results_lines.append("\nFinal Statistics:")
        results_lines.append(f"Average cost over 30 runs: {avg_cost:.2f}")
        results_lines.append(f"Standard deviation of cost: {std_dev_cost:.2f}")

        # Write the results to a text file
        with open("improved_bga_results41_oneshot.txt", "w") as file:
            for line in results_lines:
                file.write(line + "\n")
                
        print("Results saved to improved_bga_results41_oneshot.txt")
        plt.show()
        return
    else:
        raise ValueError("Invalid algorithm")

def load_dataset(path):
    with open(path, 'r') as f:
        # Remove any empty lines and strip whitespace
        lines = [line.strip() for line in f if line.strip()]
    
    # First line: total flights to cover and number of rotations (crews)
    num_total_flights, num_crews = map(int, lines[0].split())
    
    costs = []   # To store the cost of each rotation
    flights = [] # To store a 2D list of flights covered by each rotation
    
    # Process each rotation (starting from the second line)
    for line in lines[1:]:
        parts = list(map(int, line.split()))
        cost = parts[0]  # First value is the cost
        num_flights = parts[1]  # Second value is the number of flights (not used explicitly)
        rotation_flights = [f - 1 for f in parts[2:2 + num_flights]]  # Convert to 0-based index
        
        costs.append(cost)
        flights.append(rotation_flights)
    
    return num_total_flights,num_flights, num_crews, costs, flights

def simulated_annealing(data):
    num_total_flights,num_flights, num_crews, costs, flights=data
    #represent solution in a binary vector
    def constraint_matrix_calculate(solution):
        constraint_matrix = np.zeros((num_total_flights,  num_crews))
        for i in range(len(solution)):
                if solution[i] == 1:  # Crew i is active
                    for flight in flights[i]:
                        if flight < num_total_flights:  # Ensure the flight index is valid
                            constraint_matrix[flight, i] = 1
                        else:
                            print(f"Warning: Flight {flight} is out of bounds for total flights {num_total_flights}")
        return constraint_matrix
    def check_coverage(solution):
        # Calculate the constraint matrix for the solution.
        constraint_matrix = constraint_matrix_calculate(solution)
        # Calculate the number of crews covering each flight.
        cover_count = row_sum_calculate(constraint_matrix)
        # Compute the average deviation from perfect coverage (which is 1)
        total_deviation = sum(abs(count - 1) for count in cover_count)
        avg_deviation = total_deviation / len(cover_count)
        # The coverage measure is 1 when perfect and >1 when deviations exist.
        coverage_measure = 1 + avg_deviation
        print("coverage_measure", coverage_measure)
        return coverage_measure

    def cost_calculation(best_solution): #objective function
        cost= np.dot(costs, best_solution)
        return cost
    def flip(solution, num_crews):
        new_solution = solution.copy()
        index=random.randint(0, num_crews - 1)
        new_solution[index] = abs(1- new_solution[index])
        return new_solution
    def row_sum_calculate(matrix):
        return [sum(row) for row in matrix]
    # Initialize the current and best solution to all zeros 
    current_solution = [random.choice([0, 1]) for _ in range(num_crews)]
    best_solution = [0]*num_crews
    current_cost = cost_calculation(current_solution)
    # check_coverage 
    current_coverage = check_coverage(current_solution)
    best_cost,best_coverage = 0, [0]*num_total_flights
    best_cost_history = []
    current_cost_history = []
    Temperature=1500.0
    min_temp=1
    cooling_rate=0.999
    count=0
    while (Temperature>min_temp):
        neighbour_solution = flip(current_solution, num_crews) 
        # check_coverage
        neighbour_coverage=check_coverage(neighbour_solution)
        neighbour_cost=cost_calculation(neighbour_solution)
        if neighbour_coverage == 1.0: # If Feasible solution
            if neighbour_cost < best_cost or np.exp((current_cost - neighbour_cost) / Temperature) > random.random():
                best_solution = neighbour_solution.copy()
                best_cost, best_coverage = neighbour_cost, neighbour_coverage
        # print("best_cost, new_cost, temperature ",best_cost, neighbour_cost, Temperature )
        # Acceptance probability
        if abs(neighbour_coverage - 1) < abs(current_coverage - 1) or np.exp((current_cost - neighbour_cost) / Temperature) > random.random():
            current_cost, current_coverage = neighbour_cost, neighbour_coverage
            current_solution = neighbour_solution.copy()
        Temperature *= cooling_rate  # cooling
        best_cost_history.append(best_cost)
        current_cost_history.append(current_cost)
        count+=1
    best_cover_count = row_sum_calculate(constraint_matrix_calculate(best_solution))
    best_solution=[index+1 for index, value in enumerate(best_solution) if value == 1]    
    current_cover_count = row_sum_calculate(constraint_matrix_calculate(current_solution))
    current_solution=[index+1 for index, value in enumerate(current_solution) if value == 1]
        # After the algorithm ends, plot the best cost history
    plt.figure(figsize=(12, 6))
    # Plot for best cost history
    plt.subplot(1, 2, 1)
    plt.plot(best_cost_history, label='Best Cost')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Best Cost Change Over Iterations')
    plt.legend()

    # Plot for current cost history
    plt.subplot(1, 2, 2)
    plt.plot(current_cost_history, label='Current Cost', color='orange')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Current Cost Change Over Iterations')
    plt.legend()
    plt.tight_layout()
    if best_cost!=0:
        return best_solution,best_cost,best_cover_count
    else:
        print("No feasible solution found")
        return current_solution,current_cost,current_cover_count

def standard_bga(data):
    print("standard_bga")
    #data loading
    num_total_flights, num_flights, num_crews, costs, flights = data
    # parameters
    population_size = 150
    max_generations = 800
    generation=0
    mutation_rate = 1 / (num_crews)  # Probability of mutating a bit
    tournament_size = 2
    best_fitness_history = []
        # Function to calculate the constraint matrix
    def constraint_matrix_calculate(solution, flights, num_total_flights):
        # Create an empty matrix.
        matrix = np.zeros((num_total_flights, num_crews))
        # Convert solution to a NumPy array
        sol = np.array(solution)
        # For each crew j that is active, update rows based on flights[j]
        active_indices = np.where(sol == 1)[0]
        for j in active_indices:
            for flight in flights[j]:
                if flight < num_total_flights:
                    matrix[flight, j] = 1
        # print("matrix", matrix)
        return matrix
    
    def fitness(best_solution):
        return np.dot(costs, best_solution)
    
    def generate_individual():
        individual = [random.choice([0, 1]) for _ in range(num_crews)]
        # individual = [0 for _ in range(num_crews)]
        return individual
    
    def tournament_selection(population, fitness_values):
        selected_indices = random.sample(range(len(population)), tournament_size)
        selected = [population[i] for i in selected_indices]
        selected_fitness = [fitness_values[i] for i in selected_indices]
        return selected[selected_fitness.index(min(selected_fitness))]
    
    def check_flight_coverage(individual):
        # Calculate the constraint matrix
        constraint_matrix = constraint_matrix_calculate(individual)
        # Calculate the sum of each row in the constraint matrix
        cover_count = [sum(row) for row in constraint_matrix]
        # Check if all flights are covered exactly once
        return all(value == 1 for value in cover_count)
    
    # implement the mutation operator
    def mutate(individual):
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                individual[i] = 1 - individual[i]
        return individual

    # implement the one point crossover operator
    def crossover(parent1, parent2):
        # Choose a random crossover point
        crossover_point = random.randint(0, len(parent1) - 1)
        # Create a new child
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child
    # implement the selection operator

    population = [generate_individual() for _ in range(population_size)]
    fitness_values = [fitness(ind) for ind in population]
    while generation < max_generations:
        print('start generation', generation)
        # Evaluate fitness (cost) of best population
        # Selection (tournament selection)
        new_population = []
        # Generate new population in new generation
        while len(new_population)<population_size:
            parent1 = tournament_selection(population, fitness_values)
            parent2 = tournament_selection(population, fitness_values)
        # Variation (Crossover)
            child = crossover(parent1, parent2)
            child = mutate(child)
            # Reproduction (generate Xt+1)
            new_population.append(child)
        population = new_population
        fitness_values = [fitness(ind) for ind in population]
        best_fitness_history.append(min(fitness_values))
        generation+=1
        
    min_fitness_index = fitness_values.index(min(fitness_values))
    best_individual = population[min_fitness_index]
    # Compute the final constraint matrix for the best individual.
    final_matrix = constraint_matrix_calculate(best_individual, flights, num_total_flights)
    # Calculate row sums to verify flight coverage.
    cover_count = [sum(row) for row in final_matrix]

    fitness_values = [fitness(ind) for ind in population]
    plt.plot(best_fitness_history, label='Best Cost')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Best Cost Change Over Iterations')
    plt.legend()
    return best_individual, min(fitness_values), cover_count

def improved_bga(data):
    print("improved_bga")
    num_total_flights, num_flights, num_crews, costs, flights = data
    best_cost_history = []
    current_cost_history = []
    population_size = 150
    max_generations = 800
    generation=0
    mutation_rate= 1/3
    tournament_size = 3
    
    def fitness(solution):
        base=np.dot(costs, solution)
        matrix = constraint_matrix_calculate(solution, flights, num_total_flights)
        cover_count = [sum(row) for row in matrix]
        penalty = penalty = sum(abs(c - 1) for c in cover_count)
        lambda_penalty = 2000
        return base + lambda_penalty * penalty
    
    def compute_alpha_beta(num_total_flights, num_crews, flights):
        # Compute the alpha and beta sets for each flight and crew
        alpha = [set() for _ in range(num_total_flights)]
        beta = [set() for _ in range(num_crews)]
        for i, flight_list in enumerate(flights):
            for flight in flight_list:
                alpha[flight].add(i)
                beta[i].add(flight)
        return alpha, beta
    
    population_hashes = set()
    
    def individual_hash(individual):
    # Convert list to tuple to create a hashable representation.
        return tuple(individual)

    # Function to calculate the constraint matrix
    def constraint_matrix_calculate(solution, flights, num_total_flights):
        # Create an empty matrix.
        matrix = np.zeros((num_total_flights, num_crews))
        # Convert solution to a NumPy array
        sol = np.array(solution)
        # For each crew j that is active, update rows based on flights[j]
        active_indices = np.where(sol == 1)[0]
        for j in active_indices:
            for flight in flights[j]:
                if flight < num_total_flights:
                    matrix[flight, j] = 1
        # print("matrix", matrix)
        return matrix
    
    def constraint_matrix_calculate_vec(solution, flights, num_total_flights):
        sol = np.array(solution)
        matrix = np.zeros((num_total_flights, len(sol)))
        active_indices = np.where(sol == 1)[0]
        for j in active_indices:
            # Use np.add.at to update the matrix rows for all flights in one go:
            np.add.at(matrix, flights[j], 1)
        return matrix
    
    alpha, beta = compute_alpha_beta(num_total_flights, num_crews, flights)
    
    # Implement the pseudo-random initialisation method (Algorithm 2, p341 in [1])
    def initialize_population(N, I, alpha, beta):
        population = []
        for k in range(N):
            S = set()         # S: set of selected columns (crew indices)
            U = set(I)        # U: set of uncovered rows (flights)
            while U:
                i = random.choice(list(U))  # randomly select a row i from U
                # Already covered rows: I - U
                covered = set(I) - U
                # available columns: those in alpha[i] whose beta does not intersect with covered rows
                available = [j for j in alpha[i] if beta[j].intersection(covered) == set()]
                if available:
                    j = random.choice(available)
                    S.add(j)
                    # Remove all rows covered by column j from U
                    U -= beta[j]
                else:
                    U.remove(i)
            # Convert the set S into a binary vector of length num_crews.
            individual = [1 if j in S else 0 for j in range(num_crews)]
            population.append(individual)
        return population
    
    def matching_selection(parent1):
        best_match = None
        best_match_score = float('-inf')
        for candidate in current_population:
            if candidate == parent1:
                continue  # Don't select the same individual
            
            # Calculate the "match score" (how well candidate complements parent1)
            combined_coverage = np.logical_or(parent1, candidate)  # Bitwise OR to see coverage
            overcovered_flights = sum((parent1[i] + candidate[i]) > 1 for i in range(len(parent1)))
            match_score = sum(combined_coverage) - overcovered_flights  # Favor better coverage, penalize overcoverage

            if match_score > best_match_score:
                best_match_score = match_score
                best_match = candidate

        return best_match

    def uniform_crossover(parent1, parent2):
        child = [random.choice([parent1[i], parent2[i]]) for i in range(len(parent1))]
        return child
    
    def static_mutate(individual,mutation_rate=mutation_rate):
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                individual[i] = 1 - individual[i]
        return individual
        
    def adaptive_mutate(individual, epsilon=0.5, Ma=5):
        N=len(current_population)
        eta = np.zeros(num_total_flights)
        # Step 1: Compute constraint violations (ηi) for each row across the population
        for individual in current_population:
            cover_count = [sum(row) for row in constraint_matrix_calculate(individual, flights, num_total_flights)]  # Row sums
            for i in range(num_total_flights):
                if cover_count[i] != 1:  # Constraint i is violated (under- or over-covered)
                    eta[i] += 1  # Count the violation
        # Step 2: Apply adaptive mutation if ηi ≥ εN 
        for i in range(num_total_flights):
            if eta[i] >= epsilon * N:  # If constraint i is violated in too many individuals
                available_columns = list(alpha[i])  # Columns that cover row i
                if available_columns:
                    selected_columns = random.sample(available_columns, min(Ma, len(available_columns)))
                    for j in selected_columns:
                        child[j] = 1  # Force activate these columns in the child solution
        return child
    
    def heuristic_improvement(individual, flights, num_total_flights, alpha, beta, costs):
        """
        Heuristic improvement operator for the SPP.
        Tries to improve the solution by removing unnecessary columns (DROP) and adding missing ones (ADD).
        """

        # Compute the constraint matrix for the current solution
        constraint_matrix = np.array(constraint_matrix_calculate(individual, flights, num_total_flights))

        # Step 1: Compute initial row coverage: w[i] is the number of columns covering row i
        w = np.sum(constraint_matrix, axis=1)  # Row sums

        # STEP 2: DROP PROCEDURE (Randomly remove redundant columns)
        active_columns = {j for j, val in enumerate(individual) if val == 1}
        T = list(active_columns)  # Copy of S for random inspection
        random.shuffle(T)

        for j in T:
            # If removing column j keeps all rows covered at most once, remove it
            if any(w[i] >= 2 for i in beta[j]):
                individual[j] = 0  # Remove column j
                for i in beta[j]:
                    w[i] -= 1  # Update row coverage for rows covered by column j

        # STEP 3: IDENTIFY UNDER-COVERED ROWS
        U = {i for i in range(num_total_flights) if w[i] == 0}  # Rows not covered

        # STEP 4: ADD PROCEDURE (Relaxed Condition)
        V = list(U)  # Uncovered rows
        random.shuffle(V)  # Randomize row order

        while V:
            i = V.pop()  # Get an uncovered row
            candidates = []

            for j in alpha[i]:
                if individual[j] == 0:  # Consider only columns that are not already in the solution
                    ratio = costs[j] / len(beta[j])
                    # Compute overcoverage penalty: count rows in beta[j] that are already covered
                    over_penalty = sum(1 for row in beta[j] if w[row] >= 1)
                    candidates.append((j, ratio, over_penalty))

            if candidates:
                # First, try to select candidate(s) with zero overcoverage (i.e., strictly covers only uncovered rows)
                no_over_candidates = [cand for cand in candidates if cand[2] == 0]
                if no_over_candidates:
                    best_column = min(no_over_candidates, key=lambda x: x[1])[0]
                else:
                    # Otherwise, select the candidate with the smallest overcoverage penalty and best cost ratio
                    best_column = min(candidates, key=lambda x: (x[2], x[1]))[0]

                # Add the selected column to the solution
                individual[best_column] = 1

                # Update coverage for all rows covered by best_column
                for row in beta[best_column]:
                    w[row] += 1
                    # Remove row from U if it becomes covered (w[row] >= 1)
                    if row in U:
                        U.remove(row)
                        if row in V:
                            V.remove(row)

        return individual

    def unfitness(individual):
        constraint_matrix = constraint_matrix_calculate(individual,flights,num_total_flights)
        cover_count = [sum(row) for row in constraint_matrix]
        # For each row i that is not exactly covered (i.e., count != 1), add |count - 1|
        return sum(abs(count - 1) for count in cover_count if count != 1)

    def stochastic_ranking(population, num_sweeps,P=0.45):
        # divide  solution into feasible and infeasible
        N=len(population)
        # print("population", population)
        # print("n", N)
        for sweep in range(num_sweeps):
            changed = False
            for i in range(N - 1):
                # Compute unfitness and cost for the adjacent pair.
                u1 = unfitness(population[i])
                u2 = unfitness(population[i + 1])
                f1 = fitness(population[i])
                f2 = fitness(population[i + 1])
            # If both individuals are feasible (unfitness 0), compare by cost only.
            if u1 == 0 and u2 == 0:
                if f1 > f2:  # lower cost is better
                    population[i], population[i + 1] = population[i + 1], population[i]
                    changed = True
            else:
                # At least one is infeasible: decide with probability.
                if random.random() < P:
                    # With probability P, compare by cost.
                    if f1 > f2:
                        population[i], population[i + 1] = population[i + 1], population[i]
                        changed = True
                else:
                    # With probability (1-P), compare by unfitness (lower unfitness is better).
                    if u1 > u2:
                        population[i], population[i + 1] = population[i + 1], population[i]
                        changed = True        
                # If no swaps occurred in an entire sweep, we can exit early.
            if not changed:
                break
        return population
        
    I = set(range(num_total_flights))
    current_population = initialize_population(10, I, alpha, beta)
    ranked_population=[]
    combined_population=[]
    while generation < max_generations:
        print('start generation', generation)
        candidates = random.sample(current_population, tournament_size)
        parent1 = min(candidates, key=lambda x: np.dot(x, costs))  # Min cost selection
        parent2 = matching_selection(parent1)
        child = uniform_crossover(parent1, parent2)
        #apply static mutation operator
        child = static_mutate(child)
        #apply adaptive mutation operator
        child = adaptive_mutate(child)
        # heuristic improvement operator to child
        child = heuristic_improvement(child, flights, num_total_flights, alpha, beta, costs)
        # print("child", child)
        # print("current Pop",current_population)
        # When adding a child:
        child_hash = individual_hash(child)
        if child_hash in population_hashes:
            # It's a duplicate—skip it
            continue
        else:
            current_population.append(child)
            population_hashes.add(child_hash)

        ranked_population = stochastic_ranking(current_population,len(current_population))
        current_population = ranked_population[:population_size]
        fitness_values = [fitness(ind) for ind in current_population]
        min_fitness_index = fitness_values.index(min(fitness_values))
        best_individual = current_population[min_fitness_index]
        best_cost_history.append(fitness(best_individual))
        generation+=1

    final_matrix = constraint_matrix_calculate(best_individual, flights, num_total_flights)
    # Calculate row sums to verify flight coverage.
    cover_count = [sum(row) for row in final_matrix]
    plt.plot(best_cost_history, label='Best Cost')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Best Cost Change Over Iterations')
    plt.legend()
    return best_individual, min(fitness_values), cover_count

if __name__ == "__main__":
    main()
    
    