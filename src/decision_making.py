# src/get_best_compromise_solution.py

import numpy as np

def get_best_compromise_solution(pareto_solutions):
    """
    Selects the best compromise solution from a set of Pareto-optimal solutions using fuzzy logic.

    This function applies a fuzzy logic approach to evaluate each solution in the Pareto set by 
    calculating membership degrees for each objective. It then aggregates these degrees to identify 
    the solution that offers the best trade-off among all objectives.

    Args:
        pareto_solutions (numpy.ndarray): A 2D array where each row represents a solution in the 
                                          Pareto set, and each column represents an objective.

    Returns:
        numpy.ndarray: The best compromise solution from the Pareto set.
    """
    # Number of solutions and objectives
    num_solutions, num_objectives = pareto_solutions.shape

    # Initialize the membership degree matrix
    membership_degrees = np.zeros((num_solutions, num_objectives))

    # Calculate membership degrees for each objective
    for objective_index in range(num_objectives):
        max_value = np.max(pareto_solutions[:, objective_index])
        min_value = np.min(pareto_solutions[:, objective_index])
        
        # Calculate membership degree using linear normalization
        membership_degrees[:, objective_index] = (max_value - pareto_solutions[:, objective_index]) / (max_value - min_value)

    # Sum membership degrees across objectives for each solution
    total_membership_degrees = np.sum(membership_degrees, axis=1)

    # Calculate the weighted membership degree for each solution
    weighted_membership = total_membership_degrees / np.sum(total_membership_degrees)

    # Find the index of the maximum weighted membership degree
    best_solution_index = np.argmax(weighted_membership)

    # Best compromise solution
    best_compromise_solution = pareto_solutions[best_solution_index, :]

    return best_compromise_solution