import argparse
import logging
from hems import HomeEnergyManagementSystem
from decision_making import get_best_compromise_solution

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_option():
    """
    Parse command-line arguments for the training script.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Home Energy Management System Optimization')
    parser.add_argument('--mode', choices=['single', 'multi'], default='multi', help="Choose between 'single' and 'multi' objective optimization")
    parser.add_argument('--objective', choices=['energy_cost', 'PAR', 'DI'], default='energy_cost', help="Specify the single objective to optimize")
    parser.add_argument('--num_grid_points', type=int, default=7, help='Number of grid points for epsilon-constraint method')
    args = parser.parse_args()

    return args

def single_objective_optimization(hems, objective):
    """
    Perform single-objective optimization.

    Args:
        objective (str): The objective to optimize ('energy_cost', 'PAR', or 'DI').
        hems (HomeEnergyManagementSystem): The HEMS instance.

    Returns:
        dict: The results of the optimization.
    """
    objective_mapping = {
        'energy_cost': 1,
        'PAR': 4,
        'DI': 7
    }

    ObjFunc = objective_mapping.get(objective)
    if ObjFunc is None:
        raise ValueError(f"Invalid objective function: {objective}")

    # Solve the single-objective optimization problem
    results = hems.optim(ObjFunc)
    
    # Log the optimal solution
    logging.info(f"Optimal solution: {results['ObjVal']}")
    
    if results:
        logging.info(f"Optimal solution found with objective value: {results['ObjVal']}")
    else:
        logging.warning("No optimal solution was found.")

    return results

def multi_objective_optimization(hems, num_grid_points):
    """
    Perform multi-objective optimization using the epsilon-constraint method.

    Args:
        num_grid_points (int): Number of grid points for epsilon-constraint method.
        hems (HomeEnergyManagementSystem): The HEMS instance.

    Returns:
        np.ndarray: The best compromise solution from the Pareto Front.
    """
    logging.info("Multi-objective optimization using the epsilon-constraint method")

    # Generate payoff tables
    original_payoff_table, lexicographic_payoff_table = hems.generate_payoff_tables()

    # Compute the Pareto Front using the epsilon-constraint method
    pareto_front = hems.pareto_optimization(lexicographic_payoff_table, num_grid_points)

    # Apply fuzzy decision-making to find the best compromise solution on the Pareto Front
    best_compromise_solution = get_best_compromise_solution(pareto_front)

    logging.info("Optimization process completed. Best Compromise Solution:")
    logging.info(best_compromise_solution)

    return best_compromise_solution

def main():
    """
    Main function for running the Home Energy Management System (HEMS) optimization.
    """
    logging.info("Starting HEMS optimization process")

    args = parse_option()

    # Initialize the Home Energy Management System (HEMS)
    hems = HomeEnergyManagementSystem()

    if args.mode == 'single':
        # Perform single-objective optimization
        single_objective_optimization(hems, args.objective)
    elif args.mode == 'multi':
        # Perform multi-objective optimization
        multi_objective_optimization(hems, args.num_grid_points)
    else:
        logging.error("Invalid mode selected. Please choose 'single' or 'multi'.")

# python3 src/main.py --mode single --objective energy_cost
# python3 src/main.py --mode multi --num_grid_points 7
if __name__ == "__main__":
    main()