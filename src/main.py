import argparse
import logging
from gurobipy import GRB

from hems import HomeEnergyManagementSystem
from utils.decision_making import get_best_compromise_solution

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_option():
    """
    Parse command-line arguments for the optimization script.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Home Energy Management System Optimization')
    parser.add_argument('--mode', choices=['single', 'multi'], default='multi', help="Choose between 'single' and 'multi' objective optimization")
    parser.add_argument('--obj', choices=['energy_cost', 'PAR', 'discomfort_index'], default='energy_cost', help="Specify the single objective to optimize")
    parser.add_argument('--num_grid_points', type=int, default=6, help='Number of grid points for epsilon-constraint method')

    args = parser.parse_args()

    return args

def single_objective_optimization(hems, obj):
    """
    Perform single-objective optimization.
    """
    logging.info(f"Starting single-objective optimization for {obj}")
    model, z = hems.init_optim_model()

    # Set objective and optimize
    model.setObjective(z[obj])
    model.optimize()

    # Log the optimal solution
    if model.status == GRB.OPTIMAL:
        logging.info(f"Optimal {obj} value: {model.ObjVal}")
        results = hems.get_results_by_name(model)
    else:
        logging.error(f"Optimization failed for objective {obj}")

def multi_objective_optimization(hems, num_grid_points):
    """
    Perform multi-objective optimization using the epsilon-constraint method.
    """
    logging.info("Starting multi-objective optimization using epsilon-constraint method")

    # Generate payoff table
    payoff_table = hems.generate_payoff_table()
    logging.info("Payoff table generated successfully")

    # Compute the Pareto front using the epsilon-constraint method
    solutions = hems.epsilon_constraint_method(payoff_table, num_grid_points)
    logging.info(f"Pareto front obtained with {len(solutions)} solutions")

    # Apply decision-making logic to get the best compromise solution
    best_solution = get_best_compromise_solution(solutions)
    logging.info(f"Best compromise solution found: {best_solution}")

    return best_solution

def main():
    """
    Main function for running the Home Energy Management System (HEMS) optimization.
    """
    logging.info("Initializing the HEMS optimization process")

    # Parse command-line arguments
    args = parse_option()

    # Initialize the Home Energy Management System (HEMS)
    objectives = ['energy_cost', 'PAR', 'discomfort_index']
    hems = HomeEnergyManagementSystem(objectives)

    # Check the mode and perform the appropriate optimization
    if args.mode == 'single':
        single_objective_optimization(hems, args.obj)
    elif args.mode == 'multi':
        multi_objective_optimization(hems, args.num_grid_points)
    else:
        logging.error("Invalid mode selected. Please choose 'single' or 'multi'.")

# python3 src/main.py --mode single --obj energy_cost
# python3 src/main.py --mode multi --num_grid_points 6
if __name__ == "__main__":
    main()