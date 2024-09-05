import numpy as np
import gurobipy as gp
from gurobipy import GRB
import config as cfg
from util import get_pv_output, get_nc_consumption, get_ca_availability, get_ev_availability

from components.utility_grid import Grid
from components.renewables import PV
from components.energy_storage import ESS
from components.electric_vehicle import EV
from components.hvac_system import HVAC
from components.electric_water_heating import EWH

from components.non_controllable_load import NCLoad

class HomeEnergyManagementSystem:
    def __init__(self):
        """
        Initializes the Home Energy Management System (HEMS) with various parameters
        related to power exchange, pricing, PV generation, appliance usage, battery storage,
        electric vehicles, HVAC, and electric water heater operations.

        Args:
            rtp (array): Real-time pricing data for electricity.
            ghi (array): Global Horizontal Irradiance data for PV calculations.
            theta_air_out (array): Outdoor air temperature data for HVAC calculations.
            v_ewh_demand (array): Hot water demand profile for the electric water heater.
        """
        # Import settings from config
        # Real-time pricing and weather data
        self.rtp = cfg.RTP
        self.ghi = cfg.GHI
        self.theta_air_out = cfg.THETA_AIR_OUT

        # Time settings
        self.T_num = cfg.T_NUM
        self.T_set = cfg.T_SET
        self.delta_t = cfg.DELTA_T

        self.T_set_24 = cfg.T_SET_24

        # Power exchange limits
        self.grid = Grid(cfg.T_NUM, cfg.T_SET, cfg.DELTA_T, cfg.P_GRID_PUR_MAX, cfg.P_GRID_EXP_MAX, cfg.PHI_RTP)

        # PV system parameters
        self.p_pv_rate = cfg.P_PV_RATE
        self.n_pv = cfg.N_PV
        self.phi_pv = cfg.PHI_PV

        self.p_pv = get_pv_output(self.ghi, self.p_pv_rate, self.n_pv, self.theta_air_out)

        # Non-controllable appliances
        self.ncload = NCLoad(cfg.T_NUM, cfg.T_SET, cfg.DELTA_T, cfg.NUM_NC, cfg.P_NC_RATE, cfg.NUM_NC_OPERATION, cfg.T_NC_START)
        self.p_nc = self.ncload.get_power_consumption()

        # Controllable appliances
        self.num_ca = cfg.NUM_CA
        self.p_ca_rate = np.array(cfg.P_CA_RATE)
        self.num_ca_operation = np.array(cfg.NUM_CA_OPERATION)
        self.t_ca_start_max = np.array(cfg.T_CA_START_MAX)
        self.t_ca_end_max = np.array(cfg.T_CA_END_MAX)
        self.t_ca_start_prefer = np.array(cfg.T_CA_START_PREFER)
        self.t_ca_end_prefer = np.array(cfg.T_CA_END_PREFER)
        self.t_ca_range = get_ca_availability(self.T_num, self.num_ca, self.t_ca_start_max, self.t_ca_end_max)

        # Battery Energy Storage System (ess)
        self.ess = ESS(cfg.T_NUM, cfg.T_SET, cfg.DELTA_T, cfg.P_ESS_CH_MAX, cfg.P_ESS_DCH_MAX, cfg.N_ESS_CH, cfg.N_ESS_DCH, cfg.SOC_ESS_MAX, cfg.SOC_ESS_MIN, cfg.SOC_ESS_SETPOINT)

        # Plug-in Hybrid Electric Vehicles (ev)
        self.ev = EV(cfg.T_NUM, cfg.T_SET, cfg.DELTA_T, cfg.P_EV_CH_MAX, cfg.P_EV_DCH_MAX, cfg.N_EV_CH, cfg.N_EV_DCH, cfg.SOC_EV_MAX, cfg.SOC_EV_MIN, cfg.SOC_EV_SETPOINT)

        # Heating, Ventilation, and Air Conditioning (HVAC)
        self.hvac = HVAC(cfg.T_NUM, cfg.T_SET, cfg.DELTA_T, cfg.P_HVAC_MAX, cfg.COP_HVAC, cfg.R_BUILD, cfg.C_AIR_IN, cfg.M_AIR_IN, cfg.COE1_HVAC, cfg.COE2_HVAC, cfg.THETA_AIR_OUT, cfg.THETA_AIR_IN_SETPOINT, cfg.THETA_AIR_IN_MAX_OFFSET, cfg.THETA_AIR_IN_MIN_OFFSET)

        # Electric Water Heater (EWH)
        self.ewh = EWH(cfg.T_NUM, cfg.T_SET, cfg.DELTA_T, cfg.P_EWH_MAX, cfg.C_W, cfg.R_W, cfg.V_EWH_MAX, cfg.N_EWH, cfg.COE_EWH, cfg.THETA_EWH_SETPOINT, cfg.THETA_EWH_MAX, cfg.THETA_EWH_MIN, cfg.THETA_COLD_WATER, cfg.V_EWH_DEMAND)
            
    def optim(self, ObjFunc, eps1=0, eps2=0, eps3=0, r2=0, r3=0):
        """
        Optimizes the Home Energy Management System based on the specified objective function.

        Args:
            ObjFunc (int): Indicator of the objective function to be optimized.
                        (1: Energy Cost, 2: Peak-to-Average Ratio, 3: Discomfort Index)
            eps1 (float): Constraint value for energy cost in sub-problem optimizations.
            eps2 (float): Constraint value for Peak-to-Average Ratio in sub-problem optimizations.
            eps3 (float): Constraint value for Discomfort Index in sub-problem optimizations.
            r2 (float): Weighting factor for Peak-to-Average Ratio in multi-objective optimization.
            r3 (float): Weighting factor for Discomfort Index in multi-objective optimization.

        Returns:
            dict: A dictionary containing the optimized values of various decision variables 
                and performance metrics if the optimization is successful.
            None: If no optimal solution is found.
        """
        # Create a new model
        model = gp.Model('HEMS')
        model.ModelSense = GRB.MINIMIZE
        model.Params.LogToConsole = 0

        ## Grid modeling
        p_grid_pur, p_grid_exp, u_grid_pur, u_grid_exp = self.grid.add_variables(model)
        self.grid.add_constraints(model, p_grid_pur, p_grid_exp, u_grid_pur, u_grid_exp)

        ## Controllable appliances (CAs) modeling
        # Variables for Solution
        u_ca = model.addMVar((self.T_num,self.num_ca), vtype=GRB.BINARY, name="u_ca")
        on_ca = model.addMVar((self.T_num,self.num_ca), vtype=GRB.BINARY, name="on_ca")
        off_ca = model.addMVar((self.T_num,self.num_ca), vtype=GRB.BINARY, name="off_ca")
        t_ca_start = model.addMVar(self.num_ca, lb=0, vtype=GRB.INTEGER, name="t_ca_start")
        p_ca = model.addMVar(self.T_num, lb=0, vtype=GRB.CONTINUOUS, name="p_ca")

        # CAs Constraint
        for i in range(self.num_ca):
            # Summation of time operation
            model.addConstr(u_ca[:,i].sum() == self.num_ca_operation[i])

            # Summation of on mode and off mode
            model.addConstr(on_ca[:,i].sum() == 1)
            model.addConstr(off_ca[:, i].sum() == 1)

            # Constraint for range of shiftable devices
            model.addConstr(self.t_ca_range[:,i] - u_ca[:,i] >= 0)

            # Define time start
            model.addConstr(t_ca_start[i] == (on_ca[:,i] * self.T_set_24).sum())

        # Constraints for on and off mode with u mode
        for i in range(self.T_num):
            model.addConstr(p_ca[i] == (self.p_ca_rate * u_ca[i,:]).sum())

            for j in range(self.num_ca):
                if i == 0:
                    model.addConstr(u_ca[0,j] - 0 == on_ca[i,i] - off_ca[i,i])
                else:
                    model.addConstr(u_ca[i,j] - u_ca[i-1,j] == on_ca[i,j] - off_ca[i,j])

        ## Battery Energy Storage System (ESS) modeling
        p_ess_ch, p_ess_dch, u_ess_ch, u_ess_dch, soc_ess, _, _ = self.ess.add_variables(model)
        self.ess.add_constraints(model, p_ess_ch, p_ess_dch, u_ess_ch, u_ess_dch, soc_ess)

        ## Plug-in Electric Vehicle (EV) modeling
        p_ev_ch, p_ev_dch, u_ev_ch, u_ev_dch, soc_ev = self.ev.add_variables(model)
    
        # Define PEV range operation
        t_ev_arrive, t_ev_depart, ev_time_range, soc_ev_initial = self.ev.get_ev_availablity(cfg.T_EV_ARRIVE, cfg.T_EV_DEPART, cfg.SOC_EV_INITIAL)

        self.ev.add_constraints(model, p_ev_ch, p_ev_dch, u_ev_ch, u_ev_dch, soc_ev, t_ev_arrive, t_ev_depart, ev_time_range, soc_ev_initial)

        ## Heating-Ventilation-Air Conditioner (HVAC) modeling
        p_hvac_h, p_hvac_c, u_hvac_h, u_hvac_c, theta_air_in = self.hvac.add_variables(model)

        self.hvac.add_constraints(model, p_hvac_h, p_hvac_c, u_hvac_h, u_hvac_c, theta_air_in)

        ## Electric Water Heater (EWH) modeling
        p_ewh, theta_ewh = self.ewh.add_variables(model)
        self.ewh.add_constraints(model, p_ewh, theta_ewh, theta_air_in)

        ## Energy balance
        for i in range(self.T_num):
            model.addConstr((p_grid_pur[i] + self.p_pv[i] + p_ess_dch[i] + p_ev_dch[i]) == (p_grid_exp[i] + self.p_nc[i] + p_ca[i] + p_ess_ch[i] + p_ev_ch[i] + p_hvac_h[i] + p_hvac_c[i] + p_ewh[i]))

        # Cost exchange with utility grid
        energy_cost = self.grid.get_cost(self.rtp, p_grid_pur, p_grid_exp)

        ## Peak-to-Average (PAR)
        # Variables for Solution
        p_grid_max = model.addVar(vtype=GRB.CONTINUOUS, name="p_grid_max")
        u_grid_max = model.addMVar(self.T_num, vtype=GRB.BINARY, name="u_grid_max")

        p_grid_average = gp.quicksum(((p_grid_pur[i] - p_grid_exp[i]) / self.T_num) for i in range(self.T_num))

        # PAR
        PAR = self.delta_t * (p_grid_max - p_grid_average)

        # PAR Constraints
        model.addConstr(u_grid_max.sum() == 1)
        for i in range(self.T_num):
            model.addConstr(p_grid_max >= p_grid_pur[i] - p_grid_exp[i])
            model.addConstr(p_grid_max <= p_grid_pur[i] - p_grid_exp[i] + (1 - u_grid_max[i]) * 1000)

        ## Discomfort Index (DI)
        # Variables for Solution
        discomfort_index = model.addMVar(self.num_ca, lb=0, vtype=GRB.INTEGER, name = "discomfort_index")
        u_discomfort = model.addMVar(self.T_num, vtype=GRB.BINARY, name="u_discomfort")

        # DI Constraint
        for i in range(self.num_ca):
            model.addConstr(discomfort_index[i] >= self.t_ca_start_prefer[i] - t_ca_start[i])
            model.addConstr(discomfort_index[i] >= t_ca_start[i] - self.t_ca_start_prefer[i])
            model.addConstr(discomfort_index[i] <= self.t_ca_start_prefer[i] - t_ca_start[i] + u_discomfort[i] * 1000)
            model.addConstr(discomfort_index[i] <= t_ca_start[i] - self.t_ca_start_prefer[i] + (1 - u_discomfort[i]) * 1000)

        # Overall Discomfort Index
        overall_discomfort_index = discomfort_index.sum()

        ## Define Objective
        # Sub-problem 1
        if ObjFunc == 1:
            model.setObjective(energy_cost)
        elif ObjFunc == 2:
            # Add constraint for energy cost:
            model.addConstr(energy_cost == eps1)

            model.setObjective(PAR)
        elif ObjFunc == 3:
            # Add constraint for energy cost and PAR:
            model.addConstr(energy_cost == eps1)
            model.addConstr(PAR == eps2)

            model.setObjective(overall_discomfort_index)

        # Sub-problem 2
        elif ObjFunc == 4:
            model.setObjective(PAR)
        elif ObjFunc == 5:
            # Add constraint for PAR:
            model.addConstr(PAR == eps2)

            model.setObjective(energy_cost)
        elif ObjFunc == 6:
            # Add constraint for PAR and DI:
            model.addConstr(energy_cost == eps1)
            model.addConstr(PAR == eps2)

            model.setObjective(overall_discomfort_index)

        # Sub-problem 3
        elif ObjFunc == 7:
            model.setObjective(overall_discomfort_index)
        elif ObjFunc == 8:
            # Add constraint for DI:
            model.addConstr(overall_discomfort_index == eps3)

            model.setObjective(energy_cost)
        elif ObjFunc == 9:
            # Add constraint for PAR and DI:
            model.addConstr(energy_cost == eps1)
            model.addConstr(overall_discomfort_index == eps3)

            model.setObjective(PAR)

        # Multi-objective optimization problem
        elif ObjFunc == 10:
            # Slack variables:
            r1 = 1e-3
            s2 = model.addVar(lb = 0, ub = 1000, vtype=GRB.CONTINUOUS, name="s2")
            s3 = model.addVar(lb = 0, ub = 1000, vtype=GRB.CONTINUOUS, name="s3")

            model.addConstr((PAR + s2) == eps2)
            model.addConstr((overall_discomfort_index + s3) == eps3)

            model.setObjective(energy_cost - r1 * (s2/r2 + s3/r3))

        msgdict = {GRB.OPTIMAL : 'Optimal', GRB.INFEASIBLE : 'Infeasible model'}
        model.optimize()

        # Handle different optimization statuses
        if msgdict[model.status] == "Optimal":
            # print("\n\n model.objValue =================",model.ObjVal)
            # Collect the results into a dictionary
            results = {
                'ObjVal': model.ObjVal,
                'energy_cost': energy_cost.getValue().item(),
                'PAR': PAR.getValue().item(),
                'overall_discomfort_index': overall_discomfort_index.getValue().item(),
                'p_ca': p_ca.X,
                'u_ca': u_ca.X,
                'on_ca': on_ca.X,
                'off_ca': off_ca.X,
                't_ca_start': t_ca_start.X,
                'p_ess_ch': p_ess_ch.X,
                'p_ess_dch': p_ess_dch.X,
                'soc_ess': soc_ess.X,
                'p_ev_ch': p_ev_ch.X,
                'p_ev_dch': p_ev_dch.X,
                'soc_ev': soc_ev.X,
                'p_hvac_h': p_hvac_h.X,
                'p_hvac_c': p_hvac_c.X,
                'theta_air_in': theta_air_in.X,
                'p_ewh': p_ewh.X,
                'theta_ewh': theta_ewh.X,
                'p_grid_pur': p_grid_pur.X,
                'p_grid_exp': p_grid_exp.X,
                'p_grid_max': p_grid_max.X,
                'u_grid_max': u_grid_max.X,
                'discomfort_index': discomfort_index.X,
                'u_discomfort': u_discomfort.X
            }
            return results
        else:
            # print("No optimal solution found.")
            return None

    def generate_payoff_tables(self):
        """
        Solves the single-objective sub-problems and constructs both the original and 
        lexicographic payoff tables. These tables capture the trade-offs between 
        energy cost, Peak-to-Average Ratio (PAR), and Discomfort Index (DI).

        Returns:
            tuple:
                - original_payoff_table (np.ndarray): Results of optimizing each objective individually.
                - lexicographic_payoff_table (np.ndarray): Results of a lexicographic optimization approach.
        """
        # Solve Single-objective sub-problem 1 - Minimize energy cost
        results_1 = self.optim(ObjFunc=1)
        results_2 = self.optim(ObjFunc=2, eps1=results_1['energy_cost'], eps2=results_1['PAR'], eps3=results_1['overall_discomfort_index'])
        results_3 = self.optim(ObjFunc=3, eps1=results_2['energy_cost'], eps2=results_2['PAR'], eps3=results_2['overall_discomfort_index'])
        sub_problem_1 = np.array([results_3['energy_cost'], results_3['PAR'], results_3['overall_discomfort_index']])

        # Solve Single-objective sub-problem 2 - Minimize PAR
        results_4 = self.optim(ObjFunc=4)
        results_5 = self.optim(ObjFunc=5, eps1=results_4['energy_cost'], eps2=results_4['PAR'], eps3=results_4['overall_discomfort_index'])
        results_6 = self.optim(ObjFunc=6, eps1=results_5['energy_cost'], eps2=results_5['PAR'], eps3=results_5['overall_discomfort_index'])
        sub_problem_2 = np.array([results_6['energy_cost'], results_6['PAR'], results_6['overall_discomfort_index']])

        # Solve Single-objective sub-problem 3 - Minimize discomfort Index
        results_7 = self.optim(ObjFunc=7)
        results_8 = self.optim(ObjFunc=8, eps1=results_7['energy_cost'], eps2=results_7['PAR'], eps3=results_7['overall_discomfort_index'])
        results_9 = self.optim(ObjFunc=9, eps1=results_8['energy_cost'], eps2=results_8['PAR'], eps3=results_8['overall_discomfort_index'])
        sub_problem_3 = np.array([results_9['energy_cost'], results_9['PAR'], results_9['overall_discomfort_index']])

        # Construct the original payoff table
        original_payoff_table = np.array([
            [results_1['energy_cost'], results_1['PAR'], results_1['overall_discomfort_index']],
            [results_4['energy_cost'], results_4['PAR'], results_4['overall_discomfort_index']],
            [results_7['energy_cost'], results_7['PAR'], results_7['overall_discomfort_index']]
        ])

        # Construct the lexicographic payoff table
        lexicographic_payoff_table = np.vstack((sub_problem_1, sub_problem_2, sub_problem_3))

        return original_payoff_table, lexicographic_payoff_table
    
    def pareto_optimization(self, payoff_table, num_grid_points=7):
        """
        Computes the Pareto Front using the epsilon-constraint method for the HEMS.

        Parameters:
            payoff_table (np.ndarray): The payoff table.
            num_grid_points (int): Number of grid points for epsilon values.

        Returns:
            tuple:
                - Pareto_Front (np.ndarray): Solutions on the Pareto Front.
                - epsilon_idx (np.ndarray): Epsilon values used to generate each point.
        """
        # Extract the range for the constraints from the payoff table
        max_PAR, min_PAR = np.max(payoff_table[:, 1]), np.min(payoff_table[:, 1])
        max_DI, min_DI = np.max(payoff_table[:, 2]), np.min(payoff_table[:, 2])
            
        # Generate epsilon values for each objective based on grid points
        epsilon_PAR = np.linspace(max_PAR, min_PAR, num_grid_points)
        epsilon_DI = np.linspace(max_DI, min_DI, num_grid_points)

        # Calculate the range for each objective
        range_PAR = max_PAR - min_PAR
        range_DI = max_DI - min_DI

        pareto_solutions = []

        # Iterate over all epsilon values
        for eps2 in epsilon_PAR:
            for eps3 in epsilon_DI:
                result = self.optim(ObjFunc=10, eps2=eps2, eps3=eps3, r2=range_PAR, r3=range_DI)
                if result:
                    pareto_solutions.append([result['energy_cost'], result['PAR'], result['overall_discomfort_index']])

        # Convert list of Pareto solutions to a NumPy array
        pareto_front = np.array(pareto_solutions)

        return pareto_front