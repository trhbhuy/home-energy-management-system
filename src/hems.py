import numpy as np
import gurobipy as gp
from gurobipy import GRB
import config as cfg

from components.utility_grid import Grid
from components.renewables import PV
from components.energy_storage import ESS
from components.electric_vehicle import EV
from components.hvac_system import HVAC
from components.electric_water_heating import EWH
from components.non_controllable_load import NCLoad
from components.controllable_load import CALoad

class HomeEnergyManagementSystem:
    def __init__(self, objectives):
        """
        Initializes the Home Energy Management System (HEMS) with various parameters.
        """
        # Import settings from config
        self.rtp = cfg.RTP
        self.ghi = cfg.GHI
        self.theta_air_out = cfg.THETA_AIR_OUT
        self.T_num = cfg.T_NUM
        self.T_set = cfg.T_SET
        self.delta_t = cfg.DELTA_T

        # Initialize components like Grid, PV, ESS, etc.
        self._init_components()

        # Set the objectives for optimization
        self.objectives = objectives

    def _init_components(self):
        """
        Initializes components of the HEMS such as the grid, PV, etc.
        """
        # Power exchange limits (grid), PV, non-controllable and controllable loads, ESS, EV, HVAC, EWH
        self.grid = Grid(cfg.T_NUM, cfg.T_SET, cfg.DELTA_T, cfg.P_GRID_PUR_MAX, cfg.P_GRID_EXP_MAX, cfg.PHI_RTP)
        self.pv = PV(cfg.T_NUM, cfg.T_SET, cfg.DELTA_T, cfg.P_PV_RATE, cfg.N_PV)
        self.ncload = NCLoad(cfg.T_NUM, cfg.T_SET, cfg.DELTA_T, cfg.NUM_NC, cfg.P_NC_RATE, cfg.NUM_NC_OPERATION, cfg.T_NC_START)
        self.caload = CALoad(cfg.T_NUM, cfg.T_SET, cfg.DELTA_T, cfg.NUM_CA, cfg.P_CA_RATE, cfg.NUM_CA_OPERATION, cfg.T_CA_START_MAX, cfg.T_CA_END_MAX, cfg.T_CA_START_PREFER, cfg.T_CA_END_PREFER)
        self.ess = ESS(cfg.T_NUM, cfg.T_SET, cfg.DELTA_T, cfg.P_ESS_CH_MAX, cfg.P_ESS_DCH_MAX, cfg.N_ESS_CH, cfg.N_ESS_DCH, cfg.SOC_ESS_MAX, cfg.SOC_ESS_MIN, cfg.SOC_ESS_SETPOINT)
        self.ev = EV(cfg.T_NUM, cfg.T_SET, cfg.DELTA_T, cfg.P_EV_CH_MAX, cfg.P_EV_DCH_MAX, cfg.N_EV_CH, cfg.N_EV_DCH, cfg.SOC_EV_MAX, cfg.SOC_EV_MIN, cfg.SOC_EV_SETPOINT)
        self.hvac = HVAC(cfg.T_NUM, cfg.T_SET, cfg.DELTA_T, cfg.P_HVAC_MAX, cfg.COP_HVAC, cfg.R_BUILD, cfg.C_AIR_IN, cfg.M_AIR_IN, cfg.COE1_HVAC, cfg.COE2_HVAC, cfg.THETA_AIR_OUT, cfg.THETA_AIR_IN_SETPOINT, cfg.THETA_AIR_IN_MAX_OFFSET, cfg.THETA_AIR_IN_MIN_OFFSET)
        self.ewh = EWH(cfg.T_NUM, cfg.T_SET, cfg.DELTA_T, cfg.P_EWH_MAX, cfg.C_W, cfg.R_W, cfg.V_EWH_MAX, cfg.N_EWH, cfg.COE_EWH, cfg.THETA_EWH_SETPOINT, cfg.THETA_EWH_MAX, cfg.THETA_EWH_MIN, cfg.THETA_COLD_WATER, cfg.V_EWH_DEMAND)

    def init_optim_model(self):
        """
        Initializes the Gurobi optimization model.
        """
        # Create a new model
        model = gp.Model('HEMS')
        model.ModelSense = GRB.MINIMIZE
        model.Params.LogToConsole = 0

        ## Grid modeling
        p_grid_pur, p_grid_exp, u_grid_pur, u_grid_exp = self.grid.add_variables(model)
        self.grid.add_constraints(model, p_grid_pur, p_grid_exp, u_grid_pur, u_grid_exp)

        ## RES modeling
        p_pv = self.pv.get_pv_output(self.ghi, self.theta_air_out)

        ## NCAs modeling
        p_nc = self.ncload.get_power_consumption()

        ## CAs modeling
        u_ca, on_ca, off_ca, t_ca_start, p_ca = self.caload.add_variables(model)
        self.caload.add_constraints(model, u_ca, on_ca, off_ca, t_ca_start, p_ca)
        
        ## ESS modeling
        p_ess_ch, p_ess_dch, u_ess_ch, u_ess_dch, soc_ess, _, _ = self.ess.add_variables(model)
        self.ess.add_constraints(model, p_ess_ch, p_ess_dch, u_ess_ch, u_ess_dch, soc_ess)

        ## EV modeling
        p_ev_ch, p_ev_dch, u_ev_ch, u_ev_dch, soc_ev = self.ev.add_variables(model)    
        t_ev_arrive, t_ev_depart, ev_time_range, soc_ev_init = self.ev.get_ev_availablity(cfg.T_EV_ARRIVE, cfg.T_EV_DEPART, cfg.SOC_EV_INIT)
        self.ev.add_constraints(model, p_ev_ch, p_ev_dch, u_ev_ch, u_ev_dch, soc_ev, t_ev_arrive, t_ev_depart, ev_time_range, soc_ev_init)

        ## HVAC modeling
        p_hvac_h, p_hvac_c, u_hvac_h, u_hvac_c, theta_air_in = self.hvac.add_variables(model)
        self.hvac.add_constraints(model, p_hvac_h, p_hvac_c, u_hvac_h, u_hvac_c, theta_air_in)

        ## EWH modeling
        p_ewh, theta_ewh = self.ewh.add_variables(model)
        self.ewh.add_constraints(model, p_ewh, theta_ewh, theta_air_in)

        ## Energy balance
        for t in self.T_set:
            model.addConstr((p_grid_pur[t] + p_pv[t] + p_ess_dch[t] + p_ev_dch[t]) == (p_grid_exp[t] + p_nc[t] + p_ca[t] + p_ess_ch[t] + p_ev_ch[t] + p_hvac_h[t] + p_hvac_c[t] + p_ewh[t]))
        
        # Define the objective function
        z = model.addVars(self.objectives, vtype=GRB.CONTINUOUS, name="obj_value")

        # Objective function direction
        dir = {'energy_cost': -1, 'PAR': -1, 'discomfort_index': -1}

        # Energy cost objective
        model.addConstr(z['energy_cost'] == self.grid.get_cost(self.rtp, p_grid_pur, p_grid_exp))

        # Peak-to-Average (PAR) objective
        p_grid_max = self.grid.get_max_power(model, p_grid_pur, p_grid_exp)       
        p_grid_avg = (p_grid_pur.sum() - p_grid_exp.sum()) / self.T_num
        model.addConstr(z['PAR'] == self.delta_t * (p_grid_max - p_grid_avg))

        # Discomfort Index objective
        model.addConstr(z['discomfort_index'] == self.caload.get_discomfort_index(model, t_ca_start))

        return model, z
    
    def _solve_subproblem(self, model, z, objectives):
        """
        Solve the subproblems for lexicographic optimization.
        """
        payoff_table = {}

        for obj in objectives:
            # Set the objective direction
            model.setObjective(z[obj], GRB.MINIMIZE)

            # Optimize the model
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                payoff_table[obj] = {o: z[o].x for o in objectives}
                model.addConstr(z[obj] == z[obj].x, name=f"fix_{obj}")  # Freeze this objective for the next round
            else:
                print(f"Optimization failed for {obj}")
                payoff_table[obj] = {o: None for o in objectives}

        return payoff_table

    def generate_payoff_table(self):
        """
        Generate the full lexicographic payoff table for all objective combinations.
        """
        lexi_payoff_table = {}
        objectives_list = [
            ['energy_cost', 'PAR', 'discomfort_index'],  # Lexicographic order 1
            ['PAR', 'energy_cost', 'discomfort_index'],  # Lexicographic order 2
            ['discomfort_index', 'energy_cost', 'PAR']   # Lexicographic order 3
        ]

        for i, objectives in enumerate(objectives_list):
            # Initialize the optimization model for each subproblem
            model, z = self.init_optim_model()

            # Solve the subproblem for the current lexicographic order
            payoff_table = self._solve_subproblem(model, z, objectives)

            # Assign the results based on lexicographic order
            if i == 0:
                lexi_payoff_table['energy_cost'] = payoff_table['discomfort_index']
            elif i == 1:
                lexi_payoff_table['PAR'] = payoff_table['discomfort_index']
            elif i == 2:
                lexi_payoff_table['discomfort_index'] = payoff_table['PAR']

        return lexi_payoff_table

    def epsilon_constraint_method(self, payoff_table, grid_points=6):
        """
        Implements the ε-constraint method to obtain the Pareto front.
        """
        solutions = []

        # Calculate min and max values for each objective from the payoff table
        min_obj = {obj: min(payoff_table[o][obj] for o in self.objectives) for obj in self.objectives}
        max_obj = {obj: max(payoff_table[o][obj] for o in self.objectives) for obj in self.objectives}

        # ε-Constraint loop through grid points (iterate over grid for two objectives)
        for ii in range(grid_points + 1):
            for jj in range(grid_points + 1):
                # Initialize the optimization model
                model, z = self.init_optim_model()

                # Add slack variables to model
                slack = model.addVars(self.objectives, lb=0, name="slack")

                # Calculate the RHS for each ε-constraint
                rhs = {
                    self.objectives[1]: max_obj[self.objectives[1]] - ii / grid_points * (max_obj[self.objectives[1]] - min_obj[self.objectives[1]]),
                    self.objectives[2]: max_obj[self.objectives[2]] - jj / grid_points * (max_obj[self.objectives[2]] - min_obj[self.objectives[2]])
                    }  # Except first obj

                # Apply ε-constraints to remaining objectives
                for obj in self.objectives[1:]:
                    model.addConstr(z[obj] + slack[obj] == rhs[obj], name=f"eps_{obj}_{ii}_{jj}")

                # Augmented objective to avoid weakly Pareto solutions
                model.setObjective(z[self.objectives[0]] - 1e-3 * gp.quicksum(slack[obj] / (max_obj[obj] - min_obj[obj]) for obj in self.objectives[1:]), GRB.MINIMIZE)

                # Solve the model
                model.optimize()

                # Collect solutions
                if model.status == GRB.OPTIMAL:
                    solutions.append([z[o].x for o in self.objectives])

        return np.array(solutions)
    

    def get_results_by_name(self, model):
        """
        Retrieves the optimized values of variables based on their names.
        """
        var_names = ['p_grid_pur', 'p_grid_exp', 'u_grid_pur', 'u_grid_exp',
                     'p_ess_ch', 'p_ess_dch', 'u_ess_ch', 'u_ess_dch', 'soc_ess',
                     'p_ev_ch', 'p_ev_dch', 'u_ev_ch', 'u_ev_dch', 'soc_ev']

        results = {}

        for var_name in var_names:
            results[var_name] = [model.getVarByName(f"{var_name}[{t}]").x for t in self.T_set]

        for obj in self.objectives:
            results[obj] = model.getVarByName(f"obj_value[{obj}]").x

        return results