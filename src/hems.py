import numpy as np
import gurobipy as gp
from gurobipy import GRB
import config as cfg
from util import get_pv_output, get_nc_consumption, get_ca_availability, get_ev_availability

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
        self.num_time_steps = cfg.NUM_TIME_STEPS
        self.T_set = np.arange(1, self.num_time_steps + 1)
        self.delta_t = 24 / self.num_time_steps

        # Power exchange limits
        self.p_g2h_max = cfg.P_G2H_MAX
        self.p_h2g_max = cfg.P_H2G_MAX

        # PV system parameters
        self.p_pv_rate = cfg.P_PV_RATE
        self.n_pv = cfg.N_PV
        self.p_pv = get_pv_output(self.ghi, self.p_pv_rate, self.n_pv, self.theta_air_out)

        # Non-controllable appliances
        self.num_nc = cfg.NUM_NC
        self.p_nc_rate = cfg.P_NC_RATE
        self.num_nc_operation = cfg.NUM_NC_OPERATION
        self.t_nc_start = cfg.T_NC_START
        self.p_nc = get_nc_consumption(self.num_time_steps, self.num_nc, self.p_nc_rate, self.num_nc_operation, self.t_nc_start)

        # Controllable appliances
        self.num_ca = cfg.NUM_CA
        self.p_ca_rate = np.array(cfg.P_CA_RATE)
        self.num_ca_operation = np.array(cfg.NUM_CA_OPERATION)
        self.t_ca_start_max = np.array(cfg.T_CA_START_MAX)
        self.t_ca_end_max = np.array(cfg.T_CA_END_MAX)
        self.t_ca_start_prefer = np.array(cfg.T_CA_START_PREFER)
        self.t_ca_end_prefer = np.array(cfg.T_CA_END_PREFER)
        self.t_ca_range = get_ca_availability(self.num_time_steps, self.num_ca, self.t_ca_start_max, self.t_ca_end_max)

        # Battery Energy Storage System (BESS)
        self.soc_ess_max = cfg.SOC_ESS_MAX
        self.bess_dod = cfg.BESS_DOD
        self.soc_ess_min = (1 - self.bess_dod) * self.soc_ess_max
        self.p_ess_ch_max = cfg.P_ESS_CH_MAX
        self.p_ess_dch_max = cfg.P_ESS_DCH_MAX
        self.n_bess = cfg.N_BESS

        # Plug-in Hybrid Electric Vehicles (PEV)
        self.soc_pev_max = cfg.SOC_PEV_MAX
        self.pev_dod = cfg.PEV_DOD
        self.soc_pev_min = (1 - self.pev_dod) * self.soc_pev_max
        self.p_pev_max = cfg.P_PEV_MAX
        self.n_pev = cfg.N_PEV
        self.t_pev_arrive = cfg.T_PEV_ARRIVE
        self.t_pev_depart = cfg.T_PEV_DEPART
        self.soc_pev_initial = cfg.SOC_PEV_INITIAL
        self.t_pev_range, self.num_pev_operation = get_ev_availability(self.num_time_steps, self.t_pev_arrive, self.t_pev_depart)

        # Heating, Ventilation, and Air Conditioning (HVAC)
        self.p_hvac_max = cfg.P_HVAC_MAX
        self.cop_hvac = cfg.COP_HVAC
        self.R_build = cfg.R_BUILD
        self.C_air_in = cfg.C_AIR_IN
        self.m_air_in = cfg.M_AIR_IN
        self.coe1_hvac = cfg.COE1_HVAC
        self.coe2_hvac = cfg.COE2_HVAC
        self.theta_air_in_setpoint = cfg.THETA_AIR_IN_SETPOINT
        self.theta_air_in_max = self.theta_air_in_setpoint + cfg.THETA_AIR_IN_MAX_OFFSET
        self.theta_air_in_min = self.theta_air_in_setpoint - cfg.THETA_AIR_IN_MIN_OFFSET

        # Electric Water Heater (EWH)
        self.p_ewh_max = cfg.P_EWH_MAX
        self.C_w = cfg.C_W
        self.R_w = cfg.R_W
        self.v_ewh_max = cfg.V_EWH_MAX
        self.n_ewh = cfg.N_EWH
        self.coe_ewh = np.exp(-self.delta_t / (self.R_w * self.C_w))
        self.theta_ewh_setpoint = cfg.THETA_EWH_SETPOINT
        self.theta_ewh_max = cfg.THETA_EWH_MAX
        self.theta_ewh_min = cfg.THETA_EWH_MIN
        self.theta_cold_water = cfg.THETA_COLD_WATER
        self.v_ewh_demand = cfg.V_EWH_DEMAND
    
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

        ## Controllable appliances (CAs) modeling
        # Variables for Solution
        u_ca = model.addMVar((self.num_time_steps,self.num_ca), vtype=GRB.BINARY, name="u_ca")
        on_ca = model.addMVar((self.num_time_steps,self.num_ca), vtype=GRB.BINARY, name="on_ca")
        off_ca = model.addMVar((self.num_time_steps,self.num_ca), vtype=GRB.BINARY, name="off_ca")
        t_ca_start = model.addMVar(self.num_ca, lb=0, vtype=GRB.INTEGER, name="t_ca_start")
        p_ca = model.addMVar(self.num_time_steps, lb=0, vtype=GRB.CONTINUOUS, name="p_ca")

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
            model.addConstr(t_ca_start[i] == (on_ca[:,i] * self.T_set).sum())

        # Constraints for on and off mode with u mode
        for i in range(self.num_time_steps):
            model.addConstr(p_ca[i] == (self.p_ca_rate * u_ca[i,:]).sum())

            for j in range(self.num_ca):
                if i == 0:
                    model.addConstr(u_ca[0,j] - 0 == on_ca[i,i] - off_ca[i,i])
                else:
                    model.addConstr(u_ca[i,j] - u_ca[i-1,j] == on_ca[i,j] - off_ca[i,j])

        ## Battery Energy Storage System (BESS) modeling
        # Variables for Solution
        p_bess_ch = model.addMVar(self.num_time_steps, lb=0, ub=self.p_ess_ch_max, vtype=GRB.CONTINUOUS, name="p_bess_ch")
        p_bess_dch = model.addMVar(self.num_time_steps, lb=0, ub=self.p_ess_dch_max, vtype=GRB.CONTINUOUS, name="p_bess_dch")
        u_bess = model.addMVar(self.num_time_steps, vtype=GRB.BINARY, name="u_bess")
        soc_bess = model.addMVar(self.num_time_steps, lb=self.soc_ess_min, ub=self.soc_ess_max, vtype=GRB.CONTINUOUS, name="soc_bess")

        # BESS Constraints
        for i in range(self.num_time_steps):
            # BESS charging/discharging power
            model.addConstr(p_bess_ch[i] <= self.p_ess_ch_max * u_bess[i])
            model.addConstr(p_bess_dch[i] <= self.p_ess_dch_max * (1 - u_bess[i]))

            # BESS state of charge
            model.addConstr(soc_bess[i] == soc_bess[i - 1] + self.delta_t * (p_bess_ch[i] * self.n_bess - p_bess_dch[i]/self.n_bess))

        model.addConstr(soc_bess[0] == self.soc_ess_max)
        model.addConstr(soc_bess[-1] == self.soc_ess_max)

        ## Plug-in Electric Vehicle (PEV) modeling
        # Variables for Solution
        p_pev_ch = model.addMVar(self.num_time_steps, lb=0, ub=self.p_pev_max, vtype=GRB.CONTINUOUS, name="p_pev_ch")
        p_pev_dch = model.addMVar(self.num_time_steps, lb=0, ub=self.p_pev_max, vtype=GRB.CONTINUOUS, name="p_pev_dch")
        u_pev = model.addMVar(self.num_time_steps, vtype=GRB.BINARY, name="u_pev")
        soc_pev = model.addMVar(self.num_pev_operation, lb=self.soc_pev_min, ub=self.soc_pev_max, vtype=GRB.CONTINUOUS, name="soc_pev")

        # PEV Constraints
        j = self.t_pev_arrive - 1
        for i in range(self.num_pev_operation):
            if i == 0:
                model.addConstr(soc_pev[i] == self.soc_pev_initial + self.delta_t * (p_pev_ch[j] * self.n_pev - p_pev_dch[j]/self.n_pev))
            else:
                model.addConstr(soc_pev[i] == soc_pev[i - 1] + self.delta_t * (p_pev_ch[j] * self.n_pev - p_pev_dch[j]/self.n_pev))
            j = j + 1

        model.addConstr(soc_pev[0] == self.soc_pev_initial)
        model.addConstr(soc_pev[-1] == self.soc_pev_max)

        for i in range(self.num_time_steps):
            model.addConstr(p_pev_ch[i] <= self.p_pev_max * u_pev[i])
            model.addConstr(p_pev_dch[i] <= self.p_pev_max * (1 - u_pev[i]))

            model.addConstr(p_pev_ch[i] <= self.p_pev_max * self.t_pev_range[i])
            model.addConstr(p_pev_dch[i] <= self.p_pev_max * self.t_pev_range[i])

        ## Heating-Ventilation-Air Conditioner (HVAC) modeling
        # Variables for Solution
        p_hvac_h = model.addMVar(self.num_time_steps, lb=0, ub=self.p_hvac_max, vtype=GRB.CONTINUOUS, name="p_hvac_h")
        p_hvac_c = model.addMVar(self.num_time_steps, lb=0, ub=self.p_hvac_max, vtype=GRB.CONTINUOUS, name="p_hvac_c")
        u_hvac = model.addMVar(self.num_time_steps, vtype=GRB.BINARY, name="u_hvac")
        theta_air_in = model.addMVar(self.num_time_steps, lb=self.theta_air_in_min, ub=self.theta_air_in_max, vtype=GRB.CONTINUOUS, name="theta_air_in")

        # HVAC Constraints
        for i in range(self.num_time_steps):
            # HVAC heating/cooling power
            model.addConstr(p_hvac_h[i] <= self.p_hvac_max * u_hvac[i])
            model.addConstr(p_hvac_c[i] <= self.p_hvac_max * (1 - u_hvac[i]))

            # Indoor air temperature
            if i <= (self.num_time_steps - 2):
                model.addConstr(theta_air_in[i+1] == ((1 - self.delta_t/self.coe1_hvac) * theta_air_in[i] + (self.delta_t/self.coe1_hvac) * self.theta_air_out[i] + self.cop_hvac*(p_hvac_h[i] - p_hvac_c[i])*self.delta_t/self.coe2_hvac))
            elif i == (self.num_time_steps - 1):
                model.addConstr(self.theta_air_in_setpoint == ((1 - self.delta_t/self.coe1_hvac) * theta_air_in[i] + (self.delta_t/self.coe1_hvac) * self.theta_air_out[i] + self.cop_hvac*(p_hvac_h[i] - p_hvac_c[i])*self.delta_t/self.coe2_hvac))

        model.addConstr(theta_air_in[0] == self.theta_air_in_setpoint)
        model.addConstr(theta_air_in[-1] == self.theta_air_in_setpoint)

        ## Electric Water Heater (EWH) modeling
        # Variables for Solution
        p_ewh = model.addMVar(self.num_time_steps, lb=0, ub=self.p_ewh_max, vtype=GRB.CONTINUOUS, name="p_ewh")
        theta_ewh = model.addMVar(self.num_time_steps, lb=self.theta_ewh_min, ub=self.theta_ewh_max, vtype=GRB.CONTINUOUS, name="theta_ewh")

        # EWH Constraints
        for i in range(self.num_time_steps):
            # Hot water temperature
            if i <= (self.num_time_steps - 2):
                if self.v_ewh_demand[i] == 0:
                    model.addConstr(theta_ewh[i+1] == (theta_air_in[i] + p_ewh[i] * self.n_ewh * self.C_w * self.delta_t - (theta_air_in[i] - theta_ewh[i]) * self.coe_ewh))
                elif self.v_ewh_demand[i] > 0:
                    model.addConstr(theta_ewh[i+1] == ((theta_ewh[i] * (self.v_ewh_max - self.v_ewh_demand[i]) + self.theta_cold_water * self.v_ewh_demand[i])/self.v_ewh_max))
            elif i == (self.num_time_steps - 1):
                model.addConstr(self.theta_ewh_setpoint == (theta_air_in[i] + p_ewh[i] * self.n_ewh * self.C_w * self.delta_t - (theta_air_in[i] - theta_ewh[i]) * self.coe_ewh))

        model.addConstr(theta_ewh[0] == self.theta_ewh_setpoint)
        model.addConstr(theta_ewh[-1] == self.theta_ewh_setpoint)

        ## Energy balance
        # Variables for Solution
        p_g2h = model.addMVar(self.num_time_steps, lb = 0, ub = self.p_g2h_max, vtype=GRB.CONTINUOUS, name="p_g2h")
        p_h2g = model.addMVar(self.num_time_steps, lb = 0, ub = self.p_h2g_max, vtype=GRB.CONTINUOUS, name="p_h2g")
        u_grid = model.addMVar(self.num_time_steps, vtype=GRB.BINARY, name="u_grid")

        # Energy balance Constraints
        for i in range(self.num_time_steps):
            model.addConstr(p_g2h[i] <= self.p_g2h_max * u_grid[i])
            model.addConstr(p_h2g[i] <= self.p_h2g_max * (1 - u_grid[i]))

            model.addConstr((p_g2h[i] + self.p_pv[i] + p_bess_dch[i] + p_pev_dch[i]) == (p_h2g[i] + self.p_nc[i] + p_ca[i] + p_bess_ch[i] + p_pev_ch[i] + p_hvac_h[i] + p_hvac_c[i] + p_ewh[i]))

        # Minimum Energy Cost
        energy_cost = gp.quicksum(self.delta_t * (p_g2h[i]*self.rtp[i] - p_h2g[i]*self.rtp[i]) for i in range(self.num_time_steps))

        ## Peak-to-Average (PAR)
        # Variables for Solution
        p_grid_max = model.addVar(lb=0, ub=self.p_g2h_max, vtype=GRB.CONTINUOUS, name="p_grid_max")
        u_grid_max = model.addMVar(self.num_time_steps, vtype=GRB.BINARY, name="u_grid_max")

        p_grid_average = gp.quicksum(((p_g2h[i] - p_h2g[i]) / self.num_time_steps) for i in range(self.num_time_steps))

        # PAR
        PAR = self.delta_t * (p_grid_max - p_grid_average)

        # PAR Constraints
        model.addConstr(u_grid_max.sum() == 1)
        for i in range(self.num_time_steps):
            model.addConstr(p_grid_max >= p_g2h[i] - p_h2g[i])
            model.addConstr(p_grid_max <= p_g2h[i] - p_h2g[i] + (1 - u_grid_max[i]) * 1000)

        ## Discomfort Index (DI)
        # Variables for Solution
        discomfort_index = model.addMVar(self.num_ca, lb=0, vtype=GRB.INTEGER, name = "discomfort_index")
        u_discomfort = model.addMVar(self.num_time_steps, vtype=GRB.BINARY, name="u_discomfort")

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
                'p_bess_ch': p_bess_ch.X,
                'p_bess_dch': p_bess_dch.X,
                'u_bess': u_bess.X,
                'soc_bess': soc_bess.X,
                'p_pev_ch': p_pev_ch.X,
                'p_pev_dch': p_pev_dch.X,
                'u_pev': u_pev.X,
                'soc_pev': soc_pev.X,
                'p_hvac_h': p_hvac_h.X,
                'p_hvac_c': p_hvac_c.X,
                'u_hvac': u_hvac.X,
                'theta_air_in': theta_air_in.X,
                'p_ewh': p_ewh.X,
                'theta_ewh': theta_ewh.X,
                'p_g2h': p_g2h.X,
                'p_h2g': p_h2g.X,
                'u_grid': u_grid.X,
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