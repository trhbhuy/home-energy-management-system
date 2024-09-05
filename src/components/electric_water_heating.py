import numpy as np
import gurobipy as gp
from gurobipy import GRB

class EWH:
    def __init__(self, T_num, T_set, delta_t, p_ewh_max, C_w, R_w, v_ewh_max, n_ewh, COE_EWH, theta_ewh_setpoint, theta_ewh_max, theta_ewh_min, theta_cold_water, v_ewh_demand):
        """Initialize parameters."""
        self.T_num = T_num
        self.T_set = T_set
        self.delta_t = delta_t

        self.p_ewh_max = p_ewh_max
        self.C_w = C_w
        self.R_w = R_w
        self.v_ewh_max = v_ewh_max
        self.n_ewh = n_ewh
        self.coe_ewh = COE_EWH
        self.theta_ewh_setpoint = theta_ewh_setpoint
        self.theta_ewh_max = theta_ewh_max
        self.theta_ewh_min = theta_ewh_min
        self.theta_cold_water = theta_cold_water
        self.v_ewh_demand = v_ewh_demand

    def add_variables(self, model):
        """Add variables to the model."""
        p_ewh = model.addMVar(self.T_num, lb=0, ub=self.p_ewh_max, vtype=GRB.CONTINUOUS, name="p_ewh")
        theta_ewh = model.addMVar(self.T_num, lb=self.theta_ewh_min, ub=self.theta_ewh_max, vtype=GRB.CONTINUOUS, name="theta_ewh")

        return p_ewh, theta_ewh

    def add_constraints(self, model, p_ewh, theta_ewh, theta_air_in):
        """Add constraints to the model."""
        for t in self.T_set:            
            # Hot water temperature
            if t <= self.T_set[-2]:
                if self.v_ewh_demand[t] == 0:
                    model.addConstr(theta_ewh[t+1] == theta_air_in[t] + p_ewh[t] * self.n_ewh * self.C_w * self.delta_t - (theta_air_in[t] - theta_ewh[t]) * self.coe_ewh)
                elif self.v_ewh_demand[t] > 0:
                    model.addConstr(theta_ewh[t+1] == (theta_ewh[t] * (self.v_ewh_max - self.v_ewh_demand[t]) + self.theta_cold_water * self.v_ewh_demand[t])/self.v_ewh_max)
            elif t == self.T_set[-1]:
                model.addConstr(self.theta_ewh_setpoint == theta_air_in[t] + p_ewh[t] * self.n_ewh * self.C_w * self.delta_t - (theta_air_in[t] - theta_ewh[t]) * self.coe_ewh)

        model.addConstr(theta_ewh[0] == self.theta_ewh_setpoint)
        model.addConstr(theta_ewh[self.T_set[-1]] == self.theta_ewh_setpoint)