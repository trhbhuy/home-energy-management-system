import numpy as np
import gurobipy as gp
from gurobipy import GRB

class HVAC:
    def __init__(self, T_num, T_set, delta_t, p_hvac_max, cop_hvac, R_build, C_air_in, m_air_in, coe1_hvac, coe2_hvac, theta_air_out, theta_air_in_setpoint, theta_air_in_max_offset, theta_air_in_min_offset):
        """Initialize parameters."""
        self.T_num = T_num
        self.T_set = T_set
        self.delta_t = delta_t

        self.p_hvac_max = p_hvac_max
        self.cop_hvac = cop_hvac
        self.R_build = R_build
        self.C_air_in = C_air_in
        self.m_air_in = m_air_in
        self.coe1_hvac = coe1_hvac
        self.coe2_hvac = coe2_hvac
        self.theta_air_out = theta_air_out
        self.theta_air_in_setpoint = theta_air_in_setpoint
        self.theta_air_in_max = theta_air_in_setpoint + theta_air_in_max_offset
        self.theta_air_in_min = theta_air_in_setpoint - theta_air_in_min_offset

    def add_variables(self, model):
        """Add variables to the model."""
        p_hvac_h = model.addMVar(self.T_num, vtype=GRB.CONTINUOUS, name="p_hvac_h")
        p_hvac_c = model.addMVar(self.T_num, vtype=GRB.CONTINUOUS, name="p_hvac_c")
        u_hvac_h = model.addMVar(self.T_num, vtype=GRB.BINARY, name="u_hvac_h")
        u_hvac_c = model.addMVar(self.T_num, vtype=GRB.BINARY, name="u_hvac_c")
        theta_air_in = model.addMVar(self.T_num, lb=self.theta_air_in_min, ub=self.theta_air_in_max, vtype=GRB.CONTINUOUS, name="theta_air_in")

        return p_hvac_h, p_hvac_c, u_hvac_h, u_hvac_c, theta_air_in

    def add_constraints(self, model, p_hvac_h, p_hvac_c, u_hvac_h, u_hvac_c, theta_air_in):
        """Add constraints to the model."""
        for t in self.T_set:
            # HVAC heating/cooling power
            model.addConstr(p_hvac_h[t] <= self.p_hvac_max * u_hvac_h[t])
            model.addConstr(p_hvac_c[t] <= self.p_hvac_max * u_hvac_c[t])
            model.addConstr(u_hvac_h[t] + u_hvac_c[t] >= 0)
            model.addConstr(u_hvac_h[t] + u_hvac_c[t] <= 1)

            # Indoor air temperature
            if t <= self.T_set[-2]:
                model.addConstr(theta_air_in[t+1] == (1 - self.delta_t/self.coe1_hvac) * theta_air_in[t] + (self.delta_t/self.coe1_hvac) * self.theta_air_out[t] + self.cop_hvac*(p_hvac_h[t] - p_hvac_c[t])*self.delta_t/self.coe2_hvac)
            elif t == self.T_set[-1]:
                model.addConstr(self.theta_air_in_setpoint == (1 - self.delta_t/self.coe1_hvac) * theta_air_in[t] + (self.delta_t/self.coe1_hvac) * self.theta_air_out[t] + self.cop_hvac*(p_hvac_h[t] - p_hvac_c[t])*self.delta_t/self.coe2_hvac)

        model.addConstr(theta_air_in[0] == self.theta_air_in_setpoint)
        model.addConstr(theta_air_in[-1] == self.theta_air_in_setpoint)