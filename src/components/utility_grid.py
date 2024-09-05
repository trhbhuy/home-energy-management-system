import gurobipy as gp
from gurobipy import GRB

class Grid:
    def __init__(self, T_num, T_set, delta_t, p_grid_pur_max, p_grid_exp_max, phi_rtp=None, r_grid_pur=None, r_grid_exp=None):
        """Initialize parameters."""
        self.T_num = T_num
        self.T_set = T_set
        self.delta_t = delta_t

        self.p_grid_pur_max = p_grid_pur_max
        self.p_grid_exp_max = p_grid_exp_max
        self.phi_rtp = phi_rtp
        self.r_grid_pur = r_grid_pur
        self.r_grid_exp = r_grid_exp
        
    def add_variables(self, model):
        """Add variables to the model."""
        p_grid_pur = model.addMVar(self.T_num, ub = self.p_grid_pur_max, vtype=GRB.CONTINUOUS, name="p_grid_pur")
        p_grid_exp = model.addMVar(self.T_num, ub = self.p_grid_pur_max, vtype=GRB.CONTINUOUS, name="p_grid_exp")
        u_grid_pur = model.addMVar(self.T_num, vtype=GRB.BINARY, name="u_grid_pur")
        u_grid_exp = model.addMVar(self.T_num, vtype=GRB.BINARY, name="u_grid_exp")

        return p_grid_pur, p_grid_exp, u_grid_pur, u_grid_exp

    def add_constraints(self, model, p_grid_pur, p_grid_exp, u_grid_pur, u_grid_exp):
        """Add constraints to the model."""
        for t in self.T_set:
            model.addConstr(p_grid_pur[t] <= self.p_grid_pur_max * u_grid_pur[t])
            model.addConstr(p_grid_exp[t] <= self.p_grid_exp_max * u_grid_exp[t])

            model.addConstr(u_grid_pur[t] + u_grid_exp[t] >= 0)
            model.addConstr(u_grid_pur[t] + u_grid_exp[t] <= 1)

    def get_cost(self, rtp, p_grid_pur, p_grid_exp):
        """Add constraints to the model."""
        # Cost exchange with utility grid
        F_grid = gp.quicksum(self.delta_t * (p_grid_pur[i] * rtp[i] - p_grid_exp[i] * rtp[i] * self.phi_rtp) for i in self.T_set)

        return F_grid
