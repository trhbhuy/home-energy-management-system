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
        p_grid_pur = model.addVars(self.T_set, vtype=GRB.CONTINUOUS, name="p_grid_pur")
        p_grid_exp = model.addVars(self.T_set, vtype=GRB.CONTINUOUS, name="p_grid_exp")
        u_grid_pur = model.addVars(self.T_set, vtype=GRB.BINARY, name="u_grid_pur")
        u_grid_exp = model.addVars(self.T_set, vtype=GRB.BINARY, name="u_grid_exp")

        return p_grid_pur, p_grid_exp, u_grid_pur, u_grid_exp

    def add_constraints(self, model, p_grid_pur, p_grid_exp, u_grid_pur, u_grid_exp):
        """Add constraints to the model."""
        for t in self.T_set:
            model.addConstr(p_grid_pur[t] <= self.p_grid_pur_max * u_grid_pur[t])
            model.addConstr(p_grid_exp[t] <= self.p_grid_exp_max * u_grid_exp[t])

            model.addConstr(u_grid_pur[t] + u_grid_exp[t] >= 0)
            model.addConstr(u_grid_pur[t] + u_grid_exp[t] <= 1)

    def get_cost(self, rtp, p_grid_pur, p_grid_exp):
        """Calculate cost exchange with utility grid."""
        # Cost exchange with utility grid
        F_grid = gp.quicksum(self.delta_t * (p_grid_pur[i] * rtp[i] - p_grid_exp[i] * rtp[i] * self.phi_rtp) for i in self.T_set)

        return F_grid

    def get_max_power(self, model, p_grid_pur, p_grid_exp):
        """Define max power exchange with utility grid."""
        p_grid_max = model.addVar(vtype=GRB.CONTINUOUS, name="p_grid_max")
        # u_grid_max = model.addMVar(self.T_num, vtype=GRB.BINARY, name="u_grid_max")

        # # PAR Constraints
        # model.addConstr(u_grid_max.sum() == 1)
        # for i in range(self.T_num):
        #     model.addConstr(p_grid_max >= p_grid_pur[i] - p_grid_exp[i])
        #     model.addConstr(p_grid_max <= p_grid_pur[i] - p_grid_exp[i] + (1 - u_grid_max[i]) * 1000)
        
        model.addGenConstrMax(p_grid_max, p_grid_pur)

        return p_grid_max

    # def get_average_power(self, p_grid_pur, p_grid_exp):
    #     """Define average power exchange with utility grid."""        
    #     return (p_grid_pur - p_grid_exp).sum() / self.T_num



