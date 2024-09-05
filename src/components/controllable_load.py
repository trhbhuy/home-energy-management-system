import numpy as np
import gurobipy as gp
from gurobipy import GRB

class CALoad:
    def __init__(self, T_num, T_set, delta_t, num_ca, p_ca_rate, num_ca_operation, t_ca_start_max, t_ca_end_max, t_ca_start_prefer, t_ca_end_prefer):
        """Initialize parameters."""
        self.T_num = T_num
        self.T_set = T_set
        self.delta_t = delta_t
        
        self.num_ca = num_ca
        self.p_ca_rate = np.array(p_ca_rate)
        self.num_ca_operation = np.array(num_ca_operation)
        self.t_ca_start_max = np.array(t_ca_start_max)
        self.t_ca_end_max = np.array(t_ca_end_max)
        self.t_ca_start_prefer = np.array(t_ca_start_prefer)
        self.t_ca_end_prefer = np.array(t_ca_end_prefer)

        self.t_ca_range = self.get_ca_availability()

        self.T_set_24 = np.arange(1, T_num+1)

    def add_variables(self, model):
        """Add variables to the model."""
        u_ca = model.addMVar((self.T_num,self.num_ca), vtype=GRB.BINARY, name="u_ca")
        on_ca = model.addMVar((self.T_num,self.num_ca), vtype=GRB.BINARY, name="on_ca")
        off_ca = model.addMVar((self.T_num,self.num_ca), vtype=GRB.BINARY, name="off_ca")
        t_ca_start = model.addMVar(self.num_ca, lb=0, vtype=GRB.INTEGER, name="t_ca_start")
        p_ca = model.addMVar(self.T_num, lb=0, vtype=GRB.CONTINUOUS, name="p_ca")

        # discomfort_index = model.addMVar(self.num_ca, lb=0, vtype=GRB.INTEGER, name = "discomfort_index")
        # u_discomfort = model.addMVar(self.T_num, vtype=GRB.BINARY, name="u_discomfort")

        return u_ca, on_ca, off_ca, t_ca_start, p_ca

    def add_constraints(self, model, u_ca, on_ca, off_ca, t_ca_start, p_ca):
        """Add constraints to the model."""
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

    def get_ca_availability(self):
        """
        Preprocesses the available time slots for controllable appliances based on 
        their operational constraints.
        """
        # Initialize the time slot availability matrix
        t_ca_range = np.zeros((self.T_num, self.num_ca))

        # Use vectorized operations to fill in the availability matrix
        for i in range(self.num_ca):
            t_ca_range[self.t_ca_start_max[i]-1 : self.t_ca_end_max[i], i] = 1

        return t_ca_range
    
    def get_discomfort_index(self, model, t_ca_start):
        """Calculate the discomfort index for the controllable appliances."""
        discomfort_index = model.addMVar(self.num_ca, lb=0, vtype=GRB.INTEGER, name = "discomfort_index")
        u_discomfort = model.addMVar(self.T_num, vtype=GRB.BINARY, name="u_discomfort")

        # DI Constraint
        for i in range(self.num_ca):
            model.addConstr(discomfort_index[i] >= self.t_ca_start_prefer[i] - t_ca_start[i])
            model.addConstr(discomfort_index[i] >= t_ca_start[i] - self.t_ca_start_prefer[i])
            model.addConstr(discomfort_index[i] <= self.t_ca_start_prefer[i] - t_ca_start[i] + u_discomfort[i] * 1000)
            model.addConstr(discomfort_index[i] <= t_ca_start[i] - self.t_ca_start_prefer[i] + (1 - u_discomfort[i]) * 1000)

        return discomfort_index.sum()

