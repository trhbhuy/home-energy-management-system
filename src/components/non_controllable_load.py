import numpy as np
import gurobipy as gp
from gurobipy import GRB

class NCLoad:
    def __init__(self, T_num, T_set, delta_t, num_nc, p_nc_rate, num_nc_operation, t_nc_start):
        """Initialize parameters."""
        self.T_num = T_num
        self.T_set = T_set
        self.delta_t = delta_t

        self.num_nc = num_nc
        self.p_nc_rate = p_nc_rate
        self.num_nc_operation = num_nc_operation
        self.t_nc_start = t_nc_start

    def get_power_consumption(self):
        """
        Calculate the power consumption profile of non-controllable (NC) appliances.

        This function creates a time-based range matrix for each non-controllable appliance
        based on its start time and operational duration. It then calculates the total 
        power consumption profile by summing the power usage of all non-controllable 
        appliances over all time steps.

        Returns:
            numpy array: Combined power profile for all NC appliances over the time steps.
        """
        # Initialize the time range matrix for all NC appliances
        t_nc_range = np.zeros((self.T_num, self.num_nc))

        # Fill the time range matrix using vectorized operations
        for i in range(self.num_nc):
            start_idx = self.t_nc_start[i] - 1
            end_idx = start_idx + self.num_nc_operation[i]
            t_nc_range[start_idx:end_idx, i] = 1

        # Calculate the total power consumption by matrix multiplication
        p_nc = t_nc_range @ self.p_nc_rate

        return p_nc
