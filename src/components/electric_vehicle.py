import numpy as np
import gurobipy as gp
from gurobipy import GRB

class EV:
    def __init__(self, p_ev_ch_max, p_ev_dch_max, soc_ev_max, soc_ev_min, soc_ev_setpoint, n_ev_ch, n_ev_dch, T_num, T_set, delta_t):
        # Initialize EV parameters
        self.p_ev_ch_max = p_ev_ch_max
        self.p_ev_dch_max = p_ev_dch_max
        self.soc_ev_max = soc_ev_max
        self.soc_ev_min = soc_ev_min
        self.soc_ev_setpoint = soc_ev_setpoint
        self.n_ev_ch = n_ev_ch
        self.n_ev_dch = n_ev_dch

        # Time settings
        self.T_num = T_num
        self.T_set = T_set
        self.delta_t = delta_t
        
    def add_variables(self, model):
        # Add variables to the model
        p_ev_ch = model.addVars(self.T_set, vtype=GRB.CONTINUOUS, name="p_ev_ch")
        p_ev_dch = model.addVars(self.T_set, vtype=GRB.CONTINUOUS, name="p_ev_dch")
        u_ev_ch = model.addVars(self.T_set, vtype=GRB.BINARY, name="u_ev_ch")
        u_ev_dch = model.addVars(self.T_set, vtype=GRB.BINARY, name="u_ev_dch")
        soc_ev = model.addVars(self.T_set, vtype=GRB.CONTINUOUS, name="soc_ev")

        return p_ev_ch, p_ev_dch, u_ev_ch, u_ev_dch, soc_ev

    def add_constraints(self, model, p_ev_ch, p_ev_dch, u_ev_ch, u_ev_dch, soc_ev, initial_soc_ev, t_ev_arrive, t_ev_depart, ev_time_range):
        # Add constraints to the model
        for t in self.T_set:
            # Charging and discharging power constraints
            model.addConstr(p_ev_ch[t] <= self.p_ev_ch_max * u_ev_ch[t])
            model.addConstr(p_ev_dch[t] <= self.p_ev_dch_max * u_ev_dch[t])
            model.addConstr(u_ev_ch[t] + u_ev_dch[t] >= 0)
            model.addConstr(u_ev_ch[t] + u_ev_dch[t] <= 1)

            # Available time range constraints
            model.addConstr(p_ev_ch[t] <= self.p_ev_ch_max * ev_time_range[t])
            model.addConstr(p_ev_dch[t] <= self.p_ev_ch_max * ev_time_range[t])

            model.addConstr(soc_ev[t] >= self.soc_ev_min * ev_time_range[t])
            model.addConstr(soc_ev[t] <= self.soc_ev_max * ev_time_range[t])

        # State of charge (SOC) of EV constraints
        for t in range(t_ev_arrive, t_ev_depart+1):
            if t == t_ev_arrive:
                model.addConstr(soc_ev[t] == initial_soc_ev + self.delta_t * (p_ev_ch[t] * self.n_ev_ch - p_ev_dch[t] / self.n_ev_dch))
            else:
                model.addConstr(soc_ev[t] == soc_ev[t - 1] + self.delta_t * (p_ev_ch[t] * self.n_ev_ch - p_ev_dch[t] / self.n_ev_dch))

        # Set the initial and final SOC of EV
        model.addConstr(soc_ev[t_ev_arrive] == initial_soc_ev)
        model.addConstr(soc_ev[t_ev_depart] == self.soc_ev_setpoint)

    def get_cost(self, F_ev):
        return gp.quicksum(F_ev[t] * self.phi_ev for t in self.T_set)

    def get_ev_availablity(self, initial_soc_ev_percentage, ArriveTime, DepartureTime):
        """Define availablity of EV action."""
        initial_soc_ev = self.soc_ev_max * initial_soc_ev_percentage
        t_ev_arrive = int(ArriveTime - 12)   # Example: ev arrives home at 17:00 --> interval 5
        t_ev_depart = int(DepartureTime + 12)   # Example: ev is fully charged and departs at 8:00 --> interval 20

        ev_time_range = np.zeros(self.T_num)
        ev_time_range[t_ev_arrive:t_ev_depart+1] = 1
        self.num_pv_operation = int(np.sum(ev_time_range))

        return initial_soc_ev, t_ev_arrive, t_ev_depart, ev_time_range
