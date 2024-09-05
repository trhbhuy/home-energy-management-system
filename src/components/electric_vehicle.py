import numpy as np
import gurobipy as gp
from gurobipy import GRB

class EV:
    def __init__(self, T_num, T_set, delta_t, p_ev_ch_max, p_ev_dch_max, n_ev_ch, n_ev_dch, soc_ev_max, soc_ev_min, soc_ev_setpoint):
        """Initialize parameters."""
        self.T_num = T_num
        self.T_set = T_set
        self.delta_t = delta_t
 
        self.p_ev_ch_max = p_ev_ch_max
        self.p_ev_dch_max = p_ev_dch_max
        self.soc_ev_max = soc_ev_max
        self.soc_ev_min = soc_ev_min
        self.soc_ev_setpoint = soc_ev_setpoint
        self.n_ev_ch = n_ev_ch
        self.n_ev_dch = n_ev_dch
        
    def add_variables(self, model):
        """Add variables to the model."""
        p_ev_ch = model.addMVar(self.T_num, vtype=GRB.CONTINUOUS, name="p_ev_ch")
        p_ev_dch = model.addMVar(self.T_num, vtype=GRB.CONTINUOUS, name="p_ev_dch")
        u_ev_ch = model.addMVar(self.T_num, vtype=GRB.BINARY, name="u_ev_ch")
        u_ev_dch = model.addMVar(self.T_num, vtype=GRB.BINARY, name="u_ev_dch")
        soc_ev = model.addMVar(self.T_num, vtype=GRB.CONTINUOUS, name="soc_ev")

        return p_ev_ch, p_ev_dch, u_ev_ch, u_ev_dch, soc_ev

    def add_constraints(self, model, p_ev_ch, p_ev_dch, u_ev_ch, u_ev_dch, soc_ev, t_ev_arrive, t_ev_depart, ev_time_range, soc_ev_initial):
        """Add constraints to the model."""
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
                model.addConstr(soc_ev[t] == soc_ev_initial + self.delta_t * (p_ev_ch[t] * self.n_ev_ch - p_ev_dch[t] / self.n_ev_dch))
            else:
                model.addConstr(soc_ev[t] == soc_ev[t-1] + self.delta_t * (p_ev_ch[t] * self.n_ev_ch - p_ev_dch[t] / self.n_ev_dch))

        # Set the initial and final SOC of EV
        model.addConstr(soc_ev[t_ev_arrive] == soc_ev_initial)
        model.addConstr(soc_ev[t_ev_depart] == self.soc_ev_setpoint)

    def get_ev_availablity(self, ArriveTime, DepartureTime, soc_ev_initial, soc_ev_initial_percentage=None):
        """Define availablity of EV action."""
        if soc_ev_initial_percentage is not None:
            soc_ev_initial = self.soc_ev_max * soc_ev_initial_percentage

        t_ev_arrive = ArriveTime
        t_ev_depart = DepartureTime

        ev_time_range = np.zeros(self.T_num)
        ev_time_range[t_ev_arrive:t_ev_depart+1] = 1
        self.num_pv_operation = int(np.sum(ev_time_range))

        return t_ev_arrive, t_ev_depart, ev_time_range, soc_ev_initial
