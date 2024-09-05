# src/util.py

import numpy as np

def get_pv_output(ghi, p_pv_rate, n_pv, theta_air_out):
    """
    Calculate the PV generation data based on the given GHI and outdoor temperature.

    This function calculates the photovoltaic (PV) power output using a model that takes into account
    the Global Horizontal Irradiance (GHI), outdoor air temperature, and panel efficiency. The resulting
    power output is then clipped to ensure it does not exceed the maximum PV rate.

    Returns:
        numpy array: Preprocessed PV power output clipped to the maximum rate.
    """
    p_pv = p_pv_rate * (ghi * 0.25 + ghi * theta_air_out * 0.03 + (1.01 - 1.13 * n_pv) * np.square(ghi))
    return np.clip(p_pv, 0, p_pv_rate)

def get_nc_consumption(num_time_steps, num_nc, p_nc_rate, num_nc_operation, t_nc_start):
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
    t_nc_range = np.zeros((num_time_steps, num_nc))

    # Fill the time range matrix using vectorized operations
    for i in range(num_nc):
        start_idx = t_nc_start[i] - 1
        end_idx = start_idx + num_nc_operation[i]
        t_nc_range[start_idx:end_idx, i] = 1

    # Calculate the total power consumption by matrix multiplication
    p_nc = t_nc_range @ p_nc_rate

    return p_nc

def get_ca_availability(num_time_steps, num_ca, t_ca_start_max, t_ca_end_max):
    """
    Preprocesses the available time slots for controllable appliances based on 
    their operational constraints.

    This function creates a binary time-slot availability matrix for each controllable
    appliance, indicating the time periods during which each appliance can be operated.

    Returns:
        numpy array: A binary matrix indicating time slot availability for each controllable appliance.
    """
    # Initialize the time slot availability matrix
    t_ca_range = np.zeros((num_time_steps, num_ca))

    # Use vectorized operations to fill in the availability matrix
    for i in range(num_ca):
        t_ca_range[t_ca_start_max[i]-1 : t_ca_end_max[i], i] = 1

    return t_ca_range

def get_ev_availability(num_time_steps, arrival_time, departure_time):
    """
    Preprocesses the time range and operational profile for the Plug-in Electric Vehicle (PEV).

    This function generates a binary array indicating the availability of the PEV 
    for charging/discharging at each time step and calculates the total number of 
    time steps during which the PEV is available.

    Returns:
        tuple: A tuple containing:
            - numpy array: A binary array indicating the availability of the PEV 
                        for charging/discharging over the time steps.
            - int: The total number of time steps during which the PEV is available.
    """
    # Create a binary array indicating PEV availability using slicing
    ev_time_range = np.zeros(num_time_steps)
    ev_time_range[arrival_time - 1:departure_time] = 1

    # Calculate the number of available time steps using np.sum on the binary array
    num_pev_operation = int(np.sum(ev_time_range))
    
    return ev_time_range, num_pev_operation