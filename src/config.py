import os
import numpy as np

# Define base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import settings from config
# Real-time pricing and weather data
RTP = np.array([0.015, 0.019, 0.019, 0.022, 0.022, 0.025, 0.025, 0.027, 0.027, 0.032, 0.032, 0.034, 
                0.034, 0.038, 0.038, 0.041, 0.041, 0.046, 0.046, 0.048, 0.048, 0.036, 0.036, 0.031, 
                0.031, 0.028, 0.028, 0.025, 0.025, 0.022, 0.022, 0.020, 0.020, 0.016, 0.016, 0.013, 
                0.013, 0.012, 0.012, 0.012, 0.012, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.015])

GHI = np.array([0.24090, 0.24090, 0.46741, 0.46741, 0.68210, 0.68210, 0.84694, 0.84694, 0.97304, 0.97304, 1.02085, 1.02085, 
                0.97856, 0.97856, 0.90271, 0.90271, 0.73186, 0.73186, 0.52716, 0.52716, 0.30762, 0.30762, 0.09686, 0.096860, 
                0.02559, 0.02559, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.044640, 0.044640])

THETA_AIR_OUT = np.array([19.72, 19.72, 21.37, 21.37, 23.39, 23.39, 25.10, 25.10, 26.81, 26.81, 28.36, 28.36, 
                          29.61, 29.61, 30.56, 30.56, 31.18, 31.18, 31.46, 31.46, 31.41, 31.41, 31.05, 31.05, 
                          30.77, 30.77, 29.17, 29.17, 27.40, 27.40, 25.92, 25.92, 24.22, 24.22, 22.62, 22.62, 
                          21.46, 21.46, 20.59, 20.59, 19.90, 19.90, 19.29, 19.29, 18.80, 18.80, 18.95, 18.95])

# Time settings
T_NUM = 48
T_SET = np.arange(T_NUM)
DELTA_T = 24 / T_NUM

# Grid exchange parameters
P_GRID_PUR_MAX = 10  # Maximum power purchase from grid (kW)
P_GRID_EXP_MAX = 4  # Maximum power export to grid (kW)
PHI_RTP = 1  # Real-time pricing factor

# Solar PV parameters
P_PV_RATE = 1.0  # Rated power of PV (kW)
N_PV = 0.167  # Efficiency factor for PV
PHI_PV = 0.24  # OM cost for PV

# Non-controllable appliances
NUM_NC = 3
P_NC_RATE = [0.35, 0.10, 0.10]
NUM_NC_OPERATION = [48, 12, 12]
T_NC_START = [1, 22, 24]

# Controllable appliances
NUM_CA = 7
P_CA_RATE = [2.5, 3.0, 2.5, 3, 1.7, 0.1, 1.2]
NUM_CA_OPERATION = [4, 3, 2, 1, 1, 4, 1]
T_CA_START_MAX = [1, 2, 11, 2, 2, 19, 4]
T_CA_END_MAX = [19, 9, 21, 3, 3, 33, 19]
T_CA_START_PREFER = [5, 5, 13, 3, 3, 23, 5]
T_CA_END_PREFER = [8, 7, 14, 3, 3, 26, 5]

# Energy storage system (ESS) parameters
P_ESS_CH_MAX = 2  # Maximum charging power (kW)
P_ESS_DCH_MAX = 2  # Maximum discharging power (kW)
N_ESS_CH = 0.98  # Charging efficiency
N_ESS_DCH = 0.98  # Discharging efficiency
SOC_ESS_MAX = 5  # Maximum state of charge (kWh)
ESS_DOD = 0.7  # Depth of discharge
SOC_ESS_MIN = (1 - ESS_DOD) * SOC_ESS_MAX  # Minimum state of charge considering DoD
SOC_ESS_SETPOINT = SOC_ESS_MAX  # Reference state of charge for ESS

# Electric vehicle (EV) parameters
P_EV_CH_MAX = 3  # Maximum charging power (kW)
P_EV_DCH_MAX = 3  # Maximum discharging power (kW)
N_EV_CH = 0.98  # Charging efficiency
N_EV_DCH = 0.98  # Discharging efficiency
SOC_EV_MAX = 22  # Maximum state of charge (kWh)
EV_DOD = 0.8  # Depth of discharge
SOC_EV_MIN = (1 - EV_DOD) * SOC_EV_MAX  # Minimum state of charge considering DoD
SOC_EV_SETPOINT = SOC_EV_MAX  # Reference state of charge for EV
T_EV_ARRIVE = int(25)  # Time of arrival for PEV
T_EV_DEPART = int(47)  # Time of departure for PEV
SOC_EV_INIT = 11  # Initial state of charge for PEV

# Heating, Ventilation, and Air Conditioning (HVAC)
P_HVAC_MAX = 2  # Maximum power for HVAC (kW)
COP_HVAC = 1.2  # Coefficient of performance for HVAC
R_BUILD = 3.2 * 1e-6  # Thermal resistance of building (kW/K)
C_AIR_IN = 1.01  # Thermal capacitance of indoor air (kWh/K)
M_AIR_IN = 1778.40  # Mass of indoor air (kg)
COE1_HVAC = 1000 * M_AIR_IN * C_AIR_IN * R_BUILD  # Coefficient 1 for HVAC
COE2_HVAC = 0.000277 * M_AIR_IN * C_AIR_IN  # Coefficient 2 for HVAC
THETA_AIR_IN_SETPOINT = 23  # Setpoint indoor air temperature (°C)
THETA_AIR_IN_MAX_OFFSET = 0.5  # Maximum offset from setpoint (°C)
THETA_AIR_IN_MIN_OFFSET = 0.5  # Minimum offset from setpoint (°C)

# Electric Water Heater (EWH)
P_EWH_MAX = 2.1  # Maximum power for EWH (kW)
C_W = 1.52  # Thermal capacitance of water (kWh/K)
R_W = 863.40  # Thermal resistance of water (kW/K)
V_EWH_MAX = 50  # Maximum volume of EWH (L)
N_EWH = 0.9  # Efficiency factor for EWH
COE_EWH = np.exp(-DELTA_T / (R_W * C_W))  # Coefficient of EWH
THETA_EWH_SETPOINT = 45  # Setpoint water temperature for EWH (°C)
THETA_EWH_MAX = 60  # Maximum water temperature for EWH (°C)
THETA_EWH_MIN = 40  # Minimum water temperature for EWH (°C)
THETA_COLD_WATER = 20  # Cold water temperature (°C)
V_EWH_DEMAND = [0, 10, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 
                0,  3, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 
                0,  0, 0, 3, 10, 0, 0, 0, 0, 0, 0, 0,  
                0,  0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0] # Water demand (L)

def print_config():
    """
    Utility function to print the current configuration settings.
    Useful for debugging and verification.
    """
    print("Hydrogen Refueling Station Configuration Settings:")
    print(f"Time horizon: {T_NUM} hours")
    print(f"Time step: {DELTA_T} hours")
    print(f"Grid purchase power: {P_GRID_PUR_MAX} kW")
    print(f"Grid export power: {P_GRID_EXP_MAX} kW")
    # Add more configuration details as needed

if __name__ == "__main__":
    print_config()