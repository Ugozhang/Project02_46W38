import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def path_corrector(path):
    """
    normalize f_path with "\\"
    """
    # Brutally replace all escape characters for
    fixed = ""
    escape_map = {
        "\t": "\\t",
        "\n": "\\n",
        "\r": "\\r",
        "\b": "\\b",
        "\f": "\\f",
        "\v": "\\v",
        "\a": "\\a",
    }
    fixed = "".join(escape_map.get(ch, ch) for ch in path)
    path = fixed

    # detect and replace all into \\ 
    if "\\\\" in path:
        return path
    elif "//" in path:
        path = path.replace("//","\\\\")
        return path
    elif "\\" in path:
        path = path.replace("\\","\\\\")
        return path
    elif "/" in path:
        path = path.replace("/","\\\\")
        return path
    else:
        return path
    
def load_turbine_prop(f_path):
    """
    by call with path (must with double slash), return propertis as 2 dicts with value and unit
    """
    # normalize path first
    # f_path = path_corrector(f_path)

    # value and unit dicts
    turbie_prop = {}
    turbie_prop_unit = {}

    # open file 
    with open(f_path,'r') as f:
        # each line split by # 
        for line in f:
            value, dic = line.split("#",1)
            # catch ValueError
            try:
                value = float(value.strip())
            except ValueError:
                print(f"This line has no valid value: {line!r}")
                continue
            # Split key and unit
            key = dic.split()[0]
            unit = dic.split()[1]
            # Assign key, value and unit
            turbie_prop[key] = value
            turbie_prop_unit[key] = unit

    return turbie_prop, turbie_prop_unit

def load_CT(f_path):
    """
def load_WSdata(f_path, skip_first_n_secs=None):
    by call with path (must with double slash), return CT table as a tuples
    """

    # normalize path first
    # f_path = path_corrector(f_path)

    CT_table = pd.read_csv(f_path,sep='\s+',comment="#",names=["V","CT"])
    return CT_table

def load_WSdata(f_path, skip_first_n_secs=None):
    """
    by call with path (must with double slash), return CT table as a tuples
    """

    # normalize path first
    # f_path = path_corrector(f_path)

    df = pd.read_csv(f_path,sep='\s+')
    df_t = df[df["Time(s)"] >= 60]
    return df_t

def rho_CT_A(df, turbine_p, CT_table):
    """
    Compute the constant part of aerodynamic force equation:
        rho * C_T * A
    """

    # parameter calculation
    A = np.pi * (turbine_p["Dr"]/2)**2
    rho = turbine_p["rho"]
    CT = np.interp(df["V(m/s)"].mean(),CT_table["V"],CT_table["CT"])
    
    return rho*CT*A

def build_sys_matrices(turbine_p):
    """
    Build M, C, K matrices by passing turbie parameters
    """
    # Assign values
    m1 = 3*turbine_p["mb"]
    m2 = turbine_p["mn"] + turbine_p["mt"] + turbine_p["mh"]
    c1, c2 = turbine_p["c1"], turbine_p["c2"]
    k1, k2 = turbine_p["k1"], turbine_p["k2"]
    
    # Build matrices
    M = np.array([[ m1, 0],
                  [ 0, m2]])
    C = np.array([[ c1, -c1],
                  [ -c1, c1 + c2]])
    K = np.array([[ k1, -k1],
                  [ -k1, k1 + k2]])
    
    return M, C, K

def build_ws_func(df):
    
    u= df["V(m/s)"].to_numpy()
    t= df["Time(s)"].to_numpy()

    def u_of_t(t_query):
        return np.interp(t_query, df["Time(s)"], df["V(m/s)"])
    return u_of_t, t, u

def ydot(t, y, M, C, K, u_of_t, rho_CT_A):
    """
    Compute y'(t) = A*y + B(t) for the 2-DOF Turbie system.
    """
    NN = M.shape[0]
    x1, x2, dx1, dx2 = y
    y = np.array(y).reshape(-1, 1)  # 轉 column vector
    Minv = np.linalg.inv(M)

    # flatten f_aero calculation here for easier understanding
    u = u_of_t(t)
    u_rel = u - dx1
    f1_t = 0.5 * rho_CT_A * (u_rel) * abs(u_rel)
    f2_t = 0

    # list forcing vector
    F  = np.array([[ f1_t],
                   [ f2_t]])

    # list A, B matrices
    A = np.block([[ np.zeros((NN,NN)), np.eye(NN)],
                  [ -(Minv) @ K, -(Minv) @ C]])
    B = np.block([[ np.zeros((NN,1))],
                  [ (Minv) @ F]])
    
    # list dy/dt = A @ y + B
    dy = A @ y + B
    return dy.flatten()


def DrawPlot_Deflec_t(t, x1, x2, u_vec):
    """
    Give time, blade, tower deflection datasets and windspeed set; draw and return figure and axese objects.
    """

    # plot x1, x2 - t
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(t, x1, label="Blade Deflec. (x1)", color="tab:cyan")
    ax1.plot(t, x2, label="Tower Deflec. (x2)", color="tab:blue")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Displacement [m]")
    ax1.legend(loc="upper left")

    # plot u - t as right side y axis
    ax2 = ax1.twinx()
    ax2.plot(t, u_vec, label="Wind (u)", color="tab:red", alpha=0.5)
    ax2.set_ylabel("Wind [m/s]", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.legend(loc="upper right")

    return fig, ax1, ax2

