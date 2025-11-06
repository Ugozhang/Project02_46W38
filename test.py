import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

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
    f_path = path_corrector(f_path)

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
    by call with path (must with double slash), return CT table as a tuples
    """

    # normalize path first
    f_path = path_corrector(f_path)

    CT_table = pd.read_csv(f_path,sep='\s+',comment="#",names=["V","CT"])
    return CT_table

def CT_interp(V,CT_table):
    """
    Pass wind speed as float, CT_table as DataFrame return interpolation of CT
    """
    return np.interp(V,CT_table["V"],CT_table["CT"])
    # return sp.interpolation.interp1d(CT_table["V"],CT_table["CT"],kind='linear')

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

def rho_CT_A(u, turbine_p, CT_table):
    """
    Compute aerodynamic thrust force on the blades:
        f_aero = 0.5 * rho * C_T * A * (u - x1_dot)*|u - x1_dot|
    """

    # parameter calculation
    A = np.pi * (turbine_p["Dr"]/2)**2
    rho = turbine_p["rho"]
    CT = CT_interp(u, CT_table)

    return rho*CT*A

def ydot(t, y, M, C, K, u, rho_CT_A):
    """
    Compute y'(t) = A*y + B(t) for the 2-DOF Turbie system.
    """
    NN = M.shape[0]
    x1, x2, dx1, dx2 = y
    y = np.array(y).reshape(-1, 1)  # 轉 column vector
    Minv = np.linalg.inv(M)

    # flatten f_aero calculation here for easier understanding
    area = np.pi * (turbine_p["Dr"]/2)**2
    rho = turbine_p["rho"]
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
    
def mass_spring_dapmer(t,y,M,C,K):
    """
    for reference
    """
    x, v = y
    dxdt = v
    dvdt = - (C/M)*v - (K/M)*x
    return [dxdt,dvdt]


f_path = "C:\DTU_prog\Project02_46W38\inputs\turbie_inputs\CT.txt"
f_path = path_corrector(f_path)
CT_Table = load_CT(f_path)
print(CT_Table)



f_path = "C:\DTU_prog\Project02_46W38\inputs\turbie_inputs\turbie_parameters.txt"
turbine_p,B =load_turbine_prop(f_path)

M,C,K = build_sys_matrices(turbine_p)


# --- 模擬設定 ---
u = 8.0          # 風速 m/s
y0 = [0, 0, 0, 0]  # 初始位移與速度
t_span = (0, 30)   # 模擬 30 秒
t_eval = np.linspace(*t_span, 400)
rho_CT_A = rho_CT_A(u, turbine_p , CT_Table)
# --- 模擬 ---
sol = sp.integrate.solve_ivp(ydot, t_span, y0, t_eval=t_eval, args=(M, C, K, u, rho_CT_A))

# --- 繪圖 ---
t = sol.t
x1 = sol.y[0]
x2 = sol.y[1]

plt.figure(figsize=(8,4))
plt.plot(t, x1, label="Blade deflection (x1)")
plt.plot(t, x2, label="Tower deflection (x2)")
plt.xlabel("Time [s]")
plt.ylabel("Displacement [m]")
plt.title("Turbie dynamic response under constant wind (u = 8 m/s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()