import os
from pathlib import Path

# Make sure running under same folder each time by calling the path where main.py it is 
main_dir = Path(__file__).resolve().parent
os.chdir(main_dir)

import turbie_mod as turbie
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

# CT_table_path = ".\inputs\turbie_inputs\CT.txt"
CT_table_path = main_dir / "inputs" / "turbie_inputs" / "CT.txt"
CT_table = turbie.load_CT(CT_table_path)

# turbie_params_path = ".\inputs\turbie_inputs\turbie_parameters.txt"
turbie_params_path = main_dir / "inputs" / "turbie_inputs" / "turbie_parameters.txt"
turbie_params, trbie_units = turbie.load_turbine_prop(turbie_params_path)

wind_files_path = ".\inputs\wind_files\wind_TI_0.1\wind_5_ms_TI_0.1.txt"
df = turbie.load_WSdata(wind_files_path)
outputs_path = ""


rho_CT_A = turbie.rho_CT_A(df, turbie_params, CT_table)
M, C, K = turbie.build_sys_matrices(turbie_params)
u_of_t, t_vec, u_vec = turbie.build_ws_func(df)

# --- 模擬設定 ---
u = df["V(m/s)"]          # 風速 m/s
y0 = [0, 0, 0, 0]  # 初始位移與速度
t_span = (t_vec[0], t_vec[-1])   # 模擬 30 秒
t_eval = t_vec
# --- 模擬 ---
sol = sp.integrate.solve_ivp(turbie.ydot, t_span, y0, t_eval=t_eval, args=(M, C, K, u_of_t, rho_CT_A))

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