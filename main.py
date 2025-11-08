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
M, C, K = turbie.build_sys_matrices(turbie_params)

# one file for test
#wind_files_path = ".\inputs\wind_files\wind_TI_0.1\wind_5_ms_TI_0.1.txt"
#df = turbie.load_WSdata(wind_files_path)
#outputs_path = ""

M, C, K = turbie.build_sys_matrices(turbie_params)

# Read files by dir path
wind_files_path = main_dir / "inputs" / "wind_files" / "wind_TI_test_0.1"
output_root = main_dir / "outputs" / "simulation_y"
output_root.mkdir(exist_ok=True)

# initialize statistical summary
df_statistic = pd.DataFrame(columns=["TI-ws_category","TI","U_mean","x1_mean","x1_std","x2_mean","x2_std"])

for txt_file in wind_files_path.rglob("wind_*_ms_TI_*.txt"):
    print(f"Simulating：{txt_file.relative_to(main_dir)}")

    # Catch the file name
    name = txt_file.stem  # e.g. wind_6_ms_TI_0.10
    parts = name.split("_")
    V_nominal = float(parts[1])      # 風速
    TI_val = float(parts[-1])        # TI

    # Load wind speed data from files
    # skip transitory part of simulated wind speed files
    t_trans = 60 # first n seconds
    df = turbie.load_WSdata(txt_file, t_trans)
    u_func, t_vec, u_vec = turbie.build_ws_func(df)
    t_vec = np.array(t_vec)  # 確保是 numpy
    rho_ct_a = turbie.rho_CT_A(df, turbie_params, CT_table)

    # Initialization 初始條件 & 模擬設定
    y0 = np.zeros(4)
    t_span = (t_vec[0], t_vec[-1])
    t_eval = t_vec

    # ODE simulation
    sol = sp.integrate.solve_ivp(turbie.ydot, t_span, y0, t_eval=t_eval, args=(M, C, K, u_func, rho_ct_a))

    # assign value
    t = sol.t
    x1 = sol.y[0]
    x2 = sol.y[1]
    dx1 = sol.y[2]
    dx2 = sol.y[3]

    # Pass output value into files
    relative_folder = txt_file.parent.relative_to(wind_files_path)
    output_folder = output_root / relative_folder
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"response_{name}.txt"

    np.savetxt(output_file, np.column_stack([t, x1, x2, dx1, dx2]),
               header="t\tx1\tx2\tdx1\tdx2", fmt="%.3f",delimiter="\t" )
    print(f"✅ 已輸出結果：{output_file.relative_to(main_dir)}")

    # save the mean, std values, etc.
    df_statistic = pd.concat([df_statistic, pd.DataFrame([{"TI-ws_category":name,
                                                            "TI":TI_val,
                                                            "U_mean":df["V(m/s)"].mean(),
                                                            "x1_mean":x1.mean(),
                                                            "x1_std":x1.std(),
                                                            "x2_mean":x2.mean(),
                                                            "x2_std":x2.std()}])]) 

    plt.figure(figsize=(8,5))
    plt.subplot(3,1,1)
    plt.plot(t, u_vec)
    plt.ylabel("Wind [m/s]")
    plt.title(name)
    plt.subplot(3,1,2)
    plt.plot(t, x1, label="Blade")
    plt.ylabel("x1 [m]")
    plt.subplot(3,1,3)
    plt.plot(t, x2, label="Tower", color='orange')
    plt.ylabel("x2 [m]")
    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(output_folder / f"time_series_{name}.png", dpi=200)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(t, x1, label="Blade Deflec. (x1)", color="tab:cyan")
    ax1.plot(t, x2, label="Tower Deflec. (x2)", color="tab:blue")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Displacement [m]")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(t, u_vec, label="Wind (u)", color="tab:red", alpha=0.5)
    ax2.set_ylabel("Wind [m/s]", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.legend(loc="upper right")

    plt.title(name)
    plt.tight_layout()
    plt.savefig(output_folder / f"time_series_{name}.png", dpi=200)
    plt.close()
                                           
    
print(df_statistic)
"""
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
plt.show()"""