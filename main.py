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
wind_files_path = main_dir / "inputs" / "wind_files"
output_root = main_dir / "outputs"
output_root.mkdir(exist_ok=True)

for txt_file in wind_files_path.rglob("wind_*_ms_TI_*.txt"):
    print(f"Simulating：{txt_file.relative_to(main_dir)}")

    # 抓風檔名稱資訊
    name = txt_file.stem  # e.g. wind_6_ms_TI_0.10
    parts = name.split("_")
    V_nominal = float(parts[1])      # 風速
    TI_val = float(parts[-1])        # TI

    # 讀風檔
    df = turbie.load_WSdata(txt_file)
    u_func, t_vec, u_vec = turbie.build_ws_func(df)
    t_vec = np.array(t_vec)  # 確保是 numpy
    rho_ct_a = turbie.rho_CT_A(df, turbie_params, CT_table)

    # 初始條件 & 模擬設定
    y0 = np.zeros(4)
    t_span = (t_vec[0], t_vec[-1])
    t_eval = t_vec

    # 數值積分
    sol = sp.integrate.solve_ivp(turbie.ydot, t_span, y0, t_eval=t_eval, args=(M, C, K, u_func, rho_ct_a))

    t = sol.t
    x1 = sol.y[0]
    x2 = sol.y[1]

    # === 5️⃣ 輸出時間序列 ===
    relative_folder = txt_file.parent.relative_to(wind_files_path)
    output_folder = output_root / relative_folder
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"response_{name}.txt"

    np.savetxt(output_file, np.column_stack([t, u_vec, x1, x2]),
               header="time_s wind_ms x1_blade_m x2_tower_m")
    print(f"✅ 已輸出結果：{output_file.relative_to(main_dir)}")

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