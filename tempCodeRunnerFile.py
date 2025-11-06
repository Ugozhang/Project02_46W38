import numpy as np
import scipy as sp
import pandas as pd

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

def turbine_area(r):
    return r**2*np.pi


def CT_interp(V,CT_table):
    """
    Pass wind speed as float, CT_table as DataFrame return interpolation of CT
    """
    return np.interp(V,CT_table["V"],CT_table["CT"])
    # return sp.interpolation.interp1d(CT_table["V"],CT_table["CT"],kind='linear')

def build_sys_matrices(turbine_p):
    M = np.array([[ 3*turbine_p["mb"], 0],
                 [ 0, turbine_p["mn"] + turbine_p["mt"] + turbine_p["mh"]]])
    C = np.array([[ turbine_p["c1"], -turbine_p["c1"]],
                  [ -turbine_p["c1"], turbine_p["c1"] + turbine_p["c2"]]])
    K = np.array([[ turbine_p["k1"], -turbine_p["k1"]],
                  [ -turbine_p["k1"], turbine_p["k1"] + turbine_p["k2"]]])
    return M, C, K