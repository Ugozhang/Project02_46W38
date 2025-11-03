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
    
f_path = "C:\DTU_prog\Project02_46W38\inputs\turbie_inputs\turbie_parameters.txt"

print(load_turbine_prop(f_path))