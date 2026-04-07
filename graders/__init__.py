def _norm(val, min_r=-0.5, max_r=1.0):
    return max(0.0, min(1.0, (val - min_r) / (max_r - min_r)))

def name_match(name1, name2):
    return str(name1).strip().lower() == str(name2).strip().lower()

def arg_present(args, key):
    return key in args and args[key] not in (None, "")
