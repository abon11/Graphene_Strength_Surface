import numpy as np
import pandas as pd
import local_config
from filter_csv import parse_defects_json

d = {"first": '{"DV": 0.5, "SV": 0.5}', "second": 2}

# print(str(list(parse_defects_json(d["first"]).keys())[0]))
defs = parse_defects_json(d["first"])
print(defs)

string = ''
for defect, size in defs.items():
    string += f'{defect} {size}%, '

print(string[:-2])


