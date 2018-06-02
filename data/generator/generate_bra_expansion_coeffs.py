import numpy as np


def write_bra_expansion_coeffs(max_angular):
    S_file = open("slowquant/molecularintegrals/bra_expansion_coeffs.py", "w+")
    S_file.write("import numpy as np\n")
    S_file.write("from numba import jit, float64\n")
    S_file.write("from slowquant.molecularintegrals.expansion_coefficients import *\n")
    
    for lb in range(max_angular+1):
        for la in range(lb, max_angular+1):
            S_file.write("def bra_expansion_coeffs(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, E_buff_1, bra_array):\n")
            S_