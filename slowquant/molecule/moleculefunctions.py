import numpy as np
from numba import jit, float64, int64
from numba.types import Tuple

def atom_to_numbers(atom_name, number_property):
    name2number = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16,
            "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31,
            "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46,
            "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61,
            "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76,
            "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91,
            "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100, "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg":
            106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118}

    number2name = {1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S",
            17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga", 32:
            "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47:
            "Ag", 48: "Cd", 49: "In", 50: "Sn", 51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm", 62:
            "Sm", 63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb", 71: "Lu", 72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77:
            "Ir", 78: "Pt", 79: "Au", 80: "Hg", 81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At", 86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th", 91: "Pa", 92:
            "U", 93: "Np", 94: "Pu", 95: "Am", 96: "Cm", 97: "Bk", 98: "Cf", 99: "Es", 100: "Fm", 101: "Md", 102: "No", 103: "Lr", 104: "Rf", 105: "Db", 106: "Sg",
            107: "Bh", 108: "Hs", 109: "Mt", 110: "Ds", 111: "Rg", 112: "Cn", 113: "Nh", 114: "Fl", 115: "Mc", 116: "Lv", 117: "Ts", 118: "Og"}

    vdw_radii = {1: 2.26767118629, 2: 2.64561638401, 3: 3.43930129921, 4: 2.89128076253, 5: 3.62827389807, 6: 3.21253418058, 7: 2.9290752823, 8: 2.87238350264, 9:
            2.77789720321, 10: 2.91017802241, 11: 4.28967799407, 12: 3.26922596024, 13: 3.47709581899, 14: 3.96842457602, 15: 3.40150677944, 16: 3.40150677944, 17:
            3.30702048001, 18: 3.55268485853, 19: 5.19674646859, 20: 4.36526703362, 21: 3.9873218359, 22: False, 23: False, 24: False, 25: False, 26: False, 27: False, 28:
            3.08025336138, 29: 2.64561638401, 30: 2.62671912412, 31: 3.53378759864, 32: 3.9873218359, 33: 3.49599307887, 34: 3.5904793783, 35: 3.49599307887, 36:
            3.81724649693, 37: 5.72586974539, 38: 4.70541771156, 39: False, 40: False, 41: False, 42: False, 43: False, 44: False, 45: False, 46: 3.08025336138, 47:
            3.25032870036, 48: 2.98576706195, 49: 3.64717115796, 50: 4.10070539522, 51: 3.89283553647, 52: 3.89283553647, 53: 3.74165745739, 54: 4.08180813533, 55:
            6.48176014083, 56: 5.06446564939, 57: False, 58: False, 59: False, 60: False, 61: False, 62: False, 63: False, 64: False, 65: False, 66: False, 67: False, 68: False, 69: False,
            70: False, 71: False, 72: False, 73: False, 74: False, 75: False, 76: False, 77: False, 78: 3.30702048001, 79: 3.13694514104, 80: 2.9290752823, 81: 3.70386293761, 82:
            3.81724649693, 83: 3.91173279636, 84: 3.7227601975, 85: 3.81724649693, 86: 4.15739717487, 87: 6.57624644025, 88: 5.34792454768, 89: False, 90: False, 91:
            False, 92: 3.51489033876, 93: False, 94: False, 95: False, 96: False, 97: False, 98: False, 99: False, 100: False, 101: False, 102: False, 103: False, 104: False, 105: False, 106:
            False, 107: False, 108: False, 109: False, 110: False, 111: False, 112: False, 113: False, 114: False, 115: False, 116: False, 117: False, 118: False}
    
    mass = {"H": 1.008, "Na": 22.989, "Sc": 44.955, "Ga": 69.723, "Nb": 92.906, "Sb": 121.76, "Pm": False, "Lu": 174.9668, "Tl": 204.38, "Pa": 231.035, "Md": False, "Rg": False, 
            "He": 4.002, "Mg": 24.305, "Ti": 47.867, "Ge": 72.63, "Mo": 95.95, "Te": 127.6, "Sm": 150.36, "Hf": 178.49, "Pb": 207.2, "U": 238.028, "No": False, "Cn": False, 
            "Li": 6.94, "Al": 26.981, "V": 50.9415, "As": 74.921, "Tc": False, "I": 126.904, "Eu": 151.964, "Ta": 180.947, "Bi": 208.98, "Np": False, "Lr": False, "Nh": False, 
            "Be": 9.012, "Si": 28.085, "Cr": 51.9961, "Se": 78.971, "Ru": 101.07, "Xe": 131.293, "Gd": 157.25, "W": 183.84, "Po": False, "Pu": False, "Rf": False, "Fl": False, 
            "B": 10.81, "P": 30.973, "Mn": 54.938, "Br": 79.904, "Rh": 102.905, "Cs": 132.905, "Tb": 158.925, "Re": 186.207, "At": False, "Am": False, "Db": False, "Mc": False, 
            "C": 12.011, "S": 32.06, "Fe": 55.845, "Kr": 83.798, "Pd": 106.42, "Ba": 137.327, "Dy": 162.5, "Os": 190.23, "Rn": False, "Cm": False, "Sg": False, "Lv": False, 
            "N": 14.007, "Cl": 35.45, "Co": 58.933, "Rb": 85.4678, "Ag": 107.8682, "La": 138.905, "Ho": 164.93, "Ir": 192.217, "Fr": False, "Bk": False, "Bh": False, "Ts": False, 
            "O": 15.999, "Ar": 39.948, "Ni": 58.6934, "Sr": 87.62, "Cd": 112.414, "Ce": 140.116, "Er": 167.259, "Pt": 195.084, "Ra": False, "Cf": False, "Hs": False, "Og": False, 
            "F": 18.998, "K": 39.0983, "Cu": 63.546, "Y": 88.905, "In": 114.818, "Pr": 140.907, "Tm": 168.934, "Au": 196.966, "Ac": False, "Es": False, "Mt": False, 
            "Ne": 20.1797, "Ca": 40.078, "Zn": 65.38, "Zr": 91.224, "Sn": 118.71, "Nd": 144.242, "Yb": 173.045, "Hg": 200.592, "Th": 232.0377, "Fm": False, "Ds": False}
    
    if number_property.lower() == "charge":
        return name2number[atom_name]
    elif number_property.lower() == "mass":
        return mass[atom_name]
    elif number_property.lower() == "vdw_radii":
        return vdw_radii[name2number[atom_name]]
  

@jit(float64[:,:](float64[:,:]),nopython=True,cache=True)
def calc_distance_matrix(xyz):
    distance_matrix = np.zeros((len(xyz),len(xyz)))
    for i in range(0, len(xyz)):
        for j in range(i, len(xyz)):
            distance_matrix[i,j] = distance_matrix[i,j] = ((xyz[i,0]-xyz[j,0])**2.0+(xyz[i,1]-xyz[j,1])**2.0+(xyz[i,2]-xyz[j,2])**2.0)**0.5
    return distance_matrix


@jit(float64[:](float64[:],float64[:,:]),nopython=True,cache=True)
def calc_center_of_mass(mass, xyz):
    Xcm = np.sum(mass*xyz[:,0])/np.sum(mass)
    Ycm = np.sum(mass*xyz[:,1])/np.sum(mass)
    Zcm = np.sum(mass*xyz[:,2])/np.sum(mass)
    return np.array([Xcm, Ycm, Zcm])
    
    
@jit(float64[:](float64[:],float64[:,:]),nopython=True,cache=True)
def calc_center_of_nuclear_charge(charge, xyz):
    Xcm = np.sum(charge*xyz[:,0])/np.sum(charge)
    Ycm = np.sum(charge*xyz[:,1])/np.sum(charge)
    Zcm = np.sum(charge*xyz[:,2])/np.sum(charge)
    return np.array([Xcm, Xcm, Zcm])


@jit(float64(float64),nopython=True,cache=True)
def factorial2(n):
    """
    Calculate the factorial2 of an integer.
    """
    n_range = int(n)
    out = 1.0
    if n > 0:
        for i in range(0, int(n_range+1)//2):
            out = out*(n-2*i)
    return out


@jit(float64[:](float64, float64, float64, float64[:]),nopython=True,cache=True)
def Normalization(l, m, n, zeta_exp):
    # CHANGE SO IT CONTAINT SHELLS AND NOT BASISFUNCTIONS.
    """
    Calculates the normalizations coefficients of the basisfunctions.
    """
    N = np.zeros(len(zeta_exp))
    pi = 3.141592653589793238462643383279
    # Normalize primitive functions
    for i in range(len(zeta_exp)):
        c = zeta_exp[i]

        part1 = (2.0/pi)**(3.0/4.0)
        part2 = 2.0**(l+m+n) * c**((2.0*l+2.0*m+2.0*n+3.0)/(4.0))
        part3 = (factorial2(int(2*l-1))*factorial2(int(2*m-1))*factorial2(int(2*n-1)))**0.5
        N[i] = part1 * ((part2)/(part3))
    return N