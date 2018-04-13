import numpy as np
import os
from slowquant.molecule.moleculefunctions import atom_to_numbers, calc_distance_matrix, calc_center_of_mass, calc_center_of_nuclear_charge, factorial2, Normalization

class _Molecule:
    """
    Molecule class
    """
    def __init__(self, molecule_file, unit="angstrom"):
        if unit.lower() == "angstrom":
            unit_factor = 1.889725989
        elif unit.lower() == "au" or unit.lower() == "bohr":
            unit_factor = 1.0
        else:
            print("Unit not recongnized")
            return False
        
        if ".xyz" in molecule_file:
            file = open(molecule_file,"r").readlines()
            self._atom_list = [False]*(len(file) - 2)
            self._center_of_mass = False
            self._center_of_nuclear_charge = False
            for i in range(2, len(file)):
                self._atom_list[i-2] = Atom(file[i].split()[0], 
                                            float(file[i].split()[1])*unit_factor, 
                                            float(file[i].split()[2])*unit_factor, 
                                            float(file[i].split()[3])*unit_factor,
                                            atom_to_numbers(file[i].split()[0], "charge"),
                                            atom_to_numbers(file[i].split()[0], "mass"))
        self.this_file_location = os.path.dirname(os.path.abspath(__file__))
    
    def get_atom_name(self, index):
        return self._atom_list[index].atom_name
    
    def get_atom_coord(self, index):
            return np.array([self._atom_list[index].x_coord, 
                            self._atom_list[index].y_coord, 
                            self._atom_list[index].z_coord])
        
    def get_number_atoms(self):
        return len(self._atom_list)
    
    def get_atom_charge(self, index):
        return self._atom_list[index].nuclear_charge
    
    def get_atom_mass(self, index):
        return self._atom_list[index].atomic_mass
        
    def get_molecule(self):
        molecule = []
        for i in range(self.get_number_atoms()):
            molecule.append([self.get_atom_name(i), 
                             self.get_atom_coord(i)[0], 
                             self.get_atom_coord(i)[1], 
                             self.get_atom_coord(i)[2]])
        return molecule
    
    def get_distance_matrix(self):
        xyz = np.array(self.get_molecule())
        return calc_distance_matrix(xyz[:,1:].astype(np.float64))
    
    def get_center_of_mass(self):
        if self._center_of_mass == False:
            molecule = np.zeros((self.get_number_atoms(), 4))
            for i in range(self.get_number_atoms()):
                molecule[i,0] = self.get_atom_charge(i) 
                molecule[i,1:] = self.get_atom_coord(i)
            self._center_of_mass = calc_center_of_mass(molecule[:,0], molecule[:,1:])
        return self._center_of_mass
    
    def get_center_of_nuclear_charge(self):
        if self._center_of_nuclear_charge == False:
            molecule = np.zeros((self.get_number_atoms(), 4))
            for i in range(self.get_number_atoms()):
                molecule[i,0] = self.get_atom_charge(i) 
                molecule[i,1:] = self.get_atom_coord(i)
            self._center_of_nuclear_charge = calc_center_of_nuclear_charge(molecule[:,0], molecule[:,1:])
        return self._center_of_nuclear_charge
    
    def set_basis_set(self, basisname):
        self._basis_function_list = []
        for i in range(self.get_number_atoms()):
            basisfile = np.genfromtxt(self.this_file_location+"/"+str(basisname)+'.csv', dtype=str, delimiter=',')
            make_bf_check = 0
            exp = []
            for line in basisfile:
                if line[1] == self._atom_list[i].atom_name:
                    make_bf_check = 1
                elif line[0] != '' and line[0] != "FOR" and make_bf_check == 1:
                    if exp != []:
                        self.__append_basis_function(self.get_atom_coord(i),
                                                     i,
                                                     np.array(exp),
                                                     np.array(coeffs),
                                                     bf_type)
                    exp = []
                    coeffs = []
                    exp.append(float(line[1]))
                    coeffs.append(float(line[2]))
                    bf_type = line[0]
                elif line[0] == '' and make_bf_check == 1:
                    exp.append(float(line[1]))
                    coeffs.append(float(line[2]))
                elif line[0] == "FOR" and make_bf_check == 1:
                    self.__append_basis_function(self.get_atom_coord(i),
                                                 i,
                                                 np.array(exp),
                                                 np.array(coeffs),
                                                 bf_type)
                    break
                                  
    def __append_basis_function(self, xyz, atom_idx, exp, contract_coeffs, bf_type):
        if bf_type == "S":
            self._basis_function_list.append(basis_function(xyz, atom_idx, exp, contract_coeffs, np.array([0, 0, 0],dtype=int)))
        elif bf_type == "P":
            angulars = np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=int)
            for angular in angulars:
                self._basis_function_list.append(basis_function(xyz, atom_idx, exp, contract_coeffs, angular))
        elif bf_type == "D":
            angulars = np.array([[2,0,0],[1,1,0],[1,0,1],[0,2,0],[0,1,1],[0,0,2]],dtype=int)
            for angular in angulars:
                self._basis_function_list.append(basis_function(xyz, atom_idx, exp, contract_coeffs, angular))
                            

    
class Atom:
    def __init__(self, name, x, y, z, charge, mass):
        self.atom_name = name
        self.x_coord = x
        self.y_coord = y
        self.z_coord = z
        self.nuclear_charge = charge
        self.atomic_mass = mass
        
        
class basis_function:
    def __init__(self, xyz, atom_idx, exp, contract_coeffs, ang_xyz):
        self.coord = xyz
        self.atom_idx = atom_idx
        self.exponents = exp
        self.contraction_coeffs = contract_coeffs
        self.angular_moment = ang_xyz
        self.normalization = Normalization(self.angular_moment[0],self.angular_moment[1],self.angular_moment[2],self.exponents)
