import numpy as np
import scipy.linalg


class _HartreeFock():
    def __init__(self, _molobj, _intobj):
        self.molecule = _molobj
        self.integrals = _intobj
        self.rmsD_threshold = 10**-6
        self.dE_threshold = 10**-6
        self.output = []
    
    def run(self):
        S = self.integrals.get_Overlap_matrix()
        VeN = self.integrals.get_Nuclear_electron_matrix()
        T = self.integrals.get_Kinetic_energy_integral()
        Vee = self.integrals.get_Electron_electron_matrix()
    
        #Core Hamiltonian
        Hcore = VeN+Te
        
        #Diagonalizing overlap matrix
        Lambda_S, L_S = np.linalg.eigh(S)
        #Symmetric orthogonal inverse overlap matrix
        S_sqrt = np.dot(L_S,np.dot(np.diag(np.sqrt(Lambda_S),np.transpose(L_S)))
        
        #Initial Density
        F0prime = np.dot(np.dot(np.matrix.transpose(S_sqrt),Hcore),np.matrix.transpose(S_sqrt))
        eps0, C0prime = diagonlize(F0prime)
        
        C0 = np.matrix.transpose(np.dot(S_sqrt, C0prime))
        
        #Only using occupied MOs
        C0 = C0[0:int(input[0,0]/2)]
        C0T = np.matrix.transpose(C0)
        D0 = np.dot(C0T, C0)        
        
        # Initial Energy
        E0el = 0
        for i in range(0, len(D0)):
            for j in range(0, len(D0[0])):
                E0el += D0[i,j]*(Hcore[i,j]+Hcore[i,j])
        
        #SCF iterations
        if print_SCF == 'Yes':
            print("Iter\t Eel\t")
            """
            output = open('out.txt', 'a')
            output.write('Iter')
            output.write("\t")
            output.write('Eel')
            output.write("\t \t \t \t \t")
            output.write('Etot')
            output.write("\t \t \t \t")
            output.write('dE')
            output.write("\t \t \t \t \t")
            output.write('rmsD')
            if DO_DIIS == 'Yes':
                output.write("\t \t \t \t \t")
                output.write('DIIS')
            output.write("\n")
            output.write('0')
            output.write("\t \t")
            output.write("{:14.10f}".format(E0el))
            output.write("\t \t")
            output.write("{:14.10f}".format(E0el+VNN[0]))
            """
        
        for iter in range(1, Maxiter):
            #New Fock Matrix
            J = np.einsum('pqrs,sr->pq', Vee,D0)
            K = np.einsum('psqr,sr->pq', Vee,D0)
            F = Hcore + 2.0*J-K            
                
            Fprime = np.dot(np.dot(np.transpose(S_sqrt),F),S_sqrt)
            eps, Cprime = diagonlize(Fprime)
            
            C = np.dot(S_sqrt, Cprime)
            
            CT = np.matrix.transpose(C)
            CTocc = CT[0:int(input[0,0]/2)]
            Cocc = np.matrix.transpose(CTocc)
            
            D = np.dot(Cocc, CTocc)
            
            #New SCF Energy
            Eel = 0
            for i in range(0, len(D)):
                for j in range(0, len(D[0])):
                    Eel += D[i,j]*(Hcore[i,j]+F[i,j])

            #Convergance
            dE = Eel - E0el
            rmsD = 0
            for i in range(0, len(D0)):
                for j in range(0, len(D0[0])):
                    rmsD += (D[i,j] - D0[i,j])**2
            rmsD = (rmsD)**0.5
            
            if print_SCF == 'Yes':
                print(iter,"\t","{:14.10f}".format(Eel),"\t","{:14.10f}".format(Eel+VNN[0]),"\t","{: 12.8e}".format(dE),"\t","{: 12.8e}".format(rmsD))
        
            D0 = D
            E0el = Eel
            if np.abs(dE) < deTHR and rmsD < rmsTHR:
                break

        self.E_HF_elec = Eel
        self.C_MO = C
        self.F = F
        self.D = 2*D