import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

class TB_DMFT:
    def __init__(self, lattice, interaction_params, nk_points):
        self.lattice = lattice #doesn't do anything yet
        self.U = interaction_params['U']
        self.J = interaction_params.get('J', 0) #doesn't do anything yet
        self.nk_points = nk_points
        self.k_points = self.generate_k_points()
        self.local_gf = None
        self.self_energy = None

    def generate_k_points(self):
        return np.linspace(-np.pi, np.pi, self.nk_points)

    def dft_hamiltonian(self, k):
        #tb model for a 1D lattice
        t = 1  # hopping term
        h_k = -2 * t * np.cos(k)
        return np.array([[h_k]])  

    def solve_dmft_loop(self, max_iter=10, tol=1e-6):
        #initial \sigma=0 guess
        self.self_energy = np.zeros(1, dtype=complex)

        for iteration in range(max_iter):
            print(f"Iteration {iteration + 1}")

            #solve lattice problem get Green's function
            lattice_gf = self.solve_lattice_gf()

            #get local Green's function
            self.local_gf = self.extract_local_gf(lattice_gf)

            #solve the impurity problem (placeholder for now)
            new_self_energy = self.solve_impurity_problem()

            #check convergence
            error = np.linalg.norm(new_self_energy - self.self_energy)
            print(f"Self-energy norm mismatch: {error}")
            if error < tol:
                print("DMFT cycle converged!")
                break

            #update self-energy
            self.self_energy = new_self_energy

        self.compute_spectral_function()

    def solve_lattice_gf(self):
        #lattice Green's function G(k, i$\omega$)
        nk = len(self.k_points)
        omega = 1j * np.linspace(-10, 10, 1000)  #Matsubara freqs
        lattice_gf = np.zeros((nk, len(omega), 1, 1), dtype=complex)  #adjust dimensions

        for i, k in enumerate(self.k_points):
            h_k = self.dft_hamiltonian(k)
            for j, w in enumerate(omega):
                #consistent dimensions
                self_energy_term = np.eye(h_k.shape[0]) * self.self_energy[0]
                gf_k = np.linalg.inv((w * np.eye(h_k.shape[0])) - (h_k + self_energy_term))
                lattice_gf[i, j] = gf_k

        return lattice_gf

    def extract_local_gf(self, lattice_gf):
        """Extract the local Green's function."""
        return np.mean(lattice_gf, axis=0) #avg over k-points

    def solve_impurity_problem(self):
        """Solve the impurity problem (placeholder)."""
        omega = 1j * np.linspace(-10, 10, 1000)  #Matsubara freqs
        delta = self.local_gf[:, 0, 0] - (1 / omega)  
        impurity_gf = 1 / (omega - self.U + delta)
        return self.U * impurity_gf

    def compute_spectral_function(self):
        """Compute and plot the spectral function A($\omega$)."""
        omega = np.linspace(-10, 10, 1000)  #real freqs for spectral function
        spectral_function = -1 / np.pi * np.imag(1 / (omega + 1j * 0.01 - self.self_energy[0]))

        plt.figure(figsize=(10, 6))
        plt.plot(omega, spectral_function, label="Spectral Function A($\omega$)")
        plt.xlabel("$\omega$ (Frequency)")
        plt.ylabel("A($\omega$)")
        plt.title("Spectral Function")
        plt.legend()
        plt.grid()
        plt.savefig("SpectralFunction_2.pdf", format="pdf", bbox_inches="tight")
        #plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(omega, np.real(self.self_energy[0]) * np.ones_like(omega), label="Re($\Sigma$)")
        plt.plot(omega, np.imag(self.self_energy[0]) * np.ones_like(omega), label="Im($\Sigma$)")
        plt.xlabel("$\omega$ (Frequency)")
        plt.ylabel("$\Sigma$ (Self-Energy)")
        plt.title("Self-Energy")
        plt.legend()
        plt.grid()
        plt.savefig("SelfEnergy_2.pdf", format="pdf", bbox_inches="tight")
        #plt.show()

        omega_matsubara = 1j * np.linspace(-10, 10, 1000)  # Matsubara frequencies
        plt.figure(figsize=(10, 6))
        plt.plot(omega_matsubara.imag, np.real(self.local_gf[:, 0, 0]), label="Re($G_{loc}$)")
        plt.plot(omega_matsubara.imag, np.imag(self.local_gf[:, 0, 0]), label="Im($G_{loc}$)")
        plt.xlabel("i$\omega_n$ (Matsubara Frequency)")
        plt.ylabel("$G_{loc}$")
        plt.title("Local Green's Function")
        plt.legend()
        plt.grid()
        plt.savefig("LocalGF_2.pdf", format="pdf", bbox_inches="tight")
        #plt.show()

if __name__ == "__main__":
    lattice = '1D'
    interaction_params = {'U': 8.0, 'J': 0.0}
    nk_points = 100

    solver = TB_DMFT(lattice, interaction_params, nk_points)
    solver.solve_dmft_loop()
