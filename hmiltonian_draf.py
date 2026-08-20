import matplotlib.pyplot as plt
import numpy as np

def plot_functions(*,f, point_number: int, min: float, max:float, n: int):

    X = np.linspace(min, max, point_number)

    Ws, Wi = np.meshgrid(X, X, indexing="ij")

    M = f(Ws,Wi)

    U, D, V = np.linalg.svd(M)

    u_n = U[:, n]
    v_n = V[n, :]

    plt.figure(figsize=(8, 4))
    plt.plot(X, np.abs(u_n)**2, label=rf"Signal")
    plt.plot(X, np.abs(v_n)**2, label=rf"Idler")
    plt.xlabel(r"$\omega - \omega_0$ [Trad/s]")
    plt.ylabel("Densité spectrale")
    plt.legend()
    plt.grid(True)
    plt.show()

def jsa_gaussienne(ws, wi):

    return np.exp(-((ws + wi) ** 2) / (2 * 3.0**2)) * np.exp(-((ws - wi) ** 2) / (2 * 1.0**2))

#plot_functions(f=jsa_gaussienne, point_number=400, min=-8.0, max=8.0, n=1)

def jsa_resonator(ws, wi, gamma=0.5):
    denom_p = -((ws + wi) / 2) + 1j * gamma / 2
    denom_s = ws + 1j * gamma / 2
    denom_i = wi + 1j * gamma / 2
    return 1.0 / (denom_p * denom_s * denom_i)
    return 1.0 / (denom_p * denom_s * denom_i)

#plot_functions(f=jsa_resonator, point_number=400, min=-15.0, max=15.0, n=10)



def jsa_cw_comb(ws, wi, fsr=21.0, r=0.96, sigma_p=0.8):
    # ws, wi en GHz
    # sigma_p < FSR garantit que chaque mode n sélectionne une seule résonance

    # Conservation d'énergie stricte (pompe étroite)
    pump = np.exp(-((ws + wi) ** 2) / (2 * sigma_p**2))

    # Résonances du résonateur
    cav_s = 1.0 / (1.0 - r * np.exp(1j * 2 * np.pi * ws / fsr))
    cav_i = 1.0 / (1.0 - r * np.exp(1j * 2 * np.pi * wi / fsr))

    return pump * cav_s * cav_i

#plot_functions(f=jsa_cw_comb, point_number=400, min=-50.0, max=50.0, n=2)


def jsa_all_lorentzian(ws, wi, fsr=21.0, r=0.96):
    # ws, wi en GHz
    # 1. Pompe intracavité lorentzienne périodique (filtrage de pompe centré sur ws + wi = 0)
    cav_p = 1.0 / (1.0 - r * np.exp(1j * 2 * np.pi * ((ws + wi) / 2) / fsr))

    # 2. Résonances lorentziennes périodiques Signal et Idler
    cav_s = 1.0 / (1.0 - r * np.exp(1j * 2 * np.pi * ws / fsr))
    cav_i = 1.0 / (1.0 - r * np.exp(1j * 2 * np.pi * wi / fsr))

    return cav_p * cav_s * cav_i

#plot_functions(f=jsa_all_lorentzian,point_number=400,min=-50.0,max=50.0,n=2)


def jsa_bins(ws, wi, fsr=21.0, gamma=0.5):
    r = 1.0 - (np.pi * gamma / fsr)
    # Infime dispersion pour lever la dégénérescence numérique (+21 vs -21 GHz)
    eps = 1e-5

    t_p = 1.0 / (1.0 - r * np.exp(1j * 2 * np.pi * ((ws + wi) / 2) / fsr))
    t_s = 1.0 / (1.0 - r * np.exp(1j * 2 * np.pi * ws / (fsr * (1 + eps))))
    t_i = 1.0 / (1.0 - r * np.exp(1j * 2 * np.pi * wi / (fsr * (1 - eps))))

    return t_p * t_s * t_i


#plot_functions(f=jsa_bins,point_number=800,min=-50.0,max=50.0,n=2,)

def jsa_bins_clean(ws, wi, fsr=21.0, gamma=0.5):
    r = 1.0 - (np.pi * gamma / fsr)
    eps = 1e-4

    t_p = 1.0 / (-(ws + wi) / 2 + 1j * (gamma / 2))
    t_s = 1.0 / (1.0 - r * np.exp(1j * 2 * np.pi * ws / (fsr * (1 + eps))))
    t_i = 1.0 / (1.0 - r * np.exp(1j * 2 * np.pi * wi / (fsr * (1 - eps))))

    return t_p * t_s * t_i


plot_functions(f=jsa_bins_clean,point_number=800,min=-50.0,max=50.0,n=2)




















