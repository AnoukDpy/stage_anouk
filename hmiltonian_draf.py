import matplotlib.pyplot as plt
import numpy as np

def plot_schmidt_modes(*,f, point_number: int, min: float, max:float, n: int, x_unity: str):

    X = np.linspace(min, max, point_number)
    dx = X[1]-X[0]

    Ws, Wi = np.meshgrid(X, X, indexing="ij")

    M_raw = f(Ws, Wi)

    norm = np.sqrt(np.sum(np.abs(M_raw)**2)*(dx**2))

    M = (M_raw/norm)*dx

    U, D, V = np.linalg.svd(M)

    u_n = U[:, n]
    v_n = V[n, :][::-1] #wi and ws must be opposed by the energy conservation...

    norm_u = np.sqrt(np.trapezoid(np.abs(u_n)**2,X))
    norm_v = np.sqrt(np.trapezoid(np.abs(v_n)**2,X))

    u_n = u_n/norm_u
    v_n = v_n/norm_v

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(X, np.real(u_n), color="tab:blue", lw=2)
    ax1.set_title(f"Schmidt mode n = {n}, signal")
    ax1.set_xlabel(rf"$\omega - \omega_p$ [{x_unity}]")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)

    ax2.plot(X, np.real(v_n), color="tab:orange", lw=2)
    ax2.set_title(f"Schmidt mode n = {n}, idler")
    ax2.set_xlabel(rf"$\omega - \omega_p$ [{x_unity}]")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 4))

    lambdas = D**2
    n_indices = np.arange(10)
    lambdas_n = lambdas[:10]

    ax.plot(n_indices, lambdas_n, "o-", color="tab:blue", lw=2, markersize=6)
    ax.set_title(r"Schmidt weights $\lambda_n$ as a function of $n$")
    ax.set_xlabel("Mode index $n$")
    ax.set_ylabel(r"$\lambda_n$")
    ax.set_xticks(n_indices)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()



def ultra_short_pulsed_pumped_ii_spdc_jsa(wo,we):
    pump_spectral_bandwidth = 35
    ord_grp_velocity = 0.061
    extr_grp_velocity = 0.213

    spectral_envelope = np.exp(-(wo+we)**2/pump_spectral_bandwidth**2)
    phase_matching = np.sinc(-(wo*ord_grp_velocity+we*extr_grp_velocity)/(2*np.pi))

    return spectral_envelope*phase_matching

plot_schmidt_modes(f=ultra_short_pulsed_pumped_ii_spdc_jsa,point_number=800,min=-200,max=200,n=2,x_unity=r"ps$^{-1}$")    ax2.set_title(f"Schmidt mode n = {n}, idler")
    ax2.set_xlabel(rf"$\omega - \omega_p$ [{x_unity}]")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 4))

    lambdas = D**2
    n_indices = np.arange(10)
    lambdas_n = lambdas[:10]

    ax.plot(n_indices, lambdas_n, "o-", color="tab:blue", lw=2, markersize=6)
    ax.set_title(r"Schmidt weights $\lambda_n$ as a function of $n$")
    ax.set_xlabel("Mode index $n$")
    ax.set_ylabel(r"$\lambda_n$")
    ax.set_xticks(n_indices)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()



def ultra_short_pulsed_pumped_ii_spdc_jsa(wo,we):
    pump_spectral_bandwidth = 35
    ord_grp_velocity = 0.061
    extr_grp_velocity = 0.213

    spectral_envelope = np.exp(-(wo+we)**2/pump_spectral_bandwidth**2)
    phase_matching = np.sinc(-(wo*ord_grp_velocity+we*extr_grp_velocity)/(2*np.pi))

    return spectral_envelope*phase_matching

plot_schmidt_modes(f=ultra_short_pulsed_pumped_ii_spdc_jsa,point_number=800,min=-200,max=200,n=2,x_unity=r"ps$^{-1}$")
