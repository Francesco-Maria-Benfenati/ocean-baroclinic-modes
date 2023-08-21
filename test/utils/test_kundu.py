import numpy as np
import matplotlib.pyplot as plt

try:
    from ...src.model.baroclinicmodes import BaroclinicModes
    from ...src.model.interpolation import Interpolation
except ImportError:
    import sys, os

    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    from src.model.baroclinicmodes import BaroclinicModes
    from src.model.interpolation import Interpolation

if __name__ == "__main__":
    depth_kundu = -np.array(
        [
            0,
            3,
            5,
            6,
            8,
            10,
            12,
            13,
            15,
            20,
            25,
            30,
            35,
            40,
            45,
            50,
            55,
            60,
            65,
            70,
            75,
            80,
            85,
            90,
            95,
            100,
        ]
    )

    N_carnation_kundu = np.array(
        [
            0.135,
            0.18,
            0.25,
            0.47,
            0.5,
            0.47,
            0.40,
            0.36,
            0.3,
            0.22,
            0.185,
            0.165,
            0.145,
            0.13,
            0.12,
            0.105,
            0.095,
            0.085,
            0.077,
            0.069,
            0.067,
            0.06,
            0.05,
            0.045,
            0.0425,
            0.04,
        ]
    )
    N_db7_kundu = np.array(
        [
            0.1,
            0.12,
            0.15,
            0.17,
            0.24,
            0.32,
            0.36,
            0.37,
            0.34,
            0.25,
            0.185,
            0.15,
            0.135,
            0.125,
            0.11,
            0.105,
            0.095,
            0.085,
            0.077,
            0.069,
            0.067,
            0.06,
            0.055,
            0.055,
            0.05,
            0.045,
        ]
    )

    """
    Plot comparison with Kundu, Allen, Smith (1975).
    """

    # Convert BV freq from 1/2pi cpm to 1/s
    N_carnation_kundu *= (2 * np.pi) / 60
    N_db7_kundu *= (2 * np.pi) / 60
    # Problem parameters
    mean_depth = 100
    depth = -np.arange(0, mean_depth + 1, 1)
    coriolis_param = 1e-04
    dz = 1  # grid step [m]
    # Interpolate Brunt-Vaisala freq at depth levels
    interpolation = Interpolation(-depth_kundu, N_carnation_kundu, N_db7_kundu)
    N_carnation, N_db7 = interpolation.apply_interpolation(0, mean_depth + 1, dz)
    # interpolate at interfaces staggered grid
    z_0 = np.abs(depth_kundu[0]) - dz / 2
    z_N = mean_depth + dz
    interp_N_carn, interp_N_db7 = interpolation.apply_interpolation(z_0, z_N, dz)
    # Compute Eigenvals/eigenvecs
    s_param_carn = (interp_N_carn**2) / (coriolis_param**2)
    s_param_db7 = (interp_N_db7**2) / (coriolis_param**2)
    eigenvals_carn, structfunc_carn = BaroclinicModes.compute_baroclinicmodes(
        s_param_carn, dz
    )
    eigenvals_db7, structfunc_db7 = BaroclinicModes.compute_baroclinicmodes(
        s_param_db7, dz
    )
    print(f"Eigenvalues at Carnation [km^-1]: {np.sqrt(eigenvals_carn)*1000}")
    print(f"Eigenvalues at DB-7 [km^-1]: {np.sqrt(eigenvals_db7)*1000}")

    # PLOT Brunt-Vaisala frequency
    fig, ax = plt.subplots(figsize=(7, 8))
    plt.rcParams["figure.figsize"] = [7.00, 8]
    plt.rcParams["figure.autolayout"] = True
    im = plt.imread("test/kundu_BVfreq.png")
    im = ax.imshow(im, extent=[0, 100, -100, 0])
    ax.grid(visible=True)
    ax.set_title(
        "Vertical profiles of Brunt-Vaisala frequency,\n from Kundu, Allen, Smith (1975)",
        pad=20,
        fontsize=16,
    )
    ax.plot(np.nan, 0, "k")
    ax.plot(np.nan, 0, "k--")
    ax.plot(N_carnation / np.max(N_carnation) * 100, depth, "r")
    ax.plot(N_db7 / np.max(N_carnation) * 100, depth, "b--")
    ax.legend(
        [
            "Carnation from Kundu (1975)",
            "DB-7 from Kundu (1975)",
            "Carnation replica",
            "DB-7 replica",
        ]
    )
    ax.set_xlabel("N (cycles/s)", labelpad=15, fontsize=14)
    ax.set_ylabel("DEPTH (meters)", fontsize=14)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_yticks(np.linspace(-100, 0, 11))
    ax.set_xticks(np.linspace(0, 100, 11))
    ax.set_xticklabels(
        ["0", None, "0.01", None, "0.02", None, "0.03", None, "0.04", None, "0.05"]
    )
    ax.yaxis.set_tick_params(width=1, length=7)
    ax.xaxis.set_tick_params(width=1, length=7)
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")
    ax.set_yticklabels(
        ["-100", None, "-80", None, "-60", None, "-40", None, "-20", None, "0"]
    )
    ax.set_ylim(-100, 0)
    ax.set_xlim(0, 105)
    plt.show()
    plt.close()

    # PLOT BAROCLINIC MODES CARNATION
    fig1, ax1 = plt.subplots(figsize=(7, 8))
    im1 = plt.imread("test/kundu_modes.png")
    im1 = ax1.imshow(im1, extent=[-50, 50, -100, 0])
    ax1.grid(visible=True)
    ax1.plot(np.nan, 0, "k")
    ax1.plot(structfunc_carn[:, :3] / 3 * 50, depth, "r")
    ax1.plot(structfunc_carn[:, 3] / 3 * 50, depth, "r--")
    ax1.set_xlabel(
        "NORMALIZED MODE AMPLITUDE AT CARNATION,\n from Kundu, Allen, Smith (1975)",
        labelpad=15,
        fontsize=14,
    )
    ax1.set_yticks(np.linspace(-100, 0, 11))
    ax1.set_yticklabels(
        ["-100", None, "-80", None, "-60", None, "-40", None, "-20", None, "0"]
    )
    ax1.set_xticks(np.linspace(-50, 50, 7))
    ax1.set_xticklabels(["-3", "-2", "1", "0", "1", "2", "3"])
    ax1.yaxis.set_tick_params(width=1, length=7)
    ax1.xaxis.set_tick_params(width=1, length=7)
    ax1.tick_params(axis="y", direction="in")
    ax1.tick_params(axis="x", direction="in")
    ax1.set_ylabel("DEPTH (m)", fontsize=14)
    ax1.set_xlim(-55, 55)
    ax1.set_ylim(-100, 0)
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position("top")
    ax1.spines["left"].set_position("center")
    ax1.spines["right"].set_color("none")
    ax1.legend(["Kundu (1975)", "numerical results REPLICA"])
    plt.show()
    plt.close()

    # PLOT BAROCLINIC MODES DB-7
    fig2, ax2 = plt.subplots(figsize=(7, 8))
    ax2.grid(visible=True)
    ax2.plot(np.nan, 0, "r")
    ax2.plot(np.nan, 0, "b")
    ax2.plot(structfunc_carn[:, :3] / 3 * 50, depth, "r")
    ax2.plot(structfunc_db7[:, :3] / 3 * 50, depth, "b")
    ax2.plot(structfunc_carn[:, 3] / 3 * 50, depth, "r--")
    ax2.plot(structfunc_db7[:, 3] / 3 * 50, depth, "b--")
    ax2.set_xlabel(
        "NORMALIZED MODE AMPLITUDE AT DB-7,\n from Kundu, Allen, Smith (1975)",
        labelpad=15,
        fontsize=14,
    )
    ax2.set_yticks(np.linspace(-100, 0, 11))
    ax2.set_yticklabels(
        ["-100", None, "-80", None, "-60", None, "-40", None, "-20", None, "0"]
    )
    ax2.set_xticks(np.linspace(-50, 50, 7))
    ax2.set_xticklabels(["-3", "-2", "1", "0", "1", "2", "3"])
    ax2.yaxis.set_tick_params(width=1, length=7)
    ax2.xaxis.set_tick_params(width=1, length=7)
    ax2.tick_params(axis="y", direction="in")
    ax2.tick_params(axis="x", direction="in")
    ax2.set_ylabel("DEPTH (m)", fontsize=14)
    ax2.set_xlim(-55, 55)
    ax2.set_ylim(-100, 0)
    ax2.xaxis.tick_top()
    ax2.xaxis.set_label_position("top")
    ax2.spines["left"].set_position("center")
    ax2.spines["right"].set_color("none")
    ax2.legend(["CARNATION", "DB-7"])
    plt.figtext(
        0.67,
        0.3,
        "1",
        bbox={"boxstyle": "circle", "facecolor": "None", "edgecolor": "black"},
    )
    plt.figtext(
        0.54,
        0.3,
        "3",
        bbox={"boxstyle": "circle", "facecolor": "None", "edgecolor": "black"},
    )
    plt.figtext(
        0.43,
        0.3,
        "4",
        bbox={"boxstyle": "circle", "facecolor": "None", "edgecolor": "black"},
    )
    plt.figtext(
        0.37,
        0.3,
        "2",
        bbox={"boxstyle": "circle", "facecolor": "None", "edgecolor": "black"},
    )
    plt.figtext(0.29, 0.3, "Mode", fontweigth='bold')
    plt.show()
    plt.close()
