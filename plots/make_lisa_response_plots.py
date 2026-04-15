from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# User settings: edit these
# ============================================================
U_VALUES = [0.1, 0.5, 1.0, 2.0]   # dimensionless frequencies u = f / f_star
POLARIZATION = "I"                 # "R", "L", or "I"
COMPONENT = "real"                  # "abs", "real", or "imag"
PANEL_SCALE = "individual"         # "individual" or "shared"
LEVELS = 7
N_THETA = 181
N_PHI = 361

SAVE_PNGS = True
SAVE_PDF = True
OUTPUT_DIR = Path(__file__).resolve().parent / "lisa_response_manual"
PDF_NAME = "lisa_response_manual.pdf"


# ============================================================
# LISA response ingredients
# ============================================================
C_LIGHT = 299_792_458.0
L_ARM = 2.5e9
F_STAR = C_LIGHT / (2.0 * np.pi * L_ARM)

C_AET = np.array(
    [
        [-1 / np.sqrt(2), 0.0, 1 / np.sqrt(2)],
        [1 / np.sqrt(6), -2 / np.sqrt(6), 1 / np.sqrt(6)],
        [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
    ],
    dtype=float,
)
O_LIST = ["A", "E", "T"]
O_IDX = {name: idx for idx, name in enumerate(O_LIST)}
PAIR_GROUPS = {
    "AA": ("AA",),
    "EE": ("EE",),
    "TT": ("TT",),
    "AE": ("AE", "EA"),
    "AT": ("AT", "TA"),
    "ET": ("ET", "TE"),
}


def sinc(x):
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x, dtype=np.complex128)
    mask = np.abs(x) >= 1e-12
    out[mask] = np.sin(x[mask]) / x[mask]
    return out


def m_transfer(u, mu):
    arg = 0.5 * u * (1.0 + mu)
    return np.exp(1j * arg) * sinc(arg)


def t_transfer(u, mu):
    return np.exp(-1j * u) * m_transfer(u, -mu) + np.exp(-1j * u * mu) * m_transfer(u, mu)


def polarization_tensors(theta, phi):
    e_theta = np.stack(
        [
            np.cos(theta) * np.cos(phi),
            np.cos(theta) * np.sin(phi),
            -np.sin(theta),
        ],
        axis=1,
    )
    e_phi = np.stack(
        [
            -np.sin(phi),
            np.cos(phi),
            np.zeros_like(phi),
        ],
        axis=1,
    )
    e_plus = e_theta[:, :, None] * e_theta[:, None, :] - e_phi[:, :, None] * e_phi[:, None, :]
    e_cross = e_theta[:, :, None] * e_phi[:, None, :] + e_phi[:, :, None] * e_theta[:, None, :]
    return e_plus, e_cross


def g_pol_from_arm(lhat, e_tensor):
    return 0.5 * np.einsum("i,j,nij->n", lhat, lhat, e_tensor)


def unit(vector):
    return vector / np.linalg.norm(vector)


def build_geometry():
    r1 = np.array([0.0, 0.0, 0.0])
    r2 = np.array([L_ARM, 0.0, 0.0])
    r3 = np.array([0.5 * L_ARM, 0.5 * np.sqrt(3.0) * L_ARM, 0.0])

    l12 = unit(r2 - r1)
    l13 = unit(r3 - r1)
    l21 = unit(r1 - r2)
    l23 = unit(r3 - r2)
    l31 = unit(r1 - r3)
    l32 = unit(r2 - r3)
    arms_by_sc = {0: (l12, l13), 1: (l23, l21), 2: (l31, l32)}

    pos_sec = [r1 / C_LIGHT, r2 / C_LIGHT, r3 / C_LIGHT]
    delta_sec = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            delta_sec[i, j] = pos_sec[i] - pos_sec[j]

    return arms_by_sc, delta_sec


def angular_grid(n_theta, n_phi):
    theta_vals = np.linspace(0.0, np.pi, int(n_theta))
    phi_vals = np.linspace(0.0, 2.0 * np.pi, int(n_phi))
    phi_grid, theta_grid = np.meshgrid(phi_vals, theta_vals)
    theta = theta_grid.ravel()
    phi = phi_grid.ravel()
    sin_theta = np.sin(theta)
    khat = np.stack(
        [
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            np.cos(theta),
        ],
        axis=1,
    )
    return theta_vals, phi_vals, theta_grid, phi_grid, khat


def compute_pair_maps(u, polarization="R", n_theta=181, n_phi=361):
    polarization = polarization.upper()
    if polarization not in {"R", "L", "I"}:
        raise ValueError("POLARIZATION must be 'R', 'L', or 'I'")

    f = u * F_STAR
    _, _, theta_grid, phi_grid, khat = angular_grid(n_theta, n_phi)
    theta = theta_grid.ravel()
    phi = phi_grid.ravel()
    npix = theta.size

    e_plus, e_cross = polarization_tensors(theta, phi)
    arms_by_sc, delta_sec = build_geometry()

    mu_a = np.empty((3, npix), dtype=np.float64)
    mu_b = np.empty((3, npix), dtype=np.float64)
    gp_a = np.empty((3, npix), dtype=np.float64)
    gx_a = np.empty((3, npix), dtype=np.float64)
    gp_b = np.empty((3, npix), dtype=np.float64)
    gx_b = np.empty((3, npix), dtype=np.float64)

    for i in range(3):
        la, lb = arms_by_sc[i]
        mu_a[i] = khat @ la
        mu_b[i] = khat @ lb
        gp_a[i] = g_pol_from_arm(la, e_plus).real
        gx_a[i] = g_pol_from_arm(la, e_cross).real
        gp_b[i] = g_pol_from_arm(lb, e_plus).real
        gx_b[i] = g_pol_from_arm(lb, e_cross).real

    kdot_delta = np.zeros((3, 3, npix), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            kdot_delta[i, j] = khat @ delta_sec[i, j]
    kdot_flat = kdot_delta.reshape(9, npix)

    ordered_pairs = [(o, op) for o in O_LIST for op in O_LIST]
    ordered_labels = [f"{o}{op}" for o, op in ordered_pairs]
    mix = np.empty((9, 9), dtype=np.float64)
    for pidx, (o, op) in enumerate(ordered_pairs):
        co = C_AET[O_IDX[o]]
        cop = C_AET[O_IDX[op]]
        mix[pidx, :] = (co[:, None] * cop[None, :]).reshape(9)

    ta_a = t_transfer(u, mu_a)
    ta_b = t_transfer(u, mu_b)
    rplus = gp_a * ta_a - gp_b * ta_b
    rcross = gx_a * ta_a - gx_b * ta_b

    plusplus = rplus[:, None, :] * np.conj(rplus[None, :, :])
    crosscross = rcross[:, None, :] * np.conj(rcross[None, :, :])
    if polarization == "I":
        pol_sum = plusplus + crosscross
    else:
        handedness = 1.0 if polarization == "R" else -1.0
        pol_sum = 0.5 * (
            plusplus
            + crosscross
            + 1j * handedness
            * (
                rcross[:, None, :] * np.conj(rplus[None, :, :])
                - rplus[:, None, :] * np.conj(rcross[None, :, :])
            )
        )

    phase_flat = np.exp(-2j * np.pi * f * kdot_flat)
    maps_ij_flat = (1.0 / (8.0 * np.pi)) * pol_sum.reshape(9, npix) * phase_flat
    maps_pair = mix @ maps_ij_flat

    ordered_map_dict = {
        label: maps_pair[pidx].reshape(theta_grid.shape)
        for pidx, label in enumerate(ordered_labels)
    }
    pair_maps = {}
    for pair, names in PAIR_GROUPS.items():
        stacked = np.stack([ordered_map_dict[name] for name in names], axis=0)
        pair_maps[pair] = np.mean(stacked, axis=0)

    return theta_grid, phi_grid, f, pair_maps


def component_values(values, component):
    component = component.lower()
    if component == "abs":
        return np.abs(values)
    if component == "real":
        return np.real(values)
    if component == "imag":
        return np.imag(values)
    raise ValueError("COMPONENT must be 'abs', 'real', or 'imag'")


def pi_ticks():
    phi_ticks = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi]
    phi_labels = ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
    theta_ticks = [0.0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi]
    theta_labels = ["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"]
    return phi_ticks, phi_labels, theta_ticks, theta_labels


def plot_frequency(u, polarization, component, panel_scale, levels, n_theta, n_phi):
    theta_grid, phi_grid, _, pair_maps = compute_pair_maps(
        u=u,
        polarization=polarization,
        n_theta=n_theta,
        n_phi=n_phi,
    )
    pairs = ["AA", "EE", "TT", "AE", "AT", "ET"]
    values_by_pair = {
        pair: component_values(pair_maps[pair], component)
        for pair in pairs
    }

    fig, axes = plt.subplots(2, 3, figsize=(15.6, 8.0), squeeze=False)

    shared_bounds = None
    if panel_scale == "shared":
        stacked = np.stack(list(values_by_pair.values()), axis=0)
        if component == "abs":
            shared_bounds = (0.0, np.max(stacked))
        else:
            vmax = np.max(np.abs(stacked))
            shared_bounds = (-vmax, vmax)

    phi_ticks, phi_labels, theta_ticks, theta_labels = pi_ticks()

    for ax, pair in zip(axes.flat, pairs):
        values = values_by_pair[pair]
        if panel_scale == "shared":
            vmin, vmax = shared_bounds
        elif component == "abs":
            vmin, vmax = 0.0, np.max(values)
        else:
            vmax = np.max(np.abs(values))
            vmin = -vmax
        if np.isclose(vmax, vmin):
            vmax = vmin + 1e-15

        contour_levels = np.linspace(vmin, vmax, int(levels))
        contour = ax.contourf(
            phi_grid,
            theta_grid,
            values,
            levels=contour_levels,
            cmap="cividis" if component == "abs" else "RdBu_r",
        )
        ax.contour(
            phi_grid,
            theta_grid,
            values,
            levels=contour_levels,
            colors="0.35",
            linewidths=0.45,
            alpha=0.55,
        )
        if component == "abs":
            title = rf"$|R_{{{polarization}}}^{{{pair}}}|$ integrand"
        elif component == "real":
            title = rf"$\Re(R_{{{polarization}}}^{{{pair}}})$ integrand"
        else:
            title = rf"$\Im(R_{{{polarization}}}^{{{pair}}})$ integrand"
        ax.set_title(title)
        ax.set_xlabel(r"azimuthal angle $\phi$")
        ax.set_ylabel(r"polar angle $\theta$")
        ax.set_xticks(phi_ticks)
        ax.set_xticklabels(phi_labels)
        ax.set_yticks(theta_ticks)
        ax.set_yticklabels(theta_labels)
        fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Angular dependence of LISA response-function integrands "
        f"for {polarization}-handed polarization at "
        rf"$f/f_{{\star}}={u:.3g}$"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig


def format_u_tag(u):
    text = f"{u:.5g}"
    return text.replace("-", "m").replace(".", "p")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_lines = [
        "# LISA angular response plots",
        "# columns: u=f/F_STAR, frequency_Hz, png_path",
    ]

    pdf_path = OUTPUT_DIR / PDF_NAME
    pdf = PdfPages(pdf_path) if SAVE_PDF else None

    try:
        for u in U_VALUES:
            fig = plot_frequency(
                u=float(u),
                polarization=POLARIZATION,
                component=COMPONENT,
                panel_scale=PANEL_SCALE,
                levels=LEVELS,
                n_theta=N_THETA,
                n_phi=N_PHI,
            )

            png_path = OUTPUT_DIR / f"lisa_response_u_{format_u_tag(float(u))}.png"
            if SAVE_PNGS:
                fig.savefig(png_path, bbox_inches="tight", dpi=200)
            if SAVE_PDF:
                pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            manifest_lines.append(
                f"{float(u):.12g} {(float(u) * F_STAR):.12g} {png_path}"
            )
            print(png_path)
    finally:
        if pdf is not None:
            pdf.close()

    manifest_path = OUTPUT_DIR / "manifest.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n")
    print(manifest_path)
    if SAVE_PDF:
        print(pdf_path)


if __name__ == "__main__":
    main()
