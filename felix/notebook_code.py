import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from matplotlib.ticker import LogLocator, LogFormatterMathtext


# ============================================================
# Constants (paper) - Normaal direct vanuit paper
# ============================================================
C_LIGHT = 299_792_458.0
L_ARM   = 2.5e9
F_STAR  = C_LIGHT / (2.0*np.pi*L_ARM)

MPC_M     = 3.0856775814913673e22
H0_OVER_h = (100.0 * 1000.0) / MPC_M

P_OMS_PM = 15.0
A_ACC_FM = 3.0

# Schaal factor Y00
Y00 = 1.0 / np.sqrt(4.0*np.pi)

# ============================================================
# Appendix B: Noise functions
# ============================================================
def S_oms(f):
    f = np.asarray(f, float)
    return (P_OMS_PM * 1e-12)**2 * (1.0 + (2.0e-3 / f)**4)

def S_acc(f):
    f = np.asarray(f, float)
    return (A_ACC_FM * 1e-15)**2 * (1.0 + (0.4e-3 / f)**2) * (1.0 + (f / 8.0e-3)**4)

def N_tilde_AE(f):
    f = np.asarray(f, float)
    x = f / F_STAR
    cosx = np.cos(x)
    term_oms = 0.5 * (2.0 + cosx) * (S_oms(f) / L_ARM**2)
    term_acc = 2.0 * (1.0 + cosx + cosx**2) * (S_acc(f) / L_ARM**2) * (1.0/(2.0*np.pi*f))**4
    return term_oms + term_acc

def N_tilde_T(f):
    f = np.asarray(f, float)
    x = f / F_STAR
    one_minus_cos = 1.0 - np.cos(x)
    term_oms = one_minus_cos * (S_oms(f) / L_ARM**2)
    term_acc = 2.0 * (one_minus_cos**2) * (S_acc(f) / L_ARM**2) * (1.0/(2.0*np.pi*f))**4
    return term_oms + term_acc

# ============================================================
# Appendix A helper functies (deze misten denk ik ook de vorige implementatie)
# ============================================================
def sinc(x):
    x = np.asarray(x)
    out = np.ones_like(x, dtype=np.complex128)
    small = np.abs(x) < 1e-12
    xs = x[~small]
    out[~small] = np.sin(xs) / xs
    return out

def M_transfer(u, mu):
    arg = 0.5 * u * (1.0 + mu)
    return np.exp(1j * arg) * sinc(arg)

def T_transfer(u, mu):
    return np.exp(-1j*u) * M_transfer(u, -mu) + np.exp(-1j*u*mu) * M_transfer(u, mu)

# ============================================================
# Polarization tensors and projections on a HEALPix grid (aangepast door ChatGPT, ik was de som over de twee polarisaties miss vergeten) 
# ============================================================
def healpix_grid(nside):
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix), nest=False)
    sin_t = np.sin(theta)
    khat = np.stack([sin_t*np.cos(phi), sin_t*np.sin(phi), np.cos(theta)], axis=1)
    return khat, theta, phi

def polarization_tensors_from_k(khat, theta, phi):
    e_theta = np.stack([np.cos(theta)*np.cos(phi),
                        np.cos(theta)*np.sin(phi),
                        -np.sin(theta)], axis=1)
    e_phi = np.stack([-np.sin(phi),
                      np.cos(phi),
                      np.zeros_like(phi)], axis=1)
    e_plus  = e_theta[:, :, None]*e_theta[:, None, :] - e_phi[:, :, None]*e_phi[:, None, :]
    e_cross = e_theta[:, :, None]*e_phi[:, None, :]   + e_phi[:, :, None]*e_theta[:, None, :]
    return e_plus, e_cross

def g_pol_from_arm(lhat, e_tensor):
    return 0.5 * np.einsum("i,j,nij->n", lhat, lhat, e_tensor)

# ============================================================
# A,E,T channel mixing (paper Eq. 4.25) - De onderstaande matrix zijn de constanten die daar staan voor F1, F2 en F3
# ============================================================
C_AET = np.array([
    [-1/np.sqrt(2),  0.0,           1/np.sqrt(2)],  # A
    [ 1/np.sqrt(6), -2/np.sqrt(6),  1/np.sqrt(6)],  # E
    [ 1/np.sqrt(3),  1/np.sqrt(3),  1/np.sqrt(3)],  # T
], dtype=float)
O_list = ["A", "E", "T"]
O_idx  = {O: i for i, O in enumerate(O_list)}
PAIR_GROUPS_AET = {
    "AA": ("AA",),
    "EE": ("EE",),
    "TT": ("TT",),
    "AE": ("AE", "EA"),
    "AT": ("AT", "TA"),
    "ET": ("ET", "TE"),
}

# ============================================================
# Sensitivity equations (paper Eqs. 4.42-4.43) letterlijk gwn de formules net boven Figuur 9. (NOTE VOOR BERT - zou het kunnen dat we de
# de channels dubbel moeten tellen aangezien we AE en EA , ... hebben? Die factor van twee (1.0 / np.sqrt(inv2) --> 2.0 / np.sqrt(inv2)
# fixt het probleem (normaal gezien), en doordat de onderstaande functie een beetje herschreven is ben ik niet zeker of al die channels appart geteld worden.
# Ik denk dat de code gebruik maakt van de symmetrie van de kanalen en dus AE = EA doet, maar dan miss wel maar alleen AE telt... Zal
# zelf ook nog nagaan maar had ff druk. Een andere oplossing kon zijn dat een van de schaalfactoren miss in de verkeerde plaats stond en 
# zo te veel werd meegeteld ofs maar ik denk dat de schaalfactoren eig wel op de juiste plaats staan. 
# ============================================================
def omega_channel_channel(f, Rtilde, Ntilde_O, Ntilde_Op):
    pref = (4.0*np.pi**2 * np.sqrt(4.0*np.pi)) / (3.0 * (H0_OVER_h**2))
    return pref * (f**3) * (np.sqrt(Ntilde_O * Ntilde_Op) / Rtilde)

def optimal_omega_from_ordered_pairs(omega_pairs_9):
    inv2 = np.zeros_like(omega_pairs_9[0], dtype=float)
    for om in omega_pairs_9:
        inv2 += 1.0 / (om*om)
    return 1.0 / np.sqrt(inv2)

def power_law_omega_gw_h2(f, amplitude_h2=1.0e-12, alpha=0.65, f_ref=2.5e-3):
    f = np.asarray(f, float)
    return amplitude_h2 * (f / f_ref)**alpha

def _evaluate_omega_gw_h2(f_grid, omega_gw_h2):
    if callable(omega_gw_h2):
        values = np.asarray(omega_gw_h2(f_grid), dtype=float)
    else:
        values = np.asarray(omega_gw_h2, dtype=float)
        if values.shape != f_grid.shape:
            raise ValueError("If omega_gw_h2 is an array, it must have the same shape as f_grid.")
    return values

def _resolve_c_ell_gw(c_ell_gw, ell):
    if c_ell_gw is None:
        return 1.0
    if np.isscalar(c_ell_gw):
        value = float(c_ell_gw)
    elif isinstance(c_ell_gw, dict):
        value = float(c_ell_gw.get(ell, 0.0))
    else:
        values = np.asarray(c_ell_gw, dtype=float)
        if ell >= values.size:
            raise ValueError("c_ell_gw must contain a value for every requested multipole.")
        value = float(values[ell])
    if value < 0.0:
        raise ValueError("Eq. (4.44) needs C_ell^GW >= 0 for every multipole.")
    return value

def _cumulative_trapezoid(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("x and y must be one-dimensional arrays with the same shape.")
    cumulative = np.zeros_like(x, dtype=float)
    if x.size >= 2:
        cumulative[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return cumulative

def compute_snr_per_multipole_eq444(
    f_grid,
    raw_sensitivity_by_ell,
    omega_gw_h2,
    c_ell_gw=None,
    t_obs_yr=4.0,
    fmin=None,
    fmax=None,
):
    r"""
    Exact implementation of Eq. (4.44):

      <SNR>_ell^2 = T \int df
                    [ sqrt(C_ell^GW) * Omega_GW(f) h^2
                      / Omega_{GW,n}^ell(f) h^2 ]^2

    The denominator must be the raw Eq. (4.43) sensitivity, not the Figure 9
    curve divided by sqrt(4*pi).
    """
    f = np.asarray(f_grid, dtype=float)
    if np.any(np.diff(f) <= 0.0):
        raise ValueError("f_grid must be strictly increasing.")

    omega_gw_vals = _evaluate_omega_gw_h2(f, omega_gw_h2)
    t_obs_sec = float(t_obs_yr * 365.25 * 24.0 * 3600.0)

    band = np.ones_like(f, dtype=bool)
    if fmin is not None:
        band &= f >= float(fmin)
    if fmax is not None:
        band &= f <= float(fmax)
    if np.count_nonzero(band) < 2:
        raise ValueError("The selected frequency band must contain at least two grid points.")

    ells = np.array(sorted(raw_sensitivity_by_ell.keys()), dtype=int)
    snr2 = np.zeros_like(ells, dtype=float)
    integrands = {}
    cumulative_snr2 = {}
    cumulative_frequency = {}

    for i, ell in enumerate(ells):
        c_ell = _resolve_c_ell_gw(c_ell_gw, ell)
        omega_sig_ell = np.sqrt(c_ell) * omega_gw_vals
        omega_noise_ell = np.asarray(raw_sensitivity_by_ell[ell], dtype=float)
        if omega_noise_ell.shape != f.shape:
            raise ValueError(f"Sensitivity curve for ell={ell} does not match f_grid.")

        integrand = np.zeros_like(f, dtype=float)
        mask = band & np.isfinite(omega_sig_ell) & np.isfinite(omega_noise_ell) & (omega_noise_ell > 0.0)
        if np.count_nonzero(mask) >= 2:
            integrand[mask] = (omega_sig_ell[mask] / omega_noise_ell[mask])**2
            cumulative = t_obs_sec * _cumulative_trapezoid(f[mask], integrand[mask])
            snr2[i] = float(cumulative[-1])
        else:
            cumulative = np.zeros(np.count_nonzero(mask), dtype=float)
        integrands[ell] = integrand
        cumulative_snr2[ell] = cumulative
        cumulative_frequency[ell] = f[mask]

    return {
        "ells": ells,
        "snr2": snr2,
        "snr": np.sqrt(snr2),
        "integrands": integrands,
        "cumulative_snr2": cumulative_snr2,
        "cumulative_frequency": cumulative_frequency,
        "omega_gw_h2": omega_gw_vals,
    }

def print_snr_table(snr_results):
    print(f"{'ell':>3} {'SNR^2':>16} {'SNR':>16}")
    for ell, snr2, snr in zip(snr_results["ells"], snr_results["snr2"], snr_results["snr"]):
        print(f"{ell:3d} {snr2:16.8e} {snr:16.8e}")

# ============================================================
# Key point for "real basis" + iter=1 with complex sky maps
# ============================================================
# healpy.map2alm(...) assumes a *real* map (standard use case).
# Our per-frequency sky maps are complex because of exp(-2π i f k·Δx).
#
# To use healpy's SHT machinery (and thus iter=1) while keeping the *full*
# complex harmonic content, we compute:
#   a_{lm}^*basis  = ∫ f(Ω) Y_{lm}^*(Ω) dΩ  via map2alm(Re f) + i map2alm(Im f)
# and also
#   b_{lm}         = ∫ f(Ω) Y_{lm}(Ω)  dΩ  via map2alm(Re f) - i map2alm(Im f) but with
# the correct operation: b_{lm} = (∫ f*(Ω) Y_{lm}^*(Ω) dΩ)^*.
#
# For any complex f:
#   c_{lm} := ∫ f*(Ω) Y_{lm}^*(Ω) dΩ
# then
#   b_{lm} = c_{lm}^*.
#
# So we compute:
#   alm_f  = map2alm(Re f) + i map2alm(Im f)         (m>=0 stored)
#   alm_fc = map2alm(Re f) - i map2alm(Im f)         equals map2alm(conj(f)) (linearity)
# and use:
#   power_l = |a_{l0}|^2 + sum_{m=1..l} ( |a_{lm}|^2 + |c_{lm}|^2 )
# where c_{lm} are the coefficients from alm_fc (since alm_fc = c_{lm} for m>=0).
#
# This quantity equals sum_{m=-l..l} |Z_{lm}|^2 for the complex-harmonic basis,
# i.e. the rotation-invariant response used in the paper.
#
# ============================================================

# cDit (bovenstaande uitleg) was de belangrijkste additie van ChatGPT, het was op deze plaats in de code datik dingen had geporbeerd te fixen
# maar vond het niet... de logica is niet veel veranderd maar de implementatie is gewoon verbeterd, vooral die van de spherical harmonics.

def precompute_alm_indices(lmax):
    idx_by_ell = {}
    idx_mpos_by_ell = {}
    for ell in range(lmax + 1):
        idxs = [hp.Alm.getidx(lmax, ell, m) for m in range(0, ell + 1)]
        idx_by_ell[ell] = np.array(idxs, dtype=np.int64)
        if ell >= 1:
            idxs_mpos = [hp.Alm.getidx(lmax, ell, m) for m in range(1, ell + 1)]
            idx_mpos_by_ell[ell] = np.array(idxs_mpos, dtype=np.int64)
        else:
            idx_mpos_by_ell[ell] = np.array([], dtype=np.int64)
    return idx_by_ell, idx_mpos_by_ell


def _build_static_lisa_geometry():
    L = L_ARM
    r1 = np.array([0.0, 0.0, 0.0])
    r2 = np.array([L, 0.0, 0.0])
    r3 = np.array([0.5 * L, 0.5 * np.sqrt(3.0) * L, 0.0])

    def unit(v):
        return v / np.linalg.norm(v)

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


def _angular_grid(n_theta=181, n_phi=361):
    theta_vals = np.linspace(0.0, np.pi, int(n_theta))
    phi_vals = np.linspace(0.0, 2.0 * np.pi, int(n_phi))
    phi_grid, theta_grid = np.meshgrid(phi_vals, theta_vals)
    theta = theta_grid.ravel()
    phi = phi_grid.ravel()
    sin_t = np.sin(theta)
    khat = np.stack(
        [sin_t * np.cos(phi), sin_t * np.sin(phi), np.cos(theta)],
        axis=1,
    )
    return theta_vals, phi_vals, theta_grid, phi_grid, khat


def compute_lisa_response_pair_sky_maps(
    f=None,
    u=0.1,
    n_theta=181,
    n_phi=361,
    symmetrize=True,
    polarization="R",
):
    """
    Compute the angular integrand maps for the LISA A/E/T response pairs
    on a regular (phi, theta) grid.

    Args:
        f: frequency in Hz. If ``None``, use ``u * F_STAR``.
        u: dimensionless frequency ``f / F_STAR`` used when ``f`` is ``None``.
        n_theta: number of polar-angle samples in ``[0, pi]``.
        n_phi: number of azimuth samples in ``[0, 2*pi]``.
        symmetrize: if ``True``, return the six independent pairs
            ``AA, EE, TT, AE, AT, ET`` with cross-pairs averaged over
            both channel orderings.
        polarization: one of ``"R"``, ``"L"``, or ``"I"`` for
            right-handed, left-handed, or unpolarized/intensity maps.

    Returns:
        dict: containing ``theta_vals``, ``phi_vals``, ``theta_grid``,
        ``phi_grid``, ``frequency``, ``u``, and ``maps``.
    """
    if f is None:
        f = float(u) * F_STAR
    f = float(f)
    u = f / F_STAR
    polarization = polarization.upper()
    if polarization not in {"R", "L", "I"}:
        raise ValueError("polarization must be 'R', 'L', or 'I'")

    theta_vals, phi_vals, theta_grid, phi_grid, khat = _angular_grid(
        n_theta=n_theta,
        n_phi=n_phi,
    )
    theta = theta_grid.ravel()
    phi = phi_grid.ravel()
    npix = theta.size

    e_plus, e_cross = polarization_tensors_from_k(khat, theta, phi)
    arms_by_sc, delta_sec = _build_static_lisa_geometry()

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

    ordered_pairs = [(O, Op) for O in O_list for Op in O_list]
    ordered_labels = [f"{O}{Op}" for O, Op in ordered_pairs]
    mix = np.empty((9, 9), dtype=np.float64)
    for pidx, (O, Op) in enumerate(ordered_pairs):
        cO = C_AET[O_idx[O]]
        cOp = C_AET[O_idx[Op]]
        mix[pidx, :] = (cO[:, None] * cOp[None, :]).reshape(9)

    ta_a = T_transfer(u, mu_a)
    ta_b = T_transfer(u, mu_b)
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
    if not symmetrize:
        map_dict = ordered_map_dict
    else:
        map_dict = {}
        for pair, names in PAIR_GROUPS_AET.items():
            stacked = np.stack([ordered_map_dict[name] for name in names], axis=0)
            map_dict[pair] = np.mean(stacked, axis=0)

    return {
        "theta_vals": theta_vals,
        "phi_vals": phi_vals,
        "theta_grid": theta_grid,
        "phi_grid": phi_grid,
        "frequency": f,
        "u": u,
        "polarization": polarization,
        "maps": map_dict,
    }


def _pi_axis_ticks():
    return {
        "phi": (
            [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi],
            ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
        ),
        "theta": (
            [0.0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi],
            ["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"],
        ),
    }


def plot_lisa_response_pair_angular_dependence(
    f=None,
    u=0.1,
    n_theta=181,
    n_phi=361,
    pairs=None,
    component="abs",
    polarization="R",
    levels=7,
    panel_scale="individual",
    cmap_abs="cividis",
    cmap_signed="RdBu_r",
    figsize=None,
    suptitle=None,
    savepath=None,
    show=True,
):
    """
    Plot the angular dependence of the LISA response-function integrands.

    Args:
        f: frequency in Hz. If ``None``, use ``u * F_STAR``.
        u: dimensionless frequency ``f / F_STAR`` used when ``f`` is ``None``.
        n_theta: number of polar-angle samples.
        n_phi: number of azimuth samples.
        pairs: iterable of pair labels to plot. Defaults to the six
            independent pairs ``AA, EE, TT, AE, AT, ET``.
        component: one of ``"abs"``, ``"real"``, or ``"imag"``.
        polarization: one of ``"R"``, ``"L"``, or ``"I"``.
        levels: number of contour levels.
        panel_scale: ``"individual"`` or ``"shared"`` color scaling.
        cmap_abs: colormap for magnitude plots.
        cmap_signed: colormap for signed plots.
        figsize: optional figure size.
        suptitle: optional figure title.
        savepath: optional path passed to ``savefig``.
        show: call ``plt.show()`` before returning.

    Returns:
        tuple: ``(fig, axes, sky_maps)``.
    """
    component = component.lower()
    if component not in {"abs", "real", "imag"}:
        raise ValueError("component must be 'abs', 'real', or 'imag'")
    if panel_scale not in {"individual", "shared"}:
        raise ValueError("panel_scale must be 'individual' or 'shared'")

    sky_maps = compute_lisa_response_pair_sky_maps(
        f=f,
        u=u,
        n_theta=n_theta,
        n_phi=n_phi,
        symmetrize=True,
        polarization=polarization,
    )
    if pairs is None:
        pairs = ["AA", "EE", "TT", "AE", "AT", "ET"]
    pairs = list(pairs)

    def extract_component(values):
        if component == "abs":
            return np.abs(values)
        if component == "real":
            return np.real(values)
        return np.imag(values)

    values_by_pair = {
        pair: extract_component(sky_maps["maps"][pair])
        for pair in pairs
    }

    n_panels = len(pairs)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    if figsize is None:
        figsize = (5.2 * ncols, 4.0 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    shared_bounds = None
    if panel_scale == "shared":
        stacked = np.stack(list(values_by_pair.values()), axis=0)
        if component == "abs":
            vmax = np.max(stacked)
            shared_bounds = (0.0, vmax)
        else:
            vmax = np.max(np.abs(stacked))
            shared_bounds = (-vmax, vmax)

    tick_info = _pi_axis_ticks()
    phi_ticks, phi_labels = tick_info["phi"]
    theta_ticks, theta_labels = tick_info["theta"]

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
            sky_maps["phi_grid"],
            sky_maps["theta_grid"],
            values,
            levels=contour_levels,
            cmap=cmap_abs if component == "abs" else cmap_signed,
        )
        ax.contour(
            sky_maps["phi_grid"],
            sky_maps["theta_grid"],
            values,
            levels=contour_levels,
            colors="0.35",
            linewidths=0.45,
            alpha=0.55,
        )
        title_component = {
            "abs": rf"$|R_{{{sky_maps['polarization']}}}^{{{pair}}}|$ integrand",
            "real": rf"$\Re(R_{{{sky_maps['polarization']}}}^{{{pair}}})$ integrand",
            "imag": rf"$\Im(R_{{{sky_maps['polarization']}}}^{{{pair}}})$ integrand",
        }[component]
        ax.set_title(title_component)
        ax.set_xlabel(r"azimuthal angle $\phi$")
        ax.set_ylabel(r"polar angle $\theta$")
        ax.set_xticks(phi_ticks)
        ax.set_xticklabels(phi_labels)
        ax.set_yticks(theta_ticks)
        ax.set_yticklabels(theta_labels)
        fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes.flat[n_panels:]:
        ax.axis("off")

    if suptitle is None:
        suptitle = (
            "Angular dependence of LISA response-function integrands "
            rf"for {sky_maps['polarization']}-handed polarization at "
            rf"$f / f_{{\star}} = {sky_maps['u']:.3g}$"
        )
    fig.suptitle(suptitle)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    return fig, axes, sky_maps

def compute_lisa_multipole_sensitivity(f_grid, lmax=10, nside=32, iter_sht=1):
    """
    Uses healpy's spherical harmonic transform with iter=iter_sht (e.g. 1),
    while correctly accounting for complex sky maps by computing both:
      a_lm = ∫ f Y_lm^* dΩ
      c_lm = ∫ f* Y_lm^* dΩ
    and forming:
      sum_{m=-l..l} |Z_lm|^2  ==  |a_l0|^2 + Σ_{m=1..l} (|a_lm|^2 + |c_lm|^2)

    This is equivalent to working in a real spherical-harmonic basis (cos/sin modes),
    and it allows use of iter=1 without losing the negative-m information.

    Returns both:
      - raw Eq. (4.43) sensitivities
      - Figure 9 curves, which are raw / sqrt(4*pi)
    """
    f_grid = np.asarray(f_grid, float)
    if lmax > 3*nside - 1:
        raise ValueError(f"HEALPix sampling constraint: need lmax <= 3*nside-1, got lmax={lmax}, nside={nside}")

    # --- Geometry (same as before)
    L = L_ARM
    r1 = np.array([0.0, 0.0, 0.0])
    r2 = np.array([L,   0.0, 0.0])
    r3 = np.array([0.5*L, 0.5*np.sqrt(3.0)*L, 0.0])
    pos = [r1, r2, r3]

    def unit(v):
        return v / np.linalg.norm(v)

    l12 = unit(r2-r1); l13 = unit(r3-r1)
    l21 = unit(r1-r2); l23 = unit(r3-r2)
    l31 = unit(r1-r3); l32 = unit(r2-r3)

    arms_by_sc = {0: (l12, l13), 1: (l23, l21), 2: (l31, l32)}

    pos_sec = [p / C_LIGHT for p in pos]
    delta_sec = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            delta_sec[i, j] = pos_sec[i] - pos_sec[j]

    # --- HEALPix sky (ring)
    khat, theta, phi = healpix_grid(nside)
    npix = khat.shape[0]

    # Polarization tensors on sky
    e_plus, e_cross = polarization_tensors_from_k(khat, theta, phi)

    # --- Precompute arm-dependent sky projections (geometry-only)
    mu_a  = np.empty((3, npix), dtype=np.float64)
    mu_b  = np.empty((3, npix), dtype=np.float64)
    gp_a  = np.empty((3, npix), dtype=np.float64)
    gx_a  = np.empty((3, npix), dtype=np.float64)
    gp_b  = np.empty((3, npix), dtype=np.float64)
    gx_b  = np.empty((3, npix), dtype=np.float64)

    for i in range(3):
        la, lb = arms_by_sc[i]
        mu_a[i] = khat @ la
        mu_b[i] = khat @ lb
        gp_a[i] = g_pol_from_arm(la, e_plus).real
        gx_a[i] = g_pol_from_arm(la, e_cross).real
        gp_b[i] = g_pol_from_arm(lb, e_plus).real
        gx_b[i] = g_pol_from_arm(lb, e_cross).real

    # --- Precompute k·Δx_ij on pixels
    kdot_delta = np.zeros((3, 3, npix), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            kdot_delta[i, j] = khat @ delta_sec[i, j]
    kdot_flat = kdot_delta.reshape(9, npix)

    # --- Precompute mixing from ij-flat(9) to ordered channel pairs(9)
    pairs = [(O, Op) for O in O_list for Op in O_list]  # 9 ordered
    Mix = np.empty((9, 9), dtype=np.float64)
    for pidx, (O, Op) in enumerate(pairs):
        cO  = C_AET[O_idx[O]]
        cOp = C_AET[O_idx[Op]]
        Mix[pidx, :] = (cO[:, None] * cOp[None, :]).reshape(9)

    # --- Noise arrays
    N_A = N_tilde_AE(f_grid)
    N_E = N_A.copy()
    N_T = N_tilde_T(f_grid)
    N_map = {"A": N_A, "E": N_E, "T": N_T}

    # --- ALM index helpers
    idx_by_ell, idx_mpos_by_ell = precompute_alm_indices(lmax)

    # Output
    raw_omega_by_ell = {ell: np.zeros_like(f_grid) for ell in range(lmax + 1)}

    # Work arrays
    Rplus  = np.empty((3, npix), dtype=np.complex128)
    Rcross = np.empty((3, npix), dtype=np.complex128)

    # Normalization prefactor for Eq. (4.15): 1/(8π)
    pref_Rlm = 1.0 / (8.0 * np.pi)

    # Main frequency loop
    for fi, f in enumerate(f_grid):
        u = f / F_STAR

        # Transfer functions for each arm (vectorized)
        Ta_a = T_transfer(u, mu_a)  # (3,npix) complex
        Ta_b = T_transfer(u, mu_b)  # (3,npix) complex

        # R_i^+, R_i^x
        Rplus[:]  = gp_a * Ta_a - gp_b * Ta_b
        Rcross[:] = gx_a * Ta_a - gx_b * Ta_b

        # Σ_A R_i^A R_j^{A*}
        pol_sum = (Rplus[:, None, :] * np.conj(Rplus[None, :, :]) +
                   Rcross[:, None, :] * np.conj(Rcross[None, :, :]))  # (3,3,npix)

        # phase for all ij at once
        phase_flat = np.exp(-2j * np.pi * f * kdot_flat)  # (9,npix)

        # Eq. (4.15) sky "maps" for each ij (flattened): f_ij(Ω) = (1/8π) pol_sum * phase
        maps_ij_flat = pref_Rlm * pol_sum.reshape(9, npix) * phase_flat  # (9,npix) complex

        # Mix ij maps into 9 ordered channel-pair maps (still on sky pixels)
        # maps_pair[pair, pix] = Σ_ij Mix[pair,ij] * maps_ij[ij,pix]
        maps_pair = Mix @ maps_ij_flat  # (9,npix) complex

        # For each channel-pair map, compute two ALM sets:
        #  alm_a : a_lm = ∫ f Y_lm^* dΩ (m>=0 stored)
        #  alm_c : c_lm = ∫ f* Y_lm^* dΩ (m>=0 stored), which we get via map2alm(conj(f))
        #
        # Since healpy expects real map input, we do:
        #  map2alm(Re f) and map2alm(Im f) separately, then combine.
        #
        # For conj(f): Re same, Im negated.
        alms_a = []
        alms_c = []
        for pidx in range(9):
            mp = maps_pair[pidx]
            mp_re = mp.real.astype(np.float64, copy=False)
            mp_im = mp.imag.astype(np.float64, copy=False)

            # a_lm = A + i B
            A = hp.map2alm(mp_re, lmax=lmax, iter=iter_sht, pol=False)
            B = hp.map2alm(mp_im, lmax=lmax, iter=iter_sht, pol=False)
            alm_a = A + 1j * B

            # c_lm = map2alm(conj(f)) = map2alm(Re f - i Im f) = A - i B
            alm_c = A - 1j * B

            alms_a.append(alm_a)
            alms_c.append(alm_c)

        # Now compute Rtilde^ell for each pair using full m power:
        # power_ell = |a_l0|^2 + Σ_{m=1..ell} ( |a_lm|^2 + |c_lm|^2 )
        for ell in range(lmax + 1):
            idx0 = hp.Alm.getidx(lmax, ell, 0)
            idxm = idx_mpos_by_ell[ell]

            Rtilde_pairs = np.empty(9, dtype=np.float64)
            for pidx in range(9):
                aa = alms_a[pidx]
                cc = alms_c[pidx]

                power = np.abs(aa[idx0])**2
                if idxm.size > 0:
                    power += np.sum(np.abs(aa[idxm])**2) + np.sum(np.abs(cc[idxm])**2)

                # The paper's \tilde{R}^\ell_{OO'} normalization carries an
                # overall sqrt(pi) relative to the raw orthonormal-Y_lm power
                # returned by healpy. Without this factor the low-frequency
                # AA monopole limit is 9/(20*sqrt(pi)) instead of 9/20, which
                # shifts the Figure 9 curves high by ~sqrt(pi).
                Rtilde_pairs[pidx] = np.sqrt(np.pi * power)
            Rtilde_pairs = np.maximum(Rtilde_pairs, 1e-45)

            # Build 9 Omegas for ordered pairs and optimally combine
            omega_pairs = []
            for pidx, (O, Op) in enumerate(pairs):
                omega_pairs.append(
                    omega_channel_channel(f, Rtilde_pairs[pidx], N_map[O][fi], N_map[Op][fi])
                )

            raw_omega_by_ell[ell][fi] = optimal_omega_from_ordered_pairs(omega_pairs)

    figure9_omega_by_ell = {
        ell: raw_omega_by_ell[ell] * Y00 for ell in range(lmax + 1)
    }
    return {"raw": raw_omega_by_ell, "figure9": figure9_omega_by_ell}

def reproduce_figure9_healpy_real_basis_iter1(f_grid, lmax=10, nside=32, iter_sht=1):
    return compute_lisa_multipole_sensitivity(
        f_grid=f_grid,
        lmax=lmax,
        nside=nside,
        iter_sht=iter_sht,
    )["figure9"]

def show_snr_table(snr_results):
    ells = snr_results["ells"]
    snr2 = snr_results["snr2"]
    snr  = snr_results["snr"]

    rows = [
        [f"{ell:d}", f"{s2:.4e}", f"{s:.4e}"]
        for ell, s2, s in zip(ells, snr2, snr)
    ]

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(rows) + 1.5))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=[r"$\ell$", r"$\mathrm{SNR}^2$", r"$\mathrm{SNR}$"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.15, 1.35)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.8)
        if r == 0:
            cell.set_text_props(weight="bold")

    plt.tight_layout()
    plt.show()

def plot_cumulative_snr2(snr_results, ells=(0, 2, 4, 3)):
    selected_ells = [ell for ell in ells if ell in snr_results["cumulative_snr2"]]
    if not selected_ells:
        raise ValueError("No requested multipoles are available in snr_results.")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for ell in selected_ells:
        freq = snr_results["cumulative_frequency"][ell]
        cumulative = snr_results["cumulative_snr2"][ell]
        if freq.size == 0:
            continue
        linestyle = "-" if (ell % 2 == 0) else "--"
        ax.semilogx(freq, cumulative, linestyle=linestyle, linewidth=2, label=rf"$\ell={ell}$")

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"Cumulative $\mathrm{SNR}^2(<f)$")
    ax.set_title(r"Running contribution to $\mathrm{SNR}^2$")
    ax.grid(True, which="both", linestyle=":")
    ax.legend()
    fig.tight_layout()
    plt.show(block=False)
    return fig, ax

def compute_reference_omega_ell_h2(
    ells,
    omega_gw_h2,
    c_ell_gw=None,
    reference_frequency_hz=2.5e-3,
):
    ells = np.asarray(ells, dtype=int)
    omega_ref = float(
        _evaluate_omega_gw_h2(
            np.array([reference_frequency_hz], dtype=float),
            omega_gw_h2,
        )[0]
    )
    omega_ell_ref = np.array(
        [
            np.sqrt(_resolve_c_ell_gw(c_ell_gw, int(ell))) * omega_ref
            for ell in ells
        ],
        dtype=float,
    )
    return omega_ell_ref

def plot_snr_vs_omega_ell(
    snr_results,
    omega_gw_h2,
    c_ell_gw=None,
    ells=(0, 1, 2, 3, 4),
    reference_frequency_hz=2.5e-3,
    amplitude_range_factors=(1e-2, 1e2),
):
    available = {int(ell): i for i, ell in enumerate(snr_results["ells"])}
    selected_ells = [int(ell) for ell in ells if int(ell) in available]
    if not selected_ells:
        raise ValueError("No requested multipoles are available in snr_results.")

    omega_ell_ref = compute_reference_omega_ell_h2(
        selected_ells,
        omega_gw_h2=omega_gw_h2,
        c_ell_gw=c_ell_gw,
        reference_frequency_hz=reference_frequency_hz,
    )
    snr_values = np.array(
        [snr_results["snr"][available[ell]] for ell in selected_ells],
        dtype=float,
    )

    positive_omega = omega_ell_ref[np.isfinite(omega_ell_ref) & (omega_ell_ref > 0.0)]
    if positive_omega.size == 0:
        raise ValueError("Need positive reference Omega_ell amplitudes to build the SNR-vs-Omega_ell plot.")

    low_factor, high_factor = amplitude_range_factors
    omega_scan = np.logspace(
        np.log10(low_factor * positive_omega.min()),
        np.log10(high_factor * positive_omega.max()),
        200,
    )

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for ell, omega_ref, snr_ref in zip(selected_ells, omega_ell_ref, snr_values):
        if not np.isfinite(omega_ref) or omega_ref <= 0.0 or not np.isfinite(snr_ref):
            continue
        linestyle = "-" if (ell % 2 == 0) else "--"
        slope = snr_ref / omega_ref
        ax.loglog(
            omega_scan,
            slope * omega_scan,
            linestyle=linestyle,
            linewidth=2,
            label=rf"$\ell={ell}$",
        )
        ax.scatter([omega_ref], [snr_ref], s=36, zorder=3)

    ax.set_xlabel(r"Reference signal amplitude $\Omega_\ell(f_{\rm ref})\,h^2$")
    ax.set_ylabel(r"$\mathrm{SNR}_\ell$")
    ax.set_title(
        r"$\mathrm{SNR}_\ell$ versus $\Omega_\ell$ "
        + rf"at $f_{{\rm ref}}={reference_frequency_hz:.2e}\,\mathrm{{Hz}}$"
    )
    ax.grid(True, which="both", linestyle=":")
    ax.legend(ncol=2)
    fig.tight_layout()
    plt.show(block=False)
    return fig, ax, {
        "ells": np.asarray(selected_ells, dtype=int),
        "omega_ell_ref_h2": omega_ell_ref,
        "snr": snr_values,
        "reference_frequency_hz": float(reference_frequency_hz),
        "omega_scan_h2": omega_scan,
    }

def compute_minimum_detectable_a_ell(
    snr_results,
    omega_gw_h2,
    c_ell_gw=None,
    ells=None,
    reference_frequency_hz=2.5e-3,
    target_snr=1.0,
):
    if target_snr <= 0.0:
        raise ValueError("target_snr must be positive.")

    if ells is None:
        selected_ells = np.asarray(snr_results["ells"], dtype=int)
    else:
        available = {int(ell): i for i, ell in enumerate(snr_results["ells"])}
        selected_ells = np.array([int(ell) for ell in ells if int(ell) in available], dtype=int)
        if selected_ells.size == 0:
            raise ValueError("No requested multipoles are available in snr_results.")

    available = {int(ell): i for i, ell in enumerate(snr_results["ells"])}
    snr_values = np.array(
        [snr_results["snr"][available[int(ell)]] for ell in selected_ells],
        dtype=float,
    )
    a_ref = compute_reference_omega_ell_h2(
        selected_ells,
        omega_gw_h2=omega_gw_h2,
        c_ell_gw=c_ell_gw,
        reference_frequency_hz=reference_frequency_hz,
    )

    a_min = np.full_like(a_ref, np.nan, dtype=float)
    valid = np.isfinite(a_ref) & np.isfinite(snr_values) & (a_ref > 0.0) & (snr_values > 0.0)
    a_min[valid] = a_ref[valid] * (float(target_snr) / snr_values[valid])

    return {
        "ells": selected_ells,
        "a_ref_h2": a_ref,
        "snr": snr_values,
        "a_min_h2": a_min,
        "reference_frequency_hz": float(reference_frequency_hz),
        "target_snr": float(target_snr),
    }

def plot_minimum_detectable_a_ell_vs_ell(
    snr_results,
    omega_gw_h2,
    c_ell_gw=None,
    ells=None,
    reference_frequency_hz=2.5e-3,
    target_snr=1.0,
):
    summary = compute_minimum_detectable_a_ell(
        snr_results,
        omega_gw_h2=omega_gw_h2,
        c_ell_gw=c_ell_gw,
        ells=ells,
        reference_frequency_hz=reference_frequency_hz,
        target_snr=target_snr,
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    ax.plot(
        summary["ells"],
        summary["a_min_h2"],
        marker="o",
        linewidth=2.2,
        color="tab:red",
        label=rf"minimum detectable $A_\ell$ for $\mathrm{{SNR}}_\ell={target_snr:g}$",
    )

    positive = summary["a_min_h2"][np.isfinite(summary["a_min_h2"]) & (summary["a_min_h2"] > 0.0)]
    if positive.size > 0 and positive.max() / positive.min() > 20.0:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=8))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))

    ax.set_xlabel(r"Multipole $\ell$")
    ax.set_ylabel(r"Minimum detectable $A_\ell\,h^2$")
    ax.set_title(
        r"Minimum detectable per-multipole amplitude versus $\ell$"
        + "\n"
        + rf"at $f_{{\rm ref}}={reference_frequency_hz:.2e}\,\mathrm{{Hz}}$"
    )
    ax.grid(True, which="both", linestyle=":")
    ax.legend(frameon=True)
    fig.tight_layout()
    plt.show(block=False)
    return fig, ax, summary

def fit_power_law_minimum_detectable_a_ell_vs_ell(a_ell_summary, ell_min=1):
    fit_ells = np.asarray(a_ell_summary["ells"], dtype=int)
    fit_a_min = np.asarray(a_ell_summary["a_min_h2"], dtype=float)

    valid = (
        np.isfinite(fit_ells)
        & np.isfinite(fit_a_min)
        & (fit_ells >= int(ell_min))
        & (fit_a_min > 0.0)
    )
    fit_ells = fit_ells[valid]
    fit_a_min = fit_a_min[valid]
    if fit_ells.size < 2:
        raise ValueError("Need at least two positive multipoles with positive A_min to fit a power law.")

    coeffs = np.polyfit(np.log(fit_ells.astype(float)), np.log(fit_a_min), 1)
    slope, intercept = coeffs
    beta = float(slope)
    prefactor = float(np.exp(intercept))
    fitted = prefactor * fit_ells.astype(float) ** beta

    ss_res = float(np.sum((fit_a_min - fitted) ** 2))
    ss_tot = float(np.sum((fit_a_min - np.mean(fit_a_min)) ** 2))
    r_squared = np.nan if np.isclose(ss_tot, 0.0) else 1.0 - ss_res / ss_tot

    return {
        "ells": fit_ells,
        "a_min_h2": fit_a_min,
        "beta": beta,
        "prefactor": prefactor,
        "fitted_a_min_h2": fitted,
        "r_squared": r_squared,
        "target_snr": float(a_ell_summary["target_snr"]),
        "reference_frequency_hz": float(a_ell_summary["reference_frequency_hz"]),
    }

def plot_loglog_minimum_detectable_a_ell_vs_ell(
    snr_results,
    omega_gw_h2,
    c_ell_gw=None,
    ells=None,
    reference_frequency_hz=2.5e-3,
    target_snr=1.0,
    ell_min=1,
):
    summary = compute_minimum_detectable_a_ell(
        snr_results,
        omega_gw_h2=omega_gw_h2,
        c_ell_gw=c_ell_gw,
        ells=ells,
        reference_frequency_hz=reference_frequency_hz,
        target_snr=target_snr,
    )
    fit_summary = fit_power_law_minimum_detectable_a_ell_vs_ell(summary, ell_min=ell_min)
    fit_ells = fit_summary["ells"]
    fit_a_min = fit_summary["a_min_h2"]
    beta = fit_summary["beta"]
    prefactor = fit_summary["prefactor"]
    r_squared = fit_summary["r_squared"]

    ell_curve = np.logspace(np.log10(fit_ells.min()), np.log10(fit_ells.max()), 200)
    a_curve = prefactor * ell_curve ** beta

    fig, ax = plt.subplots(figsize=(7.3, 5.0))
    ax.loglog(
        fit_ells,
        fit_a_min,
        "o",
        ms=6,
        color="tab:red",
        label=r"computed $A_{\ell,\min}$",
    )
    ax.loglog(
        ell_curve,
        a_curve,
        "--",
        lw=2.2,
        color="0.25",
        label=rf"fit $\propto \ell^{{{beta:.2f}}}$",
    )

    for ell, a_min in zip(fit_ells, fit_a_min):
        ax.annotate(
            rf"$\ell={int(ell)}$",
            xy=(ell, a_min),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel(r"Multipole $\ell$")
    ax.set_ylabel(r"Minimum detectable $A_\ell\,h^2$")
    ax.set_title(r"Log-log scaling of minimum detectable $A_\ell$ with multipole")
    ax.grid(True, which="both", linestyle=":")
    ax.legend(frameon=True)
    ax.text(
        0.05,
        0.08,
        rf"$A_{{\ell,\min}} \approx {prefactor:.3e}\,\ell^{{{beta:.2f}}}$" + "\n"
        + rf"$R^2 = {r_squared:.3f}$",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.7"),
    )
    fig.tight_layout()
    plt.show(block=False)
    return fig, ax, fit_summary

def fit_power_law_snr_vs_ell(snr_results, ells=None, ell_min=1):
    if ells is None:
        candidate_ells = np.asarray(snr_results["ells"], dtype=int)
    else:
        candidate_ells = np.asarray(ells, dtype=int)

    available = {int(ell): i for i, ell in enumerate(snr_results["ells"])}
    selected_ells = np.array([ell for ell in candidate_ells if int(ell) in available], dtype=int)
    selected_snr = np.array(
        [snr_results["snr"][available[int(ell)]] for ell in selected_ells],
        dtype=float,
    )

    valid = (
        np.isfinite(selected_ells)
        & np.isfinite(selected_snr)
        & (selected_ells >= int(ell_min))
        & (selected_snr > 0.0)
    )
    fit_ells = selected_ells[valid]
    fit_snr = selected_snr[valid]
    if fit_ells.size < 2:
        raise ValueError("Need at least two positive multipoles with positive SNR to fit a power law.")

    coeffs = np.polyfit(np.log(fit_ells.astype(float)), np.log(fit_snr), 1)
    slope, intercept = coeffs
    alpha = -float(slope)
    prefactor = float(np.exp(intercept))
    fitted = prefactor * fit_ells.astype(float) ** (-alpha)

    ss_res = float(np.sum((fit_snr - fitted) ** 2))
    ss_tot = float(np.sum((fit_snr - np.mean(fit_snr)) ** 2))
    r_squared = np.nan if np.isclose(ss_tot, 0.0) else 1.0 - ss_res / ss_tot

    return {
        "ells": fit_ells,
        "snr": fit_snr,
        "alpha": alpha,
        "prefactor": prefactor,
        "fitted_snr": fitted,
        "r_squared": r_squared,
    }

def plot_loglog_snr_vs_ell(snr_results, ells=None, ell_min=1):
    fit_summary = fit_power_law_snr_vs_ell(snr_results, ells=ells, ell_min=ell_min)
    fit_ells = fit_summary["ells"]
    fit_snr = fit_summary["snr"]
    alpha = fit_summary["alpha"]
    prefactor = fit_summary["prefactor"]
    r_squared = fit_summary["r_squared"]

    ell_curve = np.logspace(np.log10(fit_ells.min()), np.log10(fit_ells.max()), 200)
    snr_curve = prefactor * ell_curve ** (-alpha)

    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    ax.loglog(fit_ells, fit_snr, "o", ms=6, label=r"computed $\mathrm{SNR}_\ell$")
    ax.loglog(
        ell_curve,
        snr_curve,
        "--",
        lw=2.2,
        label=rf"fit $\propto \ell^{{-{alpha:.2f}}}$",
    )

    for ell, snr in zip(fit_ells, fit_snr):
        ax.annotate(
            rf"$\ell={int(ell)}$",
            xy=(ell, snr),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel(r"Multipole $\ell$")
    ax.set_ylabel(r"$\mathrm{SNR}_\ell$")
    ax.set_title(r"Log-log scaling of $\mathrm{SNR}_\ell$ with multipole")
    ax.grid(True, which="both", linestyle=":")
    ax.legend(frameon=True)
    ax.text(
        0.05,
        0.08,
        rf"$\mathrm{{SNR}}_\ell \approx {prefactor:.3e}\,\ell^{{-{alpha:.2f}}}$" + "\n"
        + rf"$R^2 = {r_squared:.3f}$",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.7"),
    )
    fig.tight_layout()
    plt.show(block=False)
    return fig, ax, fit_summary

def compute_snr_time_study(
    f_grid,
    raw_sensitivity_by_ell,
    omega_gw_h2,
    c_ell_gw=None,
    t_obs_specs=None,
    fmin=None,
    fmax=None,
):
    if t_obs_specs is None:
        t_obs_specs = [
            ("2 mo", 2.0 / 12.0),
            ("6 mo", 6.0 / 12.0),
            ("1 yr", 1.0),
            ("4 yr", 4.0),
        ]

    labels = []
    t_obs_years = []
    results_by_label = {}
    ells = np.array(sorted(raw_sensitivity_by_ell.keys()), dtype=int)
    snr = np.full((len(t_obs_specs), len(ells)), np.nan, dtype=float)
    snr2 = np.full((len(t_obs_specs), len(ells)), np.nan, dtype=float)

    for i, (label, t_obs_yr) in enumerate(t_obs_specs):
        result = compute_snr_per_multipole_eq444(
            f_grid=f_grid,
            raw_sensitivity_by_ell=raw_sensitivity_by_ell,
            omega_gw_h2=omega_gw_h2,
            c_ell_gw=c_ell_gw,
            t_obs_yr=t_obs_yr,
            fmin=fmin,
            fmax=fmax,
        )
        labels.append(str(label))
        t_obs_years.append(float(t_obs_yr))
        results_by_label[str(label)] = result
        snr[i, :] = result["snr"]
        snr2[i, :] = result["snr2"]

    return {
        "labels": labels,
        "t_obs_years": np.asarray(t_obs_years, dtype=float),
        "ells": ells,
        "snr": snr,
        "snr2": snr2,
        "results_by_label": results_by_label,
    }

def show_snr_time_table(time_study, ells=(0, 1, 2, 3, 4), quantity="snr"):
    quantity = quantity.lower()
    if quantity not in {"snr", "snr2"}:
        raise ValueError("quantity must be 'snr' or 'snr2'")

    study_ells = time_study["ells"]
    ell_to_index = {ell: i for i, ell in enumerate(study_ells)}
    selected_ells = [ell for ell in ells if ell in ell_to_index]
    if not selected_ells:
        raise ValueError("No requested multipoles are available in time_study.")

    values = time_study[quantity]
    rows = []
    for ell in selected_ells:
        idx = ell_to_index[ell]
        rows.append([f"{ell:d}"] + [f"{val:.4e}" for val in values[:, idx]])

    fig, ax = plt.subplots(figsize=(1.6 + 1.55 * len(time_study["labels"]), 0.55 * len(rows) + 1.6))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=[r"$\ell$"] + list(time_study["labels"]),
        loc="center",
        cellLoc="center",
        colLoc="center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1.05, 1.3)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.8)
        if r == 0:
            cell.set_text_props(weight="bold")

    ax.set_title(rf"${quantity.upper().replace('2', '^2')}$ for selected observation times")
    plt.tight_layout()
    plt.show()
    return fig, ax

def plot_snr_vs_observation_time(time_study, ells=(0, 1, 2, 3, 4), quantity="snr"):
    quantity = quantity.lower()
    if quantity not in {"snr", "snr2"}:
        raise ValueError("quantity must be 'snr' or 'snr2'")

    study_ells = time_study["ells"]
    ell_to_index = {ell: i for i, ell in enumerate(study_ells)}
    selected_ells = [ell for ell in ells if ell in ell_to_index]
    if not selected_ells:
        raise ValueError("No requested multipoles are available in time_study.")

    x = np.arange(len(time_study["labels"]), dtype=float)
    values = time_study[quantity]

    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    for ell in selected_ells:
        idx = ell_to_index[ell]
        linestyle = "-" if (ell % 2 == 0) else "--"
        ax.plot(
            x,
            values[:, idx],
            marker="o",
            linewidth=2,
            linestyle=linestyle,
            label=rf"$\ell={ell}$",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(time_study["labels"])
    ax.set_xlabel("Observation time")
    ax.set_ylabel(r"$\mathrm{SNR}$" if quantity == "snr" else r"$\mathrm{SNR}^2$")
    ax.set_title(rf"{'SNR' if quantity == 'snr' else 'SNR$^2$'} versus observation time")
    ax.grid(True, which="both", linestyle=":")
    ax.legend(ncol=2)
    fig.tight_layout()
    plt.show(block=False)
    return fig, ax

def compute_snr_frequency_resolution_study(
    nfreq_values=(480, 640, 960),
    *,
    lmax=10,
    nside=32,
    iter_sht=1,
    t_obs_yr=4.0,
    omega_gw_h2=None,
    c_ell_gw=None,
    log10_fmin=-4.5,
    log10_fmax=-0.35,
    fmin=None,
    fmax=None,
):
    if omega_gw_h2 is None:
        raise ValueError("omega_gw_h2 must be provided for the frequency-resolution study.")

    nfreq_values = np.array(sorted({int(nf) for nf in nfreq_values if int(nf) > 0}), dtype=int)
    if nfreq_values.size == 0:
        raise ValueError("nfreq_values must contain at least one positive integer.")

    results_by_nfreq = {}
    snr = None
    snr2 = None
    ells = None

    for i, nfreq in enumerate(nfreq_values):
        f_grid = np.logspace(log10_fmin, log10_fmax, int(nfreq))
        sensitivity = compute_lisa_multipole_sensitivity(
            f_grid=f_grid,
            lmax=lmax,
            nside=nside,
            iter_sht=iter_sht,
        )
        snr_results = compute_snr_per_multipole_eq444(
            f_grid=f_grid,
            raw_sensitivity_by_ell=sensitivity["raw"],
            omega_gw_h2=omega_gw_h2,
            c_ell_gw=c_ell_gw,
            t_obs_yr=t_obs_yr,
            fmin=fmin,
            fmax=fmax,
        )

        if ells is None:
            ells = snr_results["ells"]
            snr = np.full((nfreq_values.size, len(ells)), np.nan, dtype=float)
            snr2 = np.full((nfreq_values.size, len(ells)), np.nan, dtype=float)

        snr[i, :] = snr_results["snr"]
        snr2[i, :] = snr_results["snr2"]
        results_by_nfreq[int(nfreq)] = {
            "f_grid": f_grid,
            "sensitivity": sensitivity,
            "snr_results": snr_results,
        }

    return {
        "nfreq_values": nfreq_values,
        "ells": ells,
        "snr": snr,
        "snr2": snr2,
        "results_by_nfreq": results_by_nfreq,
        "t_obs_yr": float(t_obs_yr),
    }

def show_snr_frequency_resolution_table(freq_study, ells=(0, 1, 2, 3, 4), quantity="snr"):
    quantity = quantity.lower()
    if quantity not in {"snr", "snr2"}:
        raise ValueError("quantity must be 'snr' or 'snr2'")

    study_ells = freq_study["ells"]
    ell_to_index = {ell: i for i, ell in enumerate(study_ells)}
    selected_ells = [ell for ell in ells if ell in ell_to_index]
    if not selected_ells:
        raise ValueError("No requested multipoles are available in freq_study.")

    values = freq_study[quantity]
    rows = []
    for ell in selected_ells:
        idx = ell_to_index[ell]
        rows.append([f"{ell:d}"] + [f"{val:.4e}" for val in values[:, idx]])

    headers = [r"$\ell$"] + [rf"$N_f={nf}$" for nf in freq_study["nfreq_values"]]
    fig, ax = plt.subplots(figsize=(1.8 + 1.65 * len(headers), 0.55 * len(rows) + 1.6))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1.0, 1.3)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.8)
        if r == 0:
            cell.set_text_props(weight="bold")

    ax.set_title(rf"${quantity.upper().replace('2', '^2')}$ for denser frequency grids")
    plt.tight_layout()
    plt.show()
    return fig, ax

def plot_snr_vs_frequency_samples(freq_study, ells=(0, 1, 2, 3, 4), quantity="snr"):
    quantity = quantity.lower()
    if quantity not in {"snr", "snr2"}:
        raise ValueError("quantity must be 'snr' or 'snr2'")

    study_ells = freq_study["ells"]
    ell_to_index = {ell: i for i, ell in enumerate(study_ells)}
    selected_ells = [ell for ell in ells if ell in ell_to_index]
    if not selected_ells:
        raise ValueError("No requested multipoles are available in freq_study.")

    x = freq_study["nfreq_values"]
    values = freq_study[quantity]

    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    for ell in selected_ells:
        idx = ell_to_index[ell]
        linestyle = "-" if (ell % 2 == 0) else "--"
        ax.plot(
            x,
            values[:, idx],
            marker="o",
            linewidth=2,
            linestyle=linestyle,
            label=rf"$\ell={ell}$",
        )

    ax.set_xlabel(r"Number of frequency samples $N_f$")
    ax.set_ylabel(r"$\mathrm{SNR}$" if quantity == "snr" else r"$\mathrm{SNR}^2$")
    ax.set_title(rf"{'SNR' if quantity == 'snr' else 'SNR$^2$'} versus frequency-grid resolution")
    ax.grid(True, which="both", linestyle=":")
    ax.legend(ncol=2)
    fig.tight_layout()
    plt.show(block=False)
    return fig, ax

# ============================================================
# Example run + plot
# ============================================================
if __name__ == "__main__":
    LMAX = 10
    NSIDE_FAST = 820
    NFREQ_FAST = 320
    HIGHER_NFREQ_VALUES = (480, 640, 960)
    ITER_SHT = 0
    T_OBS_YR = 4.0
    RUN_HIGHER_NFREQ_STUDY = False

    f = np.logspace(-4.5, -0.35, NFREQ_FAST)

    sensitivity = compute_lisa_multipole_sensitivity(
        f_grid=f,
        lmax=LMAX,
        nside=NSIDE_FAST,
        iter_sht=ITER_SHT,
    )
    raw_curves = sensitivity["raw"]
    figure9_curves = sensitivity["figure9"]

    omega_gw_model = lambda freq: power_law_omega_gw_h2(
        freq,
        amplitude_h2=1.0e-12,
        alpha=0.0,
        f_ref=2.5e-3,
    )
    c_ell_gw = np.ones(LMAX + 1)

    snr_results = compute_snr_per_multipole_eq444(
        f_grid=f,
        raw_sensitivity_by_ell=raw_curves,
        omega_gw_h2=omega_gw_model,
        c_ell_gw=c_ell_gw,
        t_obs_yr=T_OBS_YR,
    )
    time_study = compute_snr_time_study(
        f_grid=f,
        raw_sensitivity_by_ell=raw_curves,
        omega_gw_h2=omega_gw_model,
        c_ell_gw=c_ell_gw,
        t_obs_specs=[
            ("2 mo", 2.0 / 12.0),
            ("6 mo", 6.0 / 12.0),
            ("1 yr", 1.0),
            ("4 yr", 4.0),
        ],
    )

    # first: your nice curves
    plt.figure(figsize=(7.5, 7))
    for ell in range(0, LMAX + 1):
        ls = "-" if (ell % 2 == 0) else "--"
        plt.loglog(f, figure9_curves[ell], linestyle=ls, label=rf"$\ell={ell}$")

    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"$\Omega^{\ell}_{{\rm GW},n}(f)\,h^2 / \sqrt{4\pi}$")
    plt.ylim(1e-12, 1e0)

    ax = plt.gca()
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=100))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))

    plt.grid(True, which="both", linestyle=":")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.show(block=False)

    # second: cumulative SNR^2(<f) for representative multipoles
    plot_cumulative_snr2(snr_results, ells=(0, 2, 4, 3))

    # third: SNR values for T = 2 mo, 6 mo, 1 yr, 4 yr and ell = 0..4
    plot_snr_vs_observation_time(time_study, ells=(0, 1, 2, 3, 4), quantity="snr")
    show_snr_time_table(time_study, ells=(0, 1, 2, 3, 4), quantity="snr")

    # fourth: SNR_ell versus a scanned reference Omega_ell amplitude
    plot_snr_vs_omega_ell(
        snr_results,
        omega_gw_h2=omega_gw_model,
        c_ell_gw=c_ell_gw,
        ells=(0, 1, 2, 3, 4),
        reference_frequency_hz=2.5e-3,
    )

    # fifth: log-log SNR_ell versus ell with power-law fit (ell >= 1)
    plot_loglog_snr_vs_ell(
        snr_results,
        ells=tuple(range(0, LMAX + 1)),
        ell_min=1,
    )

    # sixth: minimum detectable A_ell versus ell for SNR_ell = 1
    plot_minimum_detectable_a_ell_vs_ell(
        snr_results,
        omega_gw_h2=omega_gw_model,
        c_ell_gw=c_ell_gw,
        ells=tuple(range(0, LMAX + 1)),
        reference_frequency_hz=2.5e-3,
        target_snr=1.0,
    )

    # seventh: log-log minimum detectable A_ell versus ell with power-law fit
    plot_loglog_minimum_detectable_a_ell_vs_ell(
        snr_results,
        omega_gw_h2=omega_gw_model,
        c_ell_gw=c_ell_gw,
        ells=tuple(range(0, LMAX + 1)),
        reference_frequency_hz=2.5e-3,
        target_snr=1.0,
        ell_min=1,
    )

    # eighth: nice table for the default single-time run
    show_snr_table(snr_results)

    # optional: denser-frequency SNR study (expensive for high NSIDE / high N_f)
    if RUN_HIGHER_NFREQ_STUDY:
        freq_study = compute_snr_frequency_resolution_study(
            nfreq_values=HIGHER_NFREQ_VALUES,
            lmax=LMAX,
            nside=NSIDE_FAST,
            iter_sht=ITER_SHT,
            t_obs_yr=T_OBS_YR,
            omega_gw_h2=omega_gw_model,
            c_ell_gw=c_ell_gw,
        )
        plot_snr_vs_frequency_samples(freq_study, ells=(0, 1, 2, 3, 4), quantity="snr")
        show_snr_frequency_resolution_table(freq_study, ells=(0, 1, 2, 3, 4), quantity="snr")
