import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext, LogLocator

import notebook_code as nb


CHANNEL_PAIRS = ("AA", "EE", "TT", "AE", "AT", "ET")
AUTO_PAIRS = ("AA", "EE", "TT")
CROSS_PAIRS = ("AE", "AT", "ET")
PAIR_MULTIPLICITY = {
    "AA": 1.0,
    "EE": 1.0,
    "TT": 1.0,
    "AE": 2.0,
    "AT": 2.0,
    "ET": 2.0,
}
PAIR_GROUPS = {
    "AA": ("AA",),
    "EE": ("EE",),
    "TT": ("TT",),
    "AE": ("AE", "EA"),
    "AT": ("AT", "TA"),
    "ET": ("ET", "TE"),
}
PAIR_COLORS = {
    "AA": "#1f77b4",
    "EE": "#ff7f0e",
    "TT": "#2ca02c",
    "AE": "#d62728",
    "AT": "#9467bd",
    "ET": "#8c564b",
    "combined": "black",
    "signal": "#111111",
}
GROUP_COLORS = {
    "auto": "#1f77b4",
    "cross": "#d62728",
}
REPRESENTATIVE_RESPONSE_STYLE = {
    "AA": {
        "color": PAIR_COLORS["AA"],
        "lw": 2.6,
        "ls": "-",
        "marker": None,
        "zorder": 3,
    },
    "EE": {
        "color": PAIR_COLORS["EE"],
        "lw": 0.0,
        "ls": "None",
        "marker": "o",
        "ms": 4.5,
        "mfc": "white",
        "mec": PAIR_COLORS["EE"],
        "mew": 1.1,
        "zorder": 4,
    },
    "AE": {
        "color": PAIR_COLORS["AE"],
        "lw": 2.2,
        "ls": "-",
        "marker": None,
        "zorder": 2,
    },
}


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
        raise ValueError("c_ell_gw must be non-negative.")
    return value


def _evaluate_omega_gw_h2(f_grid, omega_gw_h2):
    if callable(omega_gw_h2):
        values = np.asarray(omega_gw_h2(f_grid), dtype=float)
    else:
        values = np.asarray(omega_gw_h2, dtype=float)
        if values.shape != f_grid.shape:
            raise ValueError(
                "If omega_gw_h2 is an array, it must have the same shape as f_grid."
            )
    return values


def _symmetrize_ordered_pair_dict(pair_dict):
    unique = {}
    for pair, ordered_names in PAIR_GROUPS.items():
        stacked = np.stack(
            [np.asarray(pair_dict[name], dtype=float) for name in ordered_names],
            axis=0,
        )
        unique[pair] = np.mean(stacked, axis=0)
    return unique


def combine_unique_pair_omegas(pair_omega_unique):
    template = np.asarray(next(iter(pair_omega_unique.values())), dtype=float)
    inv2 = np.zeros_like(template, dtype=float)
    for pair in CHANNEL_PAIRS:
        omega = np.asarray(pair_omega_unique[pair], dtype=float)
        mask = np.isfinite(omega) & (omega > 0.0)
        inv2[mask] += PAIR_MULTIPLICITY[pair] / omega[mask] ** 2
    out = np.full_like(inv2, np.inf, dtype=float)
    good = inv2 > 0.0
    out[good] = 1.0 / np.sqrt(inv2[good])
    return out


def compute_pair_inverse_variance_terms(pair_omega_unique):
    template = np.asarray(next(iter(pair_omega_unique.values())), dtype=float)
    total_inv2 = np.zeros_like(template, dtype=float)
    terms = {}
    for pair in CHANNEL_PAIRS:
        omega = np.asarray(pair_omega_unique[pair], dtype=float)
        term = np.zeros_like(template, dtype=float)
        mask = np.isfinite(omega) & (omega > 0.0)
        term[mask] = PAIR_MULTIPLICITY[pair] / omega[mask] ** 2
        terms[pair] = term
        total_inv2 += term
    return terms, total_inv2


def compute_pair_weights(pair_omega_unique):
    template = np.asarray(next(iter(pair_omega_unique.values())), dtype=float)
    numerators, total_inv2 = compute_pair_inverse_variance_terms(pair_omega_unique)
    weights = {}
    good = total_inv2 > 0.0
    for pair in CHANNEL_PAIRS:
        weight = np.zeros_like(template, dtype=float)
        weight[good] = numerators[pair][good] / total_inv2[good]
        weights[pair] = weight
    return weights


def compute_group_sums(pair_value_dict):
    template = np.asarray(next(iter(pair_value_dict.values())), dtype=float)
    groups = {}
    for name, pairs in (
        ("auto", AUTO_PAIRS),
        ("cross", CROSS_PAIRS),
    ):
        total = np.zeros_like(template, dtype=float)
        for pair in pairs:
            total += np.asarray(pair_value_dict[pair], dtype=float)
        groups[name] = total
    return groups


def compute_group_weights(pair_weight_dict):
    return compute_group_sums(pair_weight_dict)


def compute_weight_normalization_error(pair_weight_dict):
    total = np.zeros_like(
        np.asarray(next(iter(pair_weight_dict.values())), dtype=float)
    )
    for pair in CHANNEL_PAIRS:
        total += np.asarray(pair_weight_dict[pair], dtype=float)
    return float(np.max(np.abs(total - 1.0)))


def set_axes_style(ax, ylabel=None, xlabel="Frequency [Hz]", xlog=True, ylog=True):
    if xlog:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=100))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    if ylog:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=100))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.grid(True, which="both", linestyle=":", color="0.82")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)


def set_weight_axes_style(ax, ylabel=None, xlabel="Frequency [Hz]"):
    set_axes_style(ax, ylabel=ylabel, xlabel=xlabel, xlog=True, ylog=False)
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(0.0, color="0.88", lw=0.8, zorder=0)
    ax.axhline(1.0, color="0.88", lw=0.8, zorder=0)


def _frequency_bin_edges(frequency_hz):
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    if frequency_hz.ndim != 1 or frequency_hz.size < 2:
        raise ValueError("Need at least two frequency samples to build a heatmap.")
    if np.any(np.diff(frequency_hz) <= 0.0):
        raise ValueError("frequency_hz must be strictly increasing.")

    edges = np.empty(frequency_hz.size + 1, dtype=float)
    edges[1:-1] = np.sqrt(frequency_hz[:-1] * frequency_hz[1:])
    low_ratio = frequency_hz[1] / frequency_hz[0]
    high_ratio = frequency_hz[-1] / frequency_hz[-2]
    edges[0] = frequency_hz[0] / np.sqrt(low_ratio)
    edges[-1] = frequency_hz[-1] * np.sqrt(high_ratio)
    return edges


def _ell_bin_edges(ells):
    ells = np.asarray(ells, dtype=float)
    if ells.ndim != 1 or ells.size == 0:
        raise ValueError("Need at least one multipole to build a heatmap.")
    if np.any(np.diff(ells) <= 0.0):
        raise ValueError("ells must be strictly increasing.")

    if ells.size == 1:
        return np.array([ells[0] - 0.5, ells[0] + 0.5], dtype=float)

    edges = np.empty(ells.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (ells[:-1] + ells[1:])
    edges[0] = ells[0] - 0.5 * (ells[1] - ells[0])
    edges[-1] = ells[-1] + 0.5 * (ells[-1] - ells[-2])
    return edges


def _positive_lognorm(values):
    positive = np.asarray(values, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size == 0:
        raise ValueError("Heatmap values must contain at least one positive entry.")
    vmin = positive.min()
    vmax = positive.max()
    if np.isclose(vmin, vmax):
        vmax = vmin * (1.0 + 1e-12)
    return LogNorm(vmin=vmin, vmax=vmax), vmin


def pair_linestyle(ell, pair):
    auto_pair = pair in ("AA", "EE", "TT")
    return "--" if (ell % 2 == 1 and auto_pair) else "-"


def compute_lisa_multipole_sensitivity_detailed(
    f_grid,
    lmax=10,
    nside=32,
    iter_sht=1,
):
    """
    Compute the same LISA multipole sensitivities as ``notebook_code.py``,
    but retain the intermediate channel-pair response and noise ingredients
    needed for diagnostics.
    """

    f_grid = np.asarray(f_grid, dtype=float)
    if np.any(np.diff(f_grid) <= 0.0):
        raise ValueError("f_grid must be strictly increasing.")
    if lmax > 3 * nside - 1:
        raise ValueError(
            f"HEALPix sampling constraint: need lmax <= 3*nside-1, got lmax={lmax}, nside={nside}."
        )

    l_arm = nb.L_ARM
    r1 = np.array([0.0, 0.0, 0.0])
    r2 = np.array([l_arm, 0.0, 0.0])
    r3 = np.array([0.5 * l_arm, 0.5 * np.sqrt(3.0) * l_arm, 0.0])
    positions = [r1, r2, r3]

    def unit(vector):
        return vector / np.linalg.norm(vector)

    l12 = unit(r2 - r1)
    l13 = unit(r3 - r1)
    l21 = unit(r1 - r2)
    l23 = unit(r3 - r2)
    l31 = unit(r1 - r3)
    l32 = unit(r2 - r3)
    arms_by_sc = {0: (l12, l13), 1: (l23, l21), 2: (l31, l32)}

    pos_sec = [position / nb.C_LIGHT for position in positions]
    delta_sec = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            delta_sec[i, j] = pos_sec[i] - pos_sec[j]

    khat, theta, phi = nb.healpix_grid(nside)
    npix = khat.shape[0]
    e_plus, e_cross = nb.polarization_tensors_from_k(khat, theta, phi)

    mu_a = np.empty((3, npix), dtype=np.float64)
    mu_b = np.empty((3, npix), dtype=np.float64)
    gp_a = np.empty((3, npix), dtype=np.float64)
    gx_a = np.empty((3, npix), dtype=np.float64)
    gp_b = np.empty((3, npix), dtype=np.float64)
    gx_b = np.empty((3, npix), dtype=np.float64)

    for i in range(3):
        arm_a, arm_b = arms_by_sc[i]
        mu_a[i] = khat @ arm_a
        mu_b[i] = khat @ arm_b
        gp_a[i] = nb.g_pol_from_arm(arm_a, e_plus).real
        gx_a[i] = nb.g_pol_from_arm(arm_a, e_cross).real
        gp_b[i] = nb.g_pol_from_arm(arm_b, e_plus).real
        gx_b[i] = nb.g_pol_from_arm(arm_b, e_cross).real

    kdot_delta = np.zeros((3, 3, npix), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            kdot_delta[i, j] = khat @ delta_sec[i, j]
    kdot_flat = kdot_delta.reshape(9, npix)

    ordered_pairs = [(o, op) for o in nb.O_list for op in nb.O_list]
    ordered_labels = [f"{o}{op}" for o, op in ordered_pairs]
    mix = np.empty((9, 9), dtype=np.float64)
    for pidx, (o, op) in enumerate(ordered_pairs):
        c_o = nb.C_AET[nb.O_idx[o]]
        c_op = nb.C_AET[nb.O_idx[op]]
        mix[pidx, :] = (c_o[:, None] * c_op[None, :]).reshape(9)

    n_a = nb.N_tilde_AE(f_grid)
    n_e = n_a.copy()
    n_t = nb.N_tilde_T(f_grid)
    n_map = {"A": n_a, "E": n_e, "T": n_t}

    ordered_noise_geometric_mean = {}
    ordered_eq43_noise_prefactor = {}
    eq43_prefactor = (
        4.0 * np.pi ** 2 * np.sqrt(4.0 * np.pi) / (3.0 * nb.H0_OVER_h ** 2)
    )
    for label, (o, op) in zip(ordered_labels, ordered_pairs):
        noise_geom = np.sqrt(n_map[o] * n_map[op])
        ordered_noise_geometric_mean[label] = noise_geom
        ordered_eq43_noise_prefactor[label] = eq43_prefactor * f_grid ** 3 * noise_geom

    idx_mpos_by_ell = nb.precompute_alm_indices(lmax)[1]

    raw_total = {ell: np.zeros_like(f_grid) for ell in range(lmax + 1)}
    ordered_pair_omega = {
        ell: {label: np.zeros_like(f_grid) for label in ordered_labels}
        for ell in range(lmax + 1)
    }
    ordered_rtilde = {
        ell: {label: np.zeros_like(f_grid) for label in ordered_labels}
        for ell in range(lmax + 1)
    }

    rplus = np.empty((3, npix), dtype=np.complex128)
    rcross = np.empty((3, npix), dtype=np.complex128)
    pref_rlm = 1.0 / (8.0 * np.pi)

    for fi, frequency in enumerate(f_grid):
        u = frequency / nb.F_STAR
        transfer_a = nb.T_transfer(u, mu_a)
        transfer_b = nb.T_transfer(u, mu_b)

        rplus[:] = gp_a * transfer_a - gp_b * transfer_b
        rcross[:] = gx_a * transfer_a - gx_b * transfer_b
        pol_sum = (
            rplus[:, None, :] * np.conjugate(rplus[None, :, :])
            + rcross[:, None, :] * np.conjugate(rcross[None, :, :])
        )

        phase_flat = np.exp(-2j * np.pi * frequency * kdot_flat)
        maps_ij_flat = pref_rlm * pol_sum.reshape(9, npix) * phase_flat
        maps_pair = mix @ maps_ij_flat

        alms_a = []
        alms_c = []
        for pidx in range(9):
            pair_map = maps_pair[pidx]
            pair_real = pair_map.real.astype(np.float64, copy=False)
            pair_imag = pair_map.imag.astype(np.float64, copy=False)
            alm_real = hp.map2alm(pair_real, lmax=lmax, iter=iter_sht, pol=False)
            alm_imag = hp.map2alm(pair_imag, lmax=lmax, iter=iter_sht, pol=False)
            alms_a.append(alm_real + 1j * alm_imag)
            alms_c.append(alm_real - 1j * alm_imag)

        for ell in range(lmax + 1):
            idx0 = hp.Alm.getidx(lmax, ell, 0)
            idxm = idx_mpos_by_ell[ell]
            rtilde_pairs = np.empty(9, dtype=np.float64)

            for pidx in range(9):
                alm_a = alms_a[pidx]
                alm_c = alms_c[pidx]
                power = np.abs(alm_a[idx0]) ** 2
                if idxm.size > 0:
                    power += np.sum(np.abs(alm_a[idxm]) ** 2)
                    power += np.sum(np.abs(alm_c[idxm]) ** 2)
                rtilde_pairs[pidx] = np.sqrt(np.pi * power)

            rtilde_pairs = np.maximum(rtilde_pairs, 1e-45)

            omega_pairs = []
            for pidx, (o, op) in enumerate(ordered_pairs):
                label = ordered_labels[pidx]
                omega = nb.omega_channel_channel(
                    frequency,
                    rtilde_pairs[pidx],
                    n_map[o][fi],
                    n_map[op][fi],
                )
                ordered_pair_omega[ell][label][fi] = omega
                ordered_rtilde[ell][label][fi] = rtilde_pairs[pidx]
                omega_pairs.append(omega)

            raw_total[ell][fi] = nb.optimal_omega_from_ordered_pairs(omega_pairs)

    raw_pairs = {
        ell: _symmetrize_ordered_pair_dict(ordered_pair_omega[ell])
        for ell in range(lmax + 1)
    }
    rtilde_pairs = {
        ell: _symmetrize_ordered_pair_dict(ordered_rtilde[ell])
        for ell in range(lmax + 1)
    }
    pair_inverse_variance = {
        ell: compute_pair_inverse_variance_terms(raw_pairs[ell])[0]
        for ell in range(lmax + 1)
    }
    weights = {ell: compute_pair_weights(raw_pairs[ell]) for ell in range(lmax + 1)}
    group_inverse_variance = {
        ell: compute_group_sums(pair_inverse_variance[ell]) for ell in range(lmax + 1)
    }
    group_weights = {
        ell: compute_group_weights(weights[ell]) for ell in range(lmax + 1)
    }
    figure9_total = {ell: raw_total[ell] * nb.Y00 for ell in range(lmax + 1)}
    figure9_pairs = {
        ell: {
            pair: values * nb.Y00 for pair, values in raw_pairs[ell].items()
        }
        for ell in range(lmax + 1)
    }
    unique_noise_geometric_mean = _symmetrize_ordered_pair_dict(
        ordered_noise_geometric_mean
    )
    unique_eq43_noise_prefactor = _symmetrize_ordered_pair_dict(
        ordered_eq43_noise_prefactor
    )
    recombined_raw = {
        ell: combine_unique_pair_omegas(raw_pairs[ell]) for ell in range(lmax + 1)
    }
    recombination_relerr = {}
    weight_normalization_err = {}
    for ell in range(lmax + 1):
        denom = np.maximum(raw_total[ell], 1e-300)
        recombination_relerr[ell] = np.max(
            np.abs(recombined_raw[ell] - raw_total[ell]) / denom
        )
        weight_normalization_err[ell] = compute_weight_normalization_error(weights[ell])

    return {
        "f_grid": f_grid,
        "lmax": lmax,
        "nside": nside,
        "iter_sht": iter_sht,
        "raw_total": raw_total,
        "figure9_total": figure9_total,
        "raw_pairs": raw_pairs,
        "figure9_pairs": figure9_pairs,
        "rtilde_pairs": rtilde_pairs,
        "pair_inverse_variance": pair_inverse_variance,
        "group_inverse_variance": group_inverse_variance,
        "weights": weights,
        "group_weights": group_weights,
        "auto_weight": {ell: group_weights[ell]["auto"] for ell in range(lmax + 1)},
        "cross_weight": {ell: group_weights[ell]["cross"] for ell in range(lmax + 1)},
        "auto_inverse_variance": {
            ell: group_inverse_variance[ell]["auto"] for ell in range(lmax + 1)
        },
        "cross_inverse_variance": {
            ell: group_inverse_variance[ell]["cross"] for ell in range(lmax + 1)
        },
        "noise_geometric_mean": unique_noise_geometric_mean,
        "eq43_noise_prefactor": unique_eq43_noise_prefactor,
        "recombination_relerr": recombination_relerr,
        "weight_normalization_err": weight_normalization_err,
    }


def compute_channel_signal_noise_breakdown(
    f_grid,
    detailed,
    ell,
    omega_gw_h2,
    c_ell_gw=None,
):
    f_grid = np.asarray(f_grid, dtype=float)
    signal = np.sqrt(_resolve_c_ell_gw(c_ell_gw, ell)) * _evaluate_omega_gw_h2(
        f_grid,
        omega_gw_h2,
    )
    pair_integrand = {}
    fractional_contribution = {}
    total_density = np.zeros_like(f_grid, dtype=float)

    for pair in CHANNEL_PAIRS:
        pair_sensitivity = np.asarray(detailed["raw_pairs"][ell][pair], dtype=float)
        density = np.zeros_like(f_grid, dtype=float)
        mask = np.isfinite(pair_sensitivity) & (pair_sensitivity > 0.0)
        density[mask] = (
            PAIR_MULTIPLICITY[pair] * (signal[mask] / pair_sensitivity[mask]) ** 2
        )
        pair_integrand[pair] = density
        total_density += density

    good = total_density > 0.0
    for pair in CHANNEL_PAIRS:
        fraction = np.zeros_like(f_grid, dtype=float)
        fraction[good] = pair_integrand[pair][good] / total_density[good]
        fractional_contribution[pair] = fraction

    return {
        "signal": signal,
        "combined_sensitivity": np.asarray(detailed["raw_total"][ell], dtype=float),
        "pair_integrand": pair_integrand,
        "fractional_contribution": fractional_contribution,
        "total_integrand": total_density,
    }


def compute_snr_vs_mission_duration(
    f_grid,
    raw_sensitivity_by_ell,
    omega_gw_h2,
    mission_durations_yr=(0.5, 1.0, 2.0, 4.0),
    c_ell_gw=None,
    fmin=None,
    fmax=None,
):
    mission_durations_yr = np.asarray(mission_durations_yr, dtype=float)
    if mission_durations_yr.ndim != 1 or mission_durations_yr.size == 0:
        raise ValueError("mission_durations_yr must contain at least one duration.")
    if np.any(mission_durations_yr <= 0.0):
        raise ValueError("mission_durations_yr must be strictly positive.")
    if np.any(np.diff(mission_durations_yr) <= 0.0):
        raise ValueError("mission_durations_yr must be strictly increasing.")

    reference = nb.compute_snr_per_multipole_eq444(
        f_grid,
        raw_sensitivity_by_ell,
        omega_gw_h2,
        c_ell_gw=c_ell_gw,
        t_obs_yr=1.0,
        fmin=fmin,
        fmax=fmax,
    )

    snr2_by_ell = reference["snr2"][:, None] * mission_durations_yr[None, :]
    return {
        "mission_durations_yr": mission_durations_yr,
        "ells": reference["ells"],
        "snr2_by_ell": snr2_by_ell,
        "snr_by_ell": np.sqrt(snr2_by_ell),
        "total_snr2": np.sum(snr2_by_ell, axis=0),
        "total_snr": np.sqrt(np.sum(snr2_by_ell, axis=0)),
        "reference_snr": reference,
    }


def build_multipole_heatmap_data(
    f_grid,
    detailed,
    quantity="omega",
    omega_gw_h2=None,
    c_ell_gw=None,
    ells=None,
    fmin=None,
    fmax=None,
):
    f_grid = np.asarray(f_grid, dtype=float)
    if ells is None:
        ells = np.array(sorted(detailed["raw_total"].keys()), dtype=int)
    else:
        ells = np.array(sorted(int(ell) for ell in ells), dtype=int)
    if ells.size == 0:
        raise ValueError("Need at least one multipole to build a heatmap.")
    missing = [int(ell) for ell in ells if int(ell) not in detailed["raw_total"]]
    if missing:
        raise ValueError(f"Requested heatmap multipoles outside the computed range: {missing}")

    band = np.ones_like(f_grid, dtype=bool)
    if fmin is not None:
        band &= f_grid >= float(fmin)
    if fmax is not None:
        band &= f_grid <= float(fmax)
    if np.count_nonzero(band) < 2:
        raise ValueError("The selected frequency band must contain at least two samples.")

    frequency_band = f_grid[band]
    raw_sensitivity = {
        ell: np.asarray(detailed["raw_total"][ell], dtype=float)[band]
        for ell in ells
    }

    if quantity == "omega":
        values = np.vstack([raw_sensitivity[ell] for ell in ells])
        title = r"Effective noise level $\Omega_{{\rm GW},n}^{\ell}(f)\,h^2$"
        colorbar_label = r"$\Omega_{{\rm GW},n}^{\ell}(f)\,h^2$"
    elif quantity == "snr_density":
        if omega_gw_h2 is None:
            omega_gw_h2 = lambda f: nb.power_law_omega_gw_h2(  # noqa: E731
                f,
                amplitude_h2=1.0e-12,
                alpha=0.0,
                f_ref=1.0e-3,
            )
        snr_density = nb.compute_snr_per_multipole_eq444(
            f_grid,
            {ell: np.asarray(detailed["raw_total"][ell], dtype=float) for ell in ells},
            omega_gw_h2,
            c_ell_gw=c_ell_gw,
            t_obs_yr=1.0,
            fmin=fmin,
            fmax=fmax,
        )
        values = np.vstack(
            [np.asarray(snr_density["integrands"][ell], dtype=float)[band] for ell in ells]
        )
        title = (
            "Effective Eq. (4.44) SNR density per unit observing time "
            r"$\left[\sqrt{C_\ell^{\rm GW}}\Omega_{\rm GW}/\Omega_{{\rm GW},n}^{\ell}\right]^2$"
        )
        colorbar_label = (
            r"$\left[\sqrt{C_\ell^{\rm GW}}\Omega_{\rm GW}(f)\,"
            r"/\,\Omega_{{\rm GW},n}^{\ell}(f)\right]^2$"
        )
    else:
        raise ValueError("quantity must be either 'omega' or 'snr_density'.")

    return {
        "quantity": quantity,
        "frequency_hz": frequency_band,
        "ells": ells,
        "values": values,
        "title": title,
        "colorbar_label": colorbar_label,
    }


def plot_transfer_functions_vs_frequency(
    frequency_hz,
    mus=(-0.75, -0.25, 0.25, 0.75),
    output_path=None,
    show=True,
):
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    u = frequency_hz / nb.F_STAR
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)

    for mu in mus:
        label = rf"$\mu={mu:.2f}$"
        axes[0].semilogx(
            frequency_hz,
            np.abs(nb.M_transfer(u, mu)),
            lw=2.0,
            label=label,
        )
        axes[1].semilogx(
            frequency_hz,
            np.abs(nb.T_transfer(u, mu)),
            lw=2.0,
            label=label,
        )

    for ax in axes:
        ax.axvline(nb.F_STAR, color="0.45", lw=1.2, ls="--", alpha=0.8)
        ax.grid(True, which="both", linestyle=":", color="0.82")
        ax.set_xlim(frequency_hz[0], frequency_hz[-1])
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylim(bottom=0.0)

    axes[0].set_ylabel("Magnitude")
    axes[0].set_title(r"$|M(u,\mu)|$")
    axes[1].set_title(r"$|T(u,\mu)|$")
    axes[1].legend(loc="upper right", frameon=True, fontsize=10)
    fig.suptitle(
        "Finite-arm transfer functions versus frequency "
        rf"(vertical dashed line: $f_\star = {nb.F_STAR:.3e}\,\mathrm{{Hz}}$)"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def plot_channel_pair_sensitivity_overview(
    frequency_hz,
    detailed,
    ells=(1, 2, 3, 4),
    output_path=None,
    show=True,
):
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    ells = tuple(int(ell) for ell in ells)
    n_panels = len(ells)
    ncols = min(2, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.2 * ncols, 4.6 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for panel_index, (ax, ell) in enumerate(zip(axes.flat, ells)):
        row_index = panel_index // ncols
        for pair in CHANNEL_PAIRS:
            linestyle = pair_linestyle(ell, pair)
            ax.loglog(
                frequency_hz,
                detailed["figure9_pairs"][ell][pair],
                color=PAIR_COLORS[pair],
                lw=2.1 if linestyle == "-" else 1.7,
                ls=linestyle,
                alpha=0.95 if linestyle == "-" else 0.75,
                label=pair,
            )
        ax.loglog(
            frequency_hz,
            detailed["figure9_total"][ell],
            color=PAIR_COLORS["combined"],
            lw=2.3,
            label="combined",
        )
        set_axes_style(
            ax,
            ylabel=(
                r"$\Omega_{{\rm GW},n;OO'}^{\ell}(f)\,h^2 / \sqrt{4\pi}$"
                if panel_index % ncols == 0
                else None
            ),
            xlabel="Frequency [Hz]" if row_index == nrows - 1 else None,
        )
        ax.set_ylim(1e-12, 1e0)
        ax.set_title(rf"$\ell={ell}$")

    for ax in axes.flat[n_panels:]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=min(7, len(labels)),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        fontsize=9,
        frameon=True,
    )
    fig.suptitle(
        "Channel-pair sensitivity curves "
        "(odd-$\\ell$ auto channels dashed, black = optimal combination)",
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def plot_inverse_variance_channel_weights(
    frequency_hz,
    detailed,
    ells=(1, 2, 3, 4),
    output_path=None,
    show=True,
):
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    ells = tuple(int(ell) for ell in ells)
    n_panels = len(ells)
    ncols = min(2, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.2 * ncols, 4.6 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for panel_index, (ax, ell) in enumerate(zip(axes.flat, ells)):
        row_index = panel_index // ncols
        for pair in CHANNEL_PAIRS:
            ax.semilogx(
                frequency_hz,
                detailed["weights"][ell][pair],
                color=PAIR_COLORS[pair],
                lw=2.0,
                label=pair,
            )
        set_weight_axes_style(
            ax,
            ylabel=(
                "Fractional inverse-variance weight"
                if panel_index % ncols == 0
                else None
            ),
            xlabel="Frequency [Hz]" if row_index == nrows - 1 else None,
        )
        ax.set_title(
            rf"$\ell={ell}$"
            + "\n"
            + rf"$\max|1-\sum_p w_p|={detailed['weight_normalization_err'][ell]:.1e}$"
        )

    for ax in axes.flat[n_panels:]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=min(6, len(labels)),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        fontsize=9,
        frameon=True,
    )
    fig.suptitle(
        "How much each channel pair contributes to the final combination",
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def plot_auto_vs_cross_channel_contribution(
    frequency_hz,
    detailed,
    ells=(1, 2),
    output_path=None,
    show=True,
):
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    ells = tuple(int(ell) for ell in ells)
    n_panels = len(ells)
    ncols = min(2, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.0 * ncols, 4.4 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for panel_index, (ax, ell) in enumerate(zip(axes.flat, ells)):
        row_index = panel_index // ncols
        grouped = detailed["group_inverse_variance"][ell]
        ax.loglog(
            frequency_hz,
            grouped["auto"],
            color=GROUP_COLORS["auto"],
            lw=2.4,
            label=r"auto ($AA+EE+TT$)",
        )
        ax.loglog(
            frequency_hz,
            grouped["cross"],
            color=GROUP_COLORS["cross"],
            lw=2.4,
            label=r"cross ($AE+AT+ET$)",
        )
        set_axes_style(
            ax,
            ylabel=(
                "Absolute inverse-variance contribution"
                if panel_index % ncols == 0
                else None
            ),
            xlabel="Frequency [Hz]" if row_index == nrows - 1 else None,
            xlog=True,
            ylog=True,
        )
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=8))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        ax.set_title(rf"$\ell={ell}$")

    for ax in axes.flat[n_panels:]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=min(2, len(labels)),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        fontsize=10,
        frameon=True,
    )
    fig.suptitle(
        "Absolute auto-channel versus cross-channel contribution",
        y=0.985,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def plot_representative_pair_responses(
    frequency_hz,
    detailed,
    ells=(0, 2, 3, 4),
    pairs=("AA", "EE", "AE"),
    output_path=None,
    show=True,
):
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    ells = tuple(int(ell) for ell in ells)
    pairs = tuple(str(pair) for pair in pairs)
    n_panels = len(ells)
    ncols = min(2, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.0 * ncols, 4.4 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for panel_index, (ax, ell) in enumerate(zip(axes.flat, ells)):
        row_index = panel_index // ncols
        for pair in pairs:
            style = dict(REPRESENTATIVE_RESPONSE_STYLE.get(pair, {}))
            if style.get("marker") is not None:
                style["markevery"] = max(1, len(frequency_hz) // 14)
            ax.loglog(
                frequency_hz,
                detailed["rtilde_pairs"][ell][pair],
                label=pair,
                **style,
            )
        set_axes_style(
            ax,
            ylabel=(
                r"$\widetilde{R}_{OO'}^{\ell}(f)$"
                if panel_index % ncols == 0
                else None
            ),
            xlabel="Frequency [Hz]" if row_index == nrows - 1 else None,
        )
        ax.set_title(rf"$\ell={ell}$")

    for ax in axes.flat[n_panels:]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=min(len(labels), 3),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        fontsize=10,
        frameon=True,
    )
    fig.suptitle("Rotationally invariant channel-pair responses", y=0.985)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def plot_signal_detector_noise_breakdown(
    frequency_hz,
    detailed,
    ell=2,
    omega_gw_h2=None,
    c_ell_gw=None,
    output_path=None,
    show=True,
):
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    ell = int(ell)
    if omega_gw_h2 is None:
        omega_gw_h2 = lambda f: nb.power_law_omega_gw_h2(  # noqa: E731
            f,
            amplitude_h2=1.0e-12,
            alpha=0.0,
            f_ref=1.0e-3,
        )

    budget = compute_channel_signal_noise_breakdown(
        frequency_hz,
        detailed,
        ell=ell,
        omega_gw_h2=omega_gw_h2,
        c_ell_gw=c_ell_gw,
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    signal = budget["signal"]

    axes[0, 0].loglog(
        frequency_hz,
        signal,
        color=PAIR_COLORS["signal"],
        lw=2.6,
        label=r"signal $\sqrt{C_\ell^{\rm GW}}\,\Omega_{\rm GW}(f)\,h^2$",
    )
    axes[0, 0].loglog(
        frequency_hz,
        budget["combined_sensitivity"],
        color="0.35",
        lw=2.4,
        ls="--",
        label=r"combined noise-equivalent $\Omega_{{\rm GW},n}^{\ell}(f)\,h^2$",
    )
    set_axes_style(
        axes[0, 0],
        ylabel=r"$\Omega\,h^2$",
        xlabel=None,
    )
    axes[0, 0].set_title("Injected signal versus combined effective noise")
    axes[0, 0].legend(fontsize=9, frameon=True)

    for pair in CHANNEL_PAIRS:
        axes[0, 1].loglog(
            frequency_hz,
            detailed["rtilde_pairs"][ell][pair],
            color=PAIR_COLORS[pair],
            lw=2.0,
            label=pair,
        )
    set_axes_style(
        axes[0, 1],
        ylabel=rf"$\widetilde{{R}}_{{OO'}}^{{{ell}}}(f)$",
        xlabel=None,
    )
    axes[0, 1].set_title("Detector response by channel pair")

    for pair in CHANNEL_PAIRS:
        axes[1, 0].loglog(
            frequency_hz,
            detailed["eq43_noise_prefactor"][pair],
            color=PAIR_COLORS[pair],
            lw=2.0,
            label=pair,
        )
    set_axes_style(
        axes[1, 0],
        ylabel=rf"Eq. (4.43) noise term $\propto f^3\sqrt{{\tilde N_O\tilde N_{{O'}}}}$",
        xlabel="Frequency [Hz]",
    )
    axes[1, 0].set_title("Noise contribution before dividing by response")

    for pair in CHANNEL_PAIRS:
        axes[1, 1].semilogx(
            frequency_hz,
            budget["fractional_contribution"][pair],
            color=PAIR_COLORS[pair],
            lw=2.0,
            label=pair,
        )
    set_weight_axes_style(
        axes[1, 1],
        ylabel=r"Fraction of total $(S/N)^2$ density",
        xlabel="Frequency [Hz]",
    )
    axes[1, 1].set_title("Which channels dominate the final detection band")

    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=min(6, len(labels)),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        fontsize=9,
        frameon=True,
    )
    fig.suptitle(
        "Detector, signal, and noise decomposition "
        + rf"for $\ell={ell}$ with "
        + r"$\Omega_{{\rm GW},n}^{\ell} \propto "
        + r"f^3\sqrt{\tilde N_O\tilde N_{O'}}/\widetilde{R}_{OO'}^\ell$",
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes, budget


def plot_snr_vs_mission_duration(
    frequency_hz,
    detailed,
    mission_durations_yr=(0.5, 1.0, 2.0, 4.0),
    ells=(0, 1, 2, 3, 4),
    omega_gw_h2=None,
    c_ell_gw=None,
    fmin=None,
    fmax=None,
    include_total=True,
    output_path=None,
    show=True,
):
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    ells = tuple(int(ell) for ell in ells)
    if omega_gw_h2 is None:
        omega_gw_h2 = lambda f: nb.power_law_omega_gw_h2(  # noqa: E731
            f,
            amplitude_h2=1.0e-12,
            alpha=0.0,
            f_ref=1.0e-3,
        )

    snr_summary = compute_snr_vs_mission_duration(
        frequency_hz,
        detailed["raw_total"],
        omega_gw_h2=omega_gw_h2,
        mission_durations_yr=mission_durations_yr,
        c_ell_gw=c_ell_gw,
        fmin=fmin,
        fmax=fmax,
    )
    available_indices = {int(ell): idx for idx, ell in enumerate(snr_summary["ells"])}
    missing = [ell for ell in ells if ell not in available_indices]
    if missing:
        raise ValueError(
            f"Requested SNR curves for multipoles outside the computed range: {missing}"
        )
    selected_indices = [available_indices[ell] for ell in ells]
    total_snr = np.sqrt(np.sum(snr_summary["snr2_by_ell"][selected_indices], axis=0))

    fig, ax = plt.subplots(figsize=(8.6, 5.3))
    durations = snr_summary["mission_durations_yr"]

    for ell in ells:
        idx = available_indices[ell]
        ax.plot(
            durations,
            snr_summary["snr_by_ell"][idx],
            marker="o",
            ms=5.0,
            lw=2.0,
            label=rf"$\ell={ell}$",
        )

    if include_total:
        ax.plot(
            durations,
            total_snr,
            color="black",
            marker="o",
            ms=5.5,
            lw=2.7,
            label=r"quadrature total",
        )

    ax.set_xticks(durations)
    ax.set_xlabel("Mission duration [yr]")
    ax.set_ylabel("SNR")
    ax.grid(True, which="both", linestyle=":", color="0.82")

    positive = np.concatenate(
        [
            snr_summary["snr_by_ell"][selected_indices].ravel(),
            total_snr if include_total else np.array([], dtype=float),
        ]
    )
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size > 0 and positive.max() / positive.min() > 20.0:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=8))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))

    ax.legend(ncol=min(4, len(ells) + int(include_total)), frameon=True, fontsize=9)
    ax.set_title(
        "Per-multipole SNR versus mission duration "
        + r"(Eq. (4.44), fixed frequency band and signal model)"
    )
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax, snr_summary


def plot_multipole_frequency_heatmap(
    frequency_hz,
    detailed,
    quantity="omega",
    omega_gw_h2=None,
    c_ell_gw=None,
    ells=None,
    fmin=None,
    fmax=None,
    cmap="magma",
    output_path=None,
    show=True,
):
    heatmap = build_multipole_heatmap_data(
        frequency_hz,
        detailed,
        quantity=quantity,
        omega_gw_h2=omega_gw_h2,
        c_ell_gw=c_ell_gw,
        ells=ells,
        fmin=fmin,
        fmax=fmax,
    )

    norm, floor = _positive_lognorm(heatmap["values"])
    values = np.array(heatmap["values"], dtype=float, copy=True)
    bad = ~np.isfinite(values) | (values <= 0.0)
    values[bad] = floor

    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    mesh = ax.pcolormesh(
        _frequency_bin_edges(heatmap["frequency_hz"]),
        _ell_bin_edges(heatmap["ells"]),
        values,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )
    ax.set_xscale("log")
    ax.set_xlim(heatmap["frequency_hz"][0], heatmap["frequency_hz"][-1])
    ax.set_yticks(heatmap["ells"])
    ax.set_ylabel(r"Multipole $\ell$")
    ax.set_xlabel("Frequency [Hz]")
    ax.axvline(nb.F_STAR, color="white", lw=1.0, ls="--", alpha=0.9)
    ax.set_title(heatmap["title"])

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(heatmap["colorbar_label"])
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax, heatmap
