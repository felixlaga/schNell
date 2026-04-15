import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


C_LIGHT = 299_792_458.0
C_AET = np.array(
    [
        [-1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)],
        [1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0)],
        [1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
    ],
    dtype=float,
)
DEFAULT_UNEQUAL_ARMS_M = (2.50e9, 2.475e9, 2.525e9)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "noise_plots"
XYZ_LABELS = ("X", "Y", "Z")
AET_LABELS = ("A", "E", "T")
SOURCE_ORDER = (
    "op1",
    "op1p",
    "op2",
    "op2p",
    "op3",
    "op3p",
    "pm1",
    "pm1p",
    "pm2",
    "pm2p",
    "pm3",
    "pm3p",
)


@dataclass(frozen=True)
class MissionNoiseModel:
    """Mission-specific parameters for the Appendix-B tilde-noise curves."""

    name: str
    arm_length_m: float
    p_oms_pm: float
    a_acc_fm: float
    oms_break_hz: float = 2.0e-3
    acc_low_break_hz: float = 0.4e-3
    acc_high_break_hz: float = 8.0e-3

    @property
    def f_star_hz(self) -> float:
        return C_LIGHT / (2.0 * np.pi * self.arm_length_m)


LISA = MissionNoiseModel(
    name="LISA",
    arm_length_m=2.5e9,
    p_oms_pm=15.0,
    a_acc_fm=3.0,
)


def s_oms(frequency_hz, model):
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    return (model.p_oms_pm * 1.0e-12) ** 2 * (
        1.0 + (model.oms_break_hz / frequency_hz) ** 4
    )


def s_acc(frequency_hz, model):
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    return (model.a_acc_fm * 1.0e-15) ** 2 * (
        1.0 + (model.acc_low_break_hz / frequency_hz) ** 2
    ) * (1.0 + (frequency_hz / model.acc_high_break_hz) ** 4)


def p_ims(frequency_hz, model, arm_length_m):
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    return s_oms(frequency_hz, model) / float(arm_length_m) ** 2


def p_acc(frequency_hz, model, arm_length_m):
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    return (
        s_acc(frequency_hz, model)
        / float(arm_length_m) ** 2
        * (1.0 / (2.0 * np.pi * frequency_hz)) ** 4
    )


def n_tilde_ae(frequency_hz, model):
    """Appendix-B tilde-noise for the A/E channels."""

    frequency_hz = np.asarray(frequency_hz, dtype=float)
    x = frequency_hz / model.f_star_hz
    cosx = np.cos(x)
    displacement_noise = s_oms(frequency_hz, model) / model.arm_length_m**2
    acceleration_noise = s_acc(frequency_hz, model) / model.arm_length_m**2
    return 0.5 * (2.0 + cosx) * displacement_noise + 2.0 * (
        1.0 + cosx + cosx**2
    ) * acceleration_noise * (1.0 / (2.0 * np.pi * frequency_hz)) ** 4


def n_tilde_t(frequency_hz, model):
    """Appendix-B tilde-noise for the T channel."""

    frequency_hz = np.asarray(frequency_hz, dtype=float)
    x = frequency_hz / model.f_star_hz
    one_minus_cos = 1.0 - np.cos(x)
    displacement_noise = s_oms(frequency_hz, model) / model.arm_length_m**2
    acceleration_noise = s_acc(frequency_hz, model) / model.arm_length_m**2
    return one_minus_cos * displacement_noise + 2.0 * (
        one_minus_cos**2
    ) * acceleration_noise * (1.0 / (2.0 * np.pi * frequency_hz)) ** 4


def default_frequency_grid(fmin_hz=1.0e-4, fmax_hz=1.0, nfreq=4000):
    return np.geomspace(fmin_hz, fmax_hz, nfreq)


def compute_reference_curves(frequency_hz):
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    return {
        "lisa_ae": n_tilde_ae(frequency_hz, LISA),
        "lisa_t": n_tilde_t(frequency_hz, LISA),
    }


def unequal_arm_f_star_hz(arm_lengths_m):
    arm_lengths_m = np.asarray(arm_lengths_m, dtype=float)
    return C_LIGHT / (2.0 * np.pi * np.mean(arm_lengths_m))


def tdi_weight(frequency_hz, arm_lengths_m, tdi):
    """
    Return the notebook's |W|^2 weighting for the requested TDI version.

    For unequal arms we use the mean arm length in f_* so that the
    TDI-1.5 option follows the same convention already used in the local
    notebooks.
    """

    frequency_hz = np.asarray(frequency_hz, dtype=float)
    if tdi == "1.0":
        return np.ones_like(frequency_hz)
    if tdi == "1.5":
        f_star_hz = unequal_arm_f_star_hz(arm_lengths_m)
        return 4.0 * np.sin(frequency_hz / f_star_hz) ** 2
    raise ValueError("tdi must be '1.0' or '1.5'")


def _parse_arm_lengths(arm_lengths_m):
    values = np.asarray(arm_lengths_m, dtype=float)
    if values.shape != (3,):
        raise ValueError("arm_lengths_m must contain exactly three values: L1,L2,L3.")
    return values


def _delay(coefficients, delay_factor):
    return coefficients * delay_factor[:, None]


def _empty_eta_dict(nfreq):
    return {
        key: np.zeros((nfreq, len(SOURCE_ORDER)), dtype=complex)
        for key in ("eta1", "eta1p", "eta2", "eta2p", "eta3", "eta3p")
    }


def _source_noise_amplitudes(frequency_hz, arm_lengths_m, model, component):
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    arm_lengths_m = _parse_arm_lengths(arm_lengths_m)
    nfreq = frequency_hz.size

    # Source-to-link map inferred from the eta_i / eta_i' building blocks.
    source_arm_lengths = np.array(
        [
            arm_lengths_m[2],  # op1
            arm_lengths_m[1],  # op1'
            arm_lengths_m[0],  # op2
            arm_lengths_m[2],  # op2'
            arm_lengths_m[1],  # op3
            arm_lengths_m[0],  # op3'
            arm_lengths_m[2],  # pm1
            arm_lengths_m[1],  # pm1'
            arm_lengths_m[0],  # pm2
            arm_lengths_m[2],  # pm2'
            arm_lengths_m[1],  # pm3
            arm_lengths_m[0],  # pm3'
        ],
        dtype=float,
    )

    amplitudes = np.zeros((nfreq, len(SOURCE_ORDER)), dtype=float)

    if component in ("total", "oms"):
        amplitudes[:, :6] = np.sqrt(s_oms(frequency_hz, model))[:, None] / source_arm_lengths[
            None, :6
        ]
    if component in ("total", "acc"):
        amplitudes[:, 6:] = (
            np.sqrt(s_acc(frequency_hz, model))[:, None]
            * (1.0 / (2.0 * np.pi * frequency_hz))[:, None] ** 2
            / source_arm_lengths[None, 6:]
        )
    if component not in ("total", "oms", "acc"):
        raise ValueError("component must be 'total', 'oms', or 'acc'.")

    return amplitudes


def _eta_noise_coefficients(frequency_hz, arm_lengths_m, model, component):
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    arm_lengths_m = _parse_arm_lengths(arm_lengths_m)
    nfreq = frequency_hz.size
    eta = _empty_eta_dict(nfreq)

    phase_1 = np.exp(-1j * 2.0 * np.pi * frequency_hz * arm_lengths_m[0] / C_LIGHT)
    phase_2 = np.exp(-1j * 2.0 * np.pi * frequency_hz * arm_lengths_m[1] / C_LIGHT)
    phase_3 = np.exp(-1j * 2.0 * np.pi * frequency_hz * arm_lengths_m[2] / C_LIGHT)

    eta["eta1"][:, 0] = 1.0
    eta["eta1"][:, 9] = phase_3
    eta["eta1"][:, 6] = -1.0

    eta["eta1p"][:, 1] = 1.0
    eta["eta1p"][:, 7] = 1.0
    eta["eta1p"][:, 10] = -phase_2

    eta["eta2"][:, 2] = 1.0
    eta["eta2"][:, 11] = phase_1
    eta["eta2"][:, 8] = -1.0

    eta["eta2p"][:, 3] = 1.0
    eta["eta2p"][:, 9] = 1.0
    eta["eta2p"][:, 6] = -phase_3

    eta["eta3"][:, 4] = 1.0
    eta["eta3"][:, 7] = phase_2
    eta["eta3"][:, 10] = -1.0

    eta["eta3p"][:, 5] = 1.0
    eta["eta3p"][:, 11] = 1.0
    eta["eta3p"][:, 8] = -phase_1

    amplitudes = _source_noise_amplitudes(frequency_hz, arm_lengths_m, model, component)
    for key in eta:
        eta[key] *= amplitudes

    return eta, phase_1, phase_2, phase_3


def unequal_arm_xyz_coefficients(
    frequency_hz,
    arm_lengths_m=DEFAULT_UNEQUAL_ARMS_M,
    model=LISA,
    tdi="1.0",
    component="total",
):
    """
    Build raw unequal-arm X/Y/Z noise coefficients.

    The first-generation unequal-arm Michelson combinations are taken from
    Eq. (74) of Tinto & Dhurandhar (Living Reviews in Relativity 24, 1;
    https://link.springer.com/article/10.1007/s41114-020-00029-6). The
    optional TDI-1.5 branch follows the notebook's local |W|^2 convention.
    """

    frequency_hz = np.asarray(frequency_hz, dtype=float)
    arm_lengths_m = _parse_arm_lengths(arm_lengths_m)
    eta, phase_1, phase_2, phase_3 = _eta_noise_coefficients(
        frequency_hz,
        arm_lengths_m,
        model,
        component,
    )

    x_arm_a = eta["eta1"] + _delay(eta["eta2p"], phase_3)
    x_arm_b = eta["eta1p"] + _delay(eta["eta3"], phase_2)
    y_arm_a = eta["eta2"] + _delay(eta["eta3p"], phase_1)
    y_arm_b = eta["eta2p"] + _delay(eta["eta1"], phase_3)
    z_arm_a = eta["eta3"] + _delay(eta["eta1p"], phase_2)
    z_arm_b = eta["eta3p"] + _delay(eta["eta2"], phase_1)

    x_coeff = (phase_2[:, None] ** 2 - 1.0) * x_arm_a - (
        phase_3[:, None] ** 2 - 1.0
    ) * x_arm_b
    y_coeff = (phase_3[:, None] ** 2 - 1.0) * y_arm_a - (
        phase_1[:, None] ** 2 - 1.0
    ) * y_arm_b
    z_coeff = (phase_1[:, None] ** 2 - 1.0) * z_arm_a - (
        phase_2[:, None] ** 2 - 1.0
    ) * z_arm_b

    coefficients = np.stack([x_coeff, y_coeff, z_coeff], axis=1)
    coefficients *= np.sqrt(tdi_weight(frequency_hz, arm_lengths_m, tdi))[:, None, None]
    return coefficients


def unequal_arm_xyz_covariance(
    frequency_hz,
    arm_lengths_m=DEFAULT_UNEQUAL_ARMS_M,
    model=LISA,
    tdi="1.0",
    component="total",
):
    coefficients = unequal_arm_xyz_coefficients(
        frequency_hz,
        arm_lengths_m=arm_lengths_m,
        model=model,
        tdi=tdi,
        component=component,
    )
    return np.einsum("fcs,fds->fcd", coefficients, np.conjugate(coefficients), optimize=True)


def unequal_arm_aet_covariance(
    frequency_hz,
    arm_lengths_m=DEFAULT_UNEQUAL_ARMS_M,
    model=LISA,
    tdi="1.0",
    component="total",
):
    xyz_covariance = unequal_arm_xyz_covariance(
        frequency_hz,
        arm_lengths_m=arm_lengths_m,
        model=model,
        tdi=tdi,
        component=component,
    )
    return (
        np.einsum("ai,fij,bj->fab", C_AET, xyz_covariance, C_AET, optimize=True) / 4.0
    )


def noise_component_curves(frequency_hz, model=LISA, arm_length_m=None):
    if arm_length_m is None:
        arm_length_m = model.arm_length_m
    return {
        "ims": p_ims(frequency_hz, model, arm_length_m),
        "acc": p_acc(frequency_hz, model, arm_length_m),
    }


def _auto_curves_from_covariance(covariance):
    return np.real(np.diagonal(covariance, axis1=1, axis2=2))


def _arm_length_label(arm_lengths_m):
    arm_lengths_m = _parse_arm_lengths(arm_lengths_m)
    return ", ".join(
        f"L{i + 1}={arm_length_m / 1.0e9:.3f} Gm"
        for i, arm_length_m in enumerate(arm_lengths_m)
    )


def _set_log_axes(ax, frequency_hz, values, ylabel=None):
    positive_values = np.asarray(values, dtype=float)
    positive_values = positive_values[np.isfinite(positive_values) & (positive_values > 0.0)]
    if positive_values.size:
        ax.set_ylim(0.75 * positive_values.min(), 1.5 * positive_values.max())
    ax.set_xlim(frequency_hz[0], frequency_hz[-1])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", color="0.8", alpha=0.75)
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.set_xlabel("Frequency [Hz]", fontsize=14)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=14)


def plot_aet_noise_curves(
    frequency_hz=None,
    output_path=None,
    show=True,
    plot_sqrt=True,
):
    """
    Plot the LISA tilde-noise curves for the A/E and T channels.

    Returns
    -------
    fig, ax, curves
        Matplotlib figure/axes and a dict with the two curve arrays.
    """

    if frequency_hz is None:
        frequency_hz = default_frequency_grid()

    curves = compute_reference_curves(frequency_hz)
    curves_to_plot = {
        key: np.sqrt(value) if plot_sqrt else value for key, value in curves.items()
    }

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.loglog(
        frequency_hz,
        curves_to_plot["lisa_ae"],
        color="tab:blue",
        lw=2.4,
        label=r"$\tilde N^{\mathrm{LISA}}_{A/E}$",
    )
    ax.loglog(
        frequency_hz,
        curves_to_plot["lisa_t"],
        color="tab:red",
        lw=2.4,
        label=r"$\tilde N^{\mathrm{LISA}}_{T}$",
    )

    ax.set_xlim(frequency_hz[0], frequency_hz[-1])
    ax.set_ylim(5.0e-24 if plot_sqrt else 1.0e-45, 1.0e-17 if plot_sqrt else 1.0e-33)
    ax.set_xlabel("Frequency [Hz]", fontsize=16)
    ax.set_ylabel(
        r"$\sqrt{\tilde N}\,[\mathrm{Hz}^{-1/2}]$"
        if plot_sqrt
        else r"$\tilde N\,[\mathrm{Hz}^{-1}]$",
        fontsize=16,
    )
    ax.set_title("Equal-arm A/E/T noise", fontsize=21)
    ax.grid(True, which="both", color="0.75", alpha=0.7)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(loc="upper right", fontsize=14, frameon=True)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax, curves


def plot_noise_components(
    frequency_hz=None,
    arm_length_m=None,
    model=LISA,
    output_path=None,
    show=True,
):
    if frequency_hz is None:
        frequency_hz = default_frequency_grid(fmin_hz=3.0e-5, fmax_hz=0.5)
    curves = noise_component_curves(frequency_hz, model=model, arm_length_m=arm_length_m)

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.loglog(
        frequency_hz,
        curves["acc"],
        color="blue",
        lw=2.0,
        label=r"$P_{\mathrm{acc}}$",
    )
    ax.loglog(
        frequency_hz,
        curves["ims"],
        color="magenta",
        lw=2.0,
        label=r"$P_{\mathrm{IMS}}$",
    )
    _set_log_axes(ax, frequency_hz, np.concatenate(list(curves.values())), ylabel=r"PSD [Hz$^{-1}$]")
    ax.set_title("Noise components", fontsize=21)
    ax.legend(loc="upper center", fontsize=14, frameon=True)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax, curves


def _plot_channel_panels(
    frequency_hz,
    channel_covariance_builder,
    channel_labels,
    title_root,
    arm_lengths_m,
    tdi_versions,
    model,
    output_path=None,
    show=True,
):
    tdi_versions = tuple(tdi_versions)
    ncols = len(tdi_versions)
    fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols, 5.5), sharey=True)
    if ncols == 1:
        axes = [axes]

    colors = ("tab:blue", "tab:orange", "tab:green")
    line_styles = ("-", "--", ":")
    auto_curves_by_tdi = {}

    for ax, tdi in zip(axes, tdi_versions):
        covariance = channel_covariance_builder(
            frequency_hz,
            arm_lengths_m=arm_lengths_m,
            model=model,
            tdi=tdi,
            component="total",
        )
        auto_curves = _auto_curves_from_covariance(covariance)
        auto_curves_by_tdi[tdi] = auto_curves
        for color, line_style, label, curve in zip(
            colors,
            line_styles,
            channel_labels,
            auto_curves.T,
        ):
            ax.loglog(
                frequency_hz,
                curve,
                color=color,
                ls=line_style,
                lw=2.2,
                label=label,
            )
        _set_log_axes(ax, frequency_hz, auto_curves.ravel(), ylabel=r"PSD [Hz$^{-1}$]")
        ax.set_title(f"TDI {tdi}", fontsize=17)
        ax.legend(loc="upper right", fontsize=12, frameon=True)

    fig.suptitle(title_root, fontsize=20, y=0.98)
    fig.text(0.5, 0.01, _arm_length_label(arm_lengths_m), ha="center", fontsize=11)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig, axes, auto_curves_by_tdi


def plot_unequal_xyz_noise(
    frequency_hz=None,
    arm_lengths_m=DEFAULT_UNEQUAL_ARMS_M,
    tdi_versions=("1.0", "1.5"),
    model=LISA,
    output_path=None,
    show=True,
):
    if frequency_hz is None:
        frequency_hz = default_frequency_grid()
    return _plot_channel_panels(
        frequency_hz=frequency_hz,
        channel_covariance_builder=unequal_arm_xyz_covariance,
        channel_labels=XYZ_LABELS,
        title_root="Unequal-arm XYZ noise",
        arm_lengths_m=arm_lengths_m,
        tdi_versions=tdi_versions,
        model=model,
        output_path=output_path,
        show=show,
    )


def plot_unequal_aet_noise(
    frequency_hz=None,
    arm_lengths_m=DEFAULT_UNEQUAL_ARMS_M,
    tdi_versions=("1.0", "1.5"),
    model=LISA,
    output_path=None,
    show=True,
):
    if frequency_hz is None:
        frequency_hz = default_frequency_grid()
    return _plot_channel_panels(
        frequency_hz=frequency_hz,
        channel_covariance_builder=unequal_arm_aet_covariance,
        channel_labels=AET_LABELS,
        title_root="Unequal-arm A/E/T noise",
        arm_lengths_m=arm_lengths_m,
        tdi_versions=tdi_versions,
        model=model,
        output_path=output_path,
        show=show,
    )


def _resolve_output_path(output, output_dir, default_name):
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / default_name
    return None


def _parse_arm_length_argument(raw_value):
    parts = [part.strip() for part in raw_value.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("Expected --arm-lengths to contain three comma-separated values.")
    return tuple(float(part) for part in parts)


def main():
    parser = argparse.ArgumentParser(description="Plot LISA noise curves and unequal-arm PSDs.")
    parser.add_argument(
        "--plot",
        choices=("equal-aet", "components", "unequal-xyz", "unequal-aet", "all"),
        default="equal-aet",
        help="Which figure family to generate.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path where a single figure should be written.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where default-named figures should be written.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Plot the equal-arm A/E/T PSDs instead of their square roots.",
    )
    parser.add_argument(
        "--arm-lengths",
        default=",".join(f"{value:.6e}" for value in DEFAULT_UNEQUAL_ARMS_M),
        help="Comma-separated unequal arm lengths in meters, e.g. 2.5e9,2.475e9,2.525e9.",
    )
    parser.add_argument(
        "--tdi",
        nargs="+",
        choices=("1.0", "1.5"),
        default=("1.0", "1.5"),
        help="TDI versions to include in the unequal-arm panel plots.",
    )
    parser.add_argument("--fmin", type=float, default=1.0e-4, help="Minimum frequency in Hz.")
    parser.add_argument("--fmax", type=float, default=1.0, help="Maximum frequency in Hz.")
    parser.add_argument("--nfreq", type=int, default=4000, help="Number of frequency samples.")
    args = parser.parse_args()

    if args.plot == "all" and args.output is not None:
        parser.error("--output can only be used with a single --plot choice. Use --output-dir with --plot all.")

    frequency_hz = default_frequency_grid(args.fmin, args.fmax, args.nfreq)
    arm_lengths_m = _parse_arm_length_argument(args.arm_lengths)
    show_figures = False

    if args.plot in ("equal-aet", "all"):
        plot_aet_noise_curves(
            frequency_hz=frequency_hz,
            output_path=_resolve_output_path(args.output if args.plot == "equal-aet" else None, args.output_dir, "equal_aet_noise.png"),
            show=show_figures,
            plot_sqrt=not args.raw,
        )

    if args.plot in ("components", "all"):
        plot_noise_components(
            frequency_hz=frequency_hz,
            output_path=_resolve_output_path(args.output if args.plot == "components" else None, args.output_dir, "noise_components.png"),
            show=show_figures,
        )

    if args.plot in ("unequal-xyz", "all"):
        plot_unequal_xyz_noise(
            frequency_hz=frequency_hz,
            arm_lengths_m=arm_lengths_m,
            tdi_versions=args.tdi,
            output_path=_resolve_output_path(args.output if args.plot == "unequal-xyz" else None, args.output_dir, "unequal_xyz_noise.png"),
            show=show_figures,
        )

    if args.plot in ("unequal-aet", "all"):
        plot_unequal_aet_noise(
            frequency_hz=frequency_hz,
            arm_lengths_m=arm_lengths_m,
            tdi_versions=args.tdi,
            output_path=_resolve_output_path(args.output if args.plot == "unequal-aet" else None, args.output_dir, "unequal_aet_noise.png"),
            show=show_figures,
        )


if __name__ == "__main__":
    main()
