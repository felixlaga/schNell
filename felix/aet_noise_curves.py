import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


C_LIGHT = 299_792_458.0


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
        Matplotlib figure/axes and a dict with the four curve arrays.
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

    ax.set_xlim(1.0e-4, 1.0)
    ax.set_ylim(5.0e-24 if plot_sqrt else 1.0e-45, 1.0e-17 if plot_sqrt else 1.0e-33)
    ax.set_xlabel("frequency [Hz]", fontsize=16)
    ax.set_ylabel(
        r"$\sqrt{\tilde N}\,[\mathrm{Hz}^{-1/2}]$"
        if plot_sqrt
        else r"$\tilde N\,[\mathrm{Hz}^{-1}]$",
        fontsize=16,
    )
    ax.grid(True, which="both", color="0.75", alpha=0.7)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(loc="upper right", fontsize=14, frameon=True)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax, curves


def main():
    parser = argparse.ArgumentParser(
        description="Plot the LISA A/E and T tilde-noise curves."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path where the figure should be written.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Plot the raw tilde-noise PSDs instead of their square roots.",
    )
    args = parser.parse_args()

    plot_aet_noise_curves(
        output_path=args.output,
        show=args.output is None,
        plot_sqrt=not args.raw,
    )


if __name__ == "__main__":
    main()
