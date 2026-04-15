import argparse
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "schnell_matplotlib"),
)

import matplotlib
import numpy as np

matplotlib.use("Agg")


def _parse_csv_ints(raw_value):
    return tuple(int(part.strip()) for part in raw_value.split(",") if part.strip())


def _parse_csv_floats(raw_value):
    return tuple(float(part.strip()) for part in raw_value.split(",") if part.strip())


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate LISA transfer-function and channel diagnostics plots "
            "from the existing notebook response/noise machinery."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the figures should be written.",
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=3.0e-5,
        help="Minimum frequency in Hz.",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.5,
        help="Maximum frequency in Hz.",
    )
    parser.add_argument(
        "--nfreq",
        type=int,
        default=120,
        help="Number of frequency samples.",
    )
    parser.add_argument(
        "--ells",
        default="1,2,3,4",
        help="Comma-separated representative multipoles for the overview panels.",
    )
    parser.add_argument(
        "--auto-cross-ells",
        default="1,2",
        help="Comma-separated multipoles for the auto-vs-cross contribution plot.",
    )
    parser.add_argument(
        "--response-pair-ells",
        default="0,2,3,4",
        help="Comma-separated multipoles for the AA/EE/AE response plot.",
    )
    parser.add_argument(
        "--snr-ells",
        default="0,1,2,3,4",
        help="Comma-separated multipoles for the SNR-versus-duration plot.",
    )
    parser.add_argument(
        "--mission-durations",
        default="0.5,1,2,4",
        help="Comma-separated observing times in years for the SNR plot.",
    )
    parser.add_argument(
        "--breakdown-ell",
        type=int,
        default=2,
        help="Multipole to use in the detector/signal/noise breakdown plot.",
    )
    parser.add_argument(
        "--lmax",
        type=int,
        default=None,
        help="Maximum multipole to compute. Defaults to the maximum requested ell.",
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=32,
        help="HEALPix nside used for the response integrals.",
    )
    parser.add_argument(
        "--iter-sht",
        type=int,
        default=1,
        help="Number of healpy SHT improvement iterations.",
    )
    parser.add_argument(
        "--transfer-mus",
        default="-0.75,-0.25,0.25,0.75",
        help="Comma-separated representative mu values for the arm transfer plot.",
    )
    parser.add_argument(
        "--omega-amplitude",
        type=float,
        default=1.0e-12,
        help="Signal amplitude used in the decomposition plot.",
    )
    parser.add_argument(
        "--omega-alpha",
        type=float,
        default=0.0,
        help="Spectral slope used in the decomposition plot.",
    )
    parser.add_argument(
        "--omega-fref",
        type=float,
        default=1.0e-3,
        help="Reference frequency in Hz used in the decomposition plot.",
    )
    parser.add_argument(
        "--heatmap-quantity",
        choices=("omega", "snr_density", "both"),
        default="both",
        help="Which ell-frequency heatmap(s) to produce.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    felix_dir = repo_root / "felix"
    if str(felix_dir) not in sys.path:
        sys.path.insert(0, str(felix_dir))

    import notebook_code as nb
    import lisa_channel_diagnostics as lcd

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(__file__).resolve().with_name("lisa_channel_diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    representative_ells = _parse_csv_ints(args.ells)
    if not representative_ells:
        raise ValueError("--ells must contain at least one multipole.")
    auto_cross_ells = _parse_csv_ints(args.auto_cross_ells)
    if not auto_cross_ells:
        raise ValueError("--auto-cross-ells must contain at least one multipole.")
    response_pair_ells = _parse_csv_ints(args.response_pair_ells)
    if not response_pair_ells:
        raise ValueError("--response-pair-ells must contain at least one multipole.")
    snr_ells = _parse_csv_ints(args.snr_ells)
    if not snr_ells:
        raise ValueError("--snr-ells must contain at least one multipole.")
    mission_durations = _parse_csv_floats(args.mission_durations)
    if not mission_durations:
        raise ValueError("--mission-durations must contain at least one duration.")
    transfer_mus = _parse_csv_floats(args.transfer_mus)
    if not transfer_mus:
        raise ValueError("--transfer-mus must contain at least one value.")

    lmax = (
        max(
            max(representative_ells),
            max(auto_cross_ells),
            max(response_pair_ells),
            max(snr_ells),
            int(args.breakdown_ell),
        )
        if args.lmax is None
        else int(args.lmax)
    )

    frequency_hz = np.geomspace(args.fmin, args.fmax, args.nfreq)
    detailed = lcd.compute_lisa_multipole_sensitivity_detailed(
        frequency_hz,
        lmax=lmax,
        nside=args.nside,
        iter_sht=args.iter_sht,
    )

    omega_gw_h2 = lambda f: nb.power_law_omega_gw_h2(  # noqa: E731
        f,
        amplitude_h2=args.omega_amplitude,
        alpha=args.omega_alpha,
        f_ref=args.omega_fref,
    )
    c_ell_gw = np.ones(lmax + 1)

    output_paths = []

    transfer_path = output_dir / "lisa_transfer_vs_frequency.png"
    lcd.plot_transfer_functions_vs_frequency(
        frequency_hz,
        mus=transfer_mus,
        output_path=transfer_path,
        show=False,
    )
    output_paths.append(transfer_path)

    sensitivity_path = output_dir / "lisa_channel_sensitivity_overview.png"
    lcd.plot_channel_pair_sensitivity_overview(
        frequency_hz,
        detailed=detailed,
        ells=representative_ells,
        output_path=sensitivity_path,
        show=False,
    )
    output_paths.append(sensitivity_path)

    weights_path = output_dir / "lisa_channel_weight_overview.png"
    lcd.plot_inverse_variance_channel_weights(
        frequency_hz,
        detailed=detailed,
        ells=representative_ells,
        output_path=weights_path,
        show=False,
    )
    output_paths.append(weights_path)

    auto_cross_path = output_dir / "lisa_auto_vs_cross_absolute_contribution.png"
    lcd.plot_auto_vs_cross_channel_contribution(
        frequency_hz,
        detailed=detailed,
        ells=auto_cross_ells,
        output_path=auto_cross_path,
        show=False,
    )
    output_paths.append(auto_cross_path)

    response_pairs_path = output_dir / "lisa_representative_pair_responses.png"
    lcd.plot_representative_pair_responses(
        frequency_hz,
        detailed=detailed,
        ells=response_pair_ells,
        output_path=response_pairs_path,
        show=False,
    )
    output_paths.append(response_pairs_path)

    breakdown_path = (
        output_dir / f"lisa_signal_detector_noise_breakdown_ell{args.breakdown_ell}.png"
    )
    lcd.plot_signal_detector_noise_breakdown(
        frequency_hz,
        detailed=detailed,
        ell=args.breakdown_ell,
        omega_gw_h2=omega_gw_h2,
        c_ell_gw=c_ell_gw,
        output_path=breakdown_path,
        show=False,
    )
    output_paths.append(breakdown_path)

    snr_duration_path = output_dir / "lisa_snr_vs_mission_duration.png"
    lcd.plot_snr_vs_mission_duration(
        frequency_hz,
        detailed=detailed,
        mission_durations_yr=mission_durations,
        ells=snr_ells,
        omega_gw_h2=omega_gw_h2,
        c_ell_gw=c_ell_gw,
        output_path=snr_duration_path,
        show=False,
    )
    output_paths.append(snr_duration_path)

    heatmap_ells = tuple(range(lmax + 1))
    if args.heatmap_quantity in ("omega", "both"):
        omega_heatmap_path = output_dir / "lisa_multipole_omega_heatmap.png"
        lcd.plot_multipole_frequency_heatmap(
            frequency_hz,
            detailed=detailed,
            quantity="omega",
            ells=heatmap_ells,
            output_path=omega_heatmap_path,
            show=False,
        )
        output_paths.append(omega_heatmap_path)

    if args.heatmap_quantity in ("snr_density", "both"):
        snr_density_heatmap_path = output_dir / "lisa_multipole_snr_density_heatmap.png"
        lcd.plot_multipole_frequency_heatmap(
            frequency_hz,
            detailed=detailed,
            quantity="snr_density",
            omega_gw_h2=omega_gw_h2,
            c_ell_gw=c_ell_gw,
            ells=heatmap_ells,
            output_path=snr_density_heatmap_path,
            show=False,
        )
        output_paths.append(snr_density_heatmap_path)

    print("# LISA channel diagnostics")
    print(f"output_dir: {output_dir}")
    print(f"frequencies: {frequency_hz[0]:.3e} .. {frequency_hz[-1]:.3e} Hz")
    print(f"lmax: {lmax}")
    print(f"nside: {args.nside}")
    print(f"mission_durations_yr: {mission_durations}")
    print(f"heatmap_quantity: {args.heatmap_quantity}")
    print("files:")
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
