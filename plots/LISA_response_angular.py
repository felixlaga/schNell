import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the angular dependence of the LISA A/E/T response-function "
            "integrands on a regular (phi, theta) grid."
        )
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=None,
        help="Frequency in Hz. If omitted, use --u times F_STAR.",
    )
    parser.add_argument(
        "--u",
        type=float,
        default=0.1,
        help="Dimensionless frequency f / F_STAR used when --frequency is omitted.",
    )
    parser.add_argument(
        "--n-theta",
        type=int,
        default=181,
        help="Number of polar-angle samples in [0, pi].",
    )
    parser.add_argument(
        "--n-phi",
        type=int,
        default=361,
        help="Number of azimuthal samples in [0, 2*pi].",
    )
    parser.add_argument(
        "--component",
        choices=("abs", "real", "imag"),
        default="abs",
        help="Map component to visualize.",
    )
    parser.add_argument(
        "--polarization",
        choices=("R", "L", "I"),
        default="R",
        help="Response polarization to visualize: right-handed, left-handed, or intensity.",
    )
    parser.add_argument(
        "--panel-scale",
        choices=("individual", "shared"),
        default="individual",
        help="Use individual or shared color scaling across panels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to plots/lisa_response_angular_dependence.png.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=7,
        help="Number of contour levels per panel.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    felix_dir = repo_root / "felix"
    if str(felix_dir) not in sys.path:
        sys.path.insert(0, str(felix_dir))

    import notebook_code as nb

    output = args.output
    if output is None:
        output = Path(__file__).resolve().with_name(
            "lisa_response_angular_dependence.png"
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    nb.plot_lisa_response_pair_angular_dependence(
        f=args.frequency,
        u=args.u,
        n_theta=args.n_theta,
        n_phi=args.n_phi,
        component=args.component,
        polarization=args.polarization,
        panel_scale=args.panel_scale,
        levels=args.levels,
        savepath=output,
        show=False,
    )

    print(output)


if __name__ == "__main__":
    main()
