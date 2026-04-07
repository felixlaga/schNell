import argparse
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib-codex"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate LISA angular response-function plots for a range "
            "of frequencies."
        )
    )
    parser.add_argument(
        "--u-values",
        type=str,
        default=None,
        help=(
            "Comma-separated list of dimensionless frequencies f/F_STAR. "
            "If omitted, a log-spaced sweep is used."
        ),
    )
    parser.add_argument(
        "--u-min",
        type=float,
        default=0.03,
        help="Minimum f/F_STAR for the default log-spaced sweep.",
    )
    parser.add_argument(
        "--u-max",
        type=float,
        default=2.0,
        help="Maximum f/F_STAR for the default log-spaced sweep.",
    )
    parser.add_argument(
        "--num-frequencies",
        type=int,
        default=7,
        help="Number of frequencies in the default log-spaced sweep.",
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
        help="Response polarization to visualize.",
    )
    parser.add_argument(
        "--panel-scale",
        choices=("individual", "shared"),
        default="individual",
        help="Use individual or shared color scaling across panels.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=7,
        help="Number of contour levels per panel.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for PNG outputs and a manifest text file. "
            "Defaults to plots/lisa_response_angular_sweep."
        ),
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help=(
            "Optional multipage PDF path. Defaults to "
            "plots/lisa_response_angular_sweep.pdf."
        ),
    )
    return parser.parse_args()


def parse_u_values(args):
    if args.u_values:
        values = [float(part.strip()) for part in args.u_values.split(",") if part.strip()]
        if not values:
            raise ValueError("--u-values was provided but no values were parsed.")
        return np.asarray(values, dtype=float)
    return np.geomspace(args.u_min, args.u_max, args.num_frequencies)


def format_u_tag(u):
    text = f"{u:.5g}"
    return text.replace("-", "m").replace(".", "p")


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    felix_dir = repo_root / "felix"
    if str(felix_dir) not in sys.path:
        sys.path.insert(0, str(felix_dir))

    import notebook_code as nb

    u_values = parse_u_values(args)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(__file__).resolve().with_name("lisa_response_angular_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = args.pdf
    if pdf_path is None:
        pdf_path = Path(__file__).resolve().with_name("lisa_response_angular_sweep.pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_lines = [
        "# LISA angular response sweep",
        "# columns: u=f/F_STAR, frequency_Hz, png_path",
    ]

    with PdfPages(pdf_path) as pdf:
        for u in u_values:
            tag = format_u_tag(u)
            png_path = output_dir / f"lisa_response_u_{tag}.png"
            title = (
                "Angular dependence of LISA response-function integrands "
                f"for {args.polarization}-handed polarization at "
                rf"$f/f_{{\star}}={u:.3g}$"
            )

            fig, _, sky_maps = nb.plot_lisa_response_pair_angular_dependence(
                u=float(u),
                n_theta=args.n_theta,
                n_phi=args.n_phi,
                component=args.component,
                polarization=args.polarization,
                panel_scale=args.panel_scale,
                levels=args.levels,
                suptitle=title,
                savepath=png_path,
                show=False,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            manifest_lines.append(
                f"{u:.12g} {sky_maps['frequency']:.12g} {png_path}"
            )
            print(png_path)

    manifest_path = output_dir / "manifest.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n")
    print(pdf_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
