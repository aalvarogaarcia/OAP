# models/mixin/inv_benders_analysis_mixin.py
"""Post-mortem PDF report generator for the Inverted Benders model."""

from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from models.typing_oap import NumericArray
from utils.utils import format_cut_string, load_farkas_logs, plot_farkas_ray_network

logger = logging.getLogger(__name__)


class InvBendersAnalysisMixin:
    """Mixin that generates a PDF report from an inverted-Benders cut log.

    Mirrors :class:`~models.mixin.benders_analysis_mixin.BendersAnalysisMixin`.
    The only difference is that each log record contains ``active_y`` /
    ``active_yp`` fields (triangle-indexed) instead of ``active_x`` (arc-indexed).
    ``plot_farkas_ray_network`` operates on ``dual_components``, which is
    arc-keyed in both cases, so it is reused unchanged.
    """

    points: NumericArray
    N: int
    name: str
    log_path: str | None

    def generate_inv_benders_report(
        self,
        output_pdf_path: str | None = None,
    ) -> None:
        """Read the JSONL cut log and write a PDF with one page per cut.

        Each page shows the Farkas ray network (reusing
        :func:`~utils.visualization.plot_farkas_ray_network`) and the textual
        cut expression derived from the master variables ``y``/``yp``.

        Args:
            output_pdf_path: Output PDF path.  Defaults to
                ``outputs/Analysis/PostMortem_Inv_{name}.pdf``.
        """
        if not hasattr(self, "log_path") or not self.log_path:
            logger.warning(f"[{self.name}] Cannot generate report — run solve(save_cuts=True) first.")
            return

        if not os.path.exists(self.log_path):
            logger.warning(f"[{self.name}] Log file not found at {self.log_path}.")
            return

        logs = load_farkas_logs(self.log_path)
        if not logs:
            logger.warning("Log file is empty — no cuts to plot.")
            return

        logger.info(f"Loaded {len(logs)} cut records from {self.log_path}.")

        if output_pdf_path is None:
            output_pdf_path = f"outputs/Analysis/PostMortem_Inv_{self.name}.pdf"

        parent = os.path.dirname(output_pdf_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with PdfPages(output_pdf_path) as pdf:
            for i, log_entry in enumerate(logs):
                plt.figure(figsize=(10, 8))
                plot_farkas_ray_network(
                    log_entry,
                    points=self.points,
                    save_path=None,
                    show_plot=False,
                )

                cut_expr_data = log_entry.get("cut_expr")
                cut_sense = log_entry.get("sense")
                if cut_expr_data and cut_sense in ("<=", ">="):
                    text = format_cut_string(cut_expr_data, sense=cut_sense)
                    plt.text(
                        0.5,
                        -0.05,
                        text,
                        transform=plt.gca().transAxes,
                        fontsize=9,
                        ha="center",
                        va="top",
                        style="italic",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
                    )

                pdf.savefig(bbox_inches="tight")
                plt.close("all")

                if (i + 1) % 10 == 0 or (i + 1) == len(logs):
                    logger.info(f"Processed {i + 1}/{len(logs)} cuts...")

        logger.info(f"Report saved to: {output_pdf_path}")
