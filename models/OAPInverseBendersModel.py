# models/OAPInverseBendersModel.py
"""Inverted Benders decomposition for the OAP.

In this decomposition the roles of the two sets of variables are swapped with
respect to :class:`~models.OAPBendersModel`:

* **Master**     — binary triangle-assignment variables ``y`` (internal) and
  ``yp`` (external).  Objective: maximise internal area.
* **Subproblem** — continuous tour-arc variables ``x`` (existence and
  feasibility of a consistent Hamiltonian tour given a fixed assignment).

When the subproblem LP is infeasible for a candidate ``(y*, yp*)``, a Farkas
cut is extracted and injected into the master as a lazy constraint.

Standard workflow::

    from models import OAPInverseBendersModel
    from utils.utils import read_indexed_instance, compute_triangles

    points    = read_indexed_instance("path/to/instance.instance")
    triangles = compute_triangles(points)
    model     = OAPInverseBendersModel(points, triangles, name="my_inst")
    model.build(objective="Internal", maximize=True)
    model.solve(time_limit=300, verbose=True)
"""
from __future__ import annotations

import logging
from typing import Literal

import gurobipy as gp
import numpy as np
from numpy.typing import NDArray

from models.OAPBaseModel import OAPBaseModel
from models.mixin.inv_benders_analysis_mixin import InvBendersAnalysisMixin
from models.mixin.inv_benders_master_mixin import InvBendersMasterMixin
from models.mixin.inv_benders_optimize_mixin import InvBendersOptimizeMixin
from models.mixin.inv_benders_sub_mixin import InvBendersSubMixin

logger = logging.getLogger(__name__)


class OAPInverseBendersModel(
    InvBendersMasterMixin,
    InvBendersSubMixin,
    InvBendersOptimizeMixin,
    InvBendersAnalysisMixin,
    OAPBaseModel,
):
    """Inverted Benders decomposition solver for the OAP.

    MRO (left-to-right):
    ``InvBendersMasterMixin → InvBendersSubMixin →
    InvBendersOptimizeMixin → InvBendersAnalysisMixin → OAPBaseModel``
    """

    def __init__(
        self,
        points:    NDArray[np.int64],
        triangles: NDArray[np.int64],
        name:      str = "OAPInverseBendersModel",
    ) -> None:
        """Initialise shared data structures and the subproblem LP.

        Args:
            points:    ``(N, 2)`` integer array of point coordinates.
            triangles: ``(T, 3)`` integer array of triangle vertex indices.
            name:      Gurobi model name (also used in default log path).
        """
        # 1. Base: creates self.model (master), self.points, self.CH,
        #          self.N, self.N_list, self.V_list,
        #          self.triangles, self.triangles_adj_list
        super().__init__(points, triangles, name)

        # 2. Subproblem LP — silenced; InfUnbdInfo + no DualReductions for
        #    Farkas ray availability (further params set in build_inv_sub)
        self.sub_x = gp.Model(f"{name}_Sub_X")
        self.sub_x.Params.OutputFlag     = 0
        self.sub_x.Params.InfUnbdInfo    = 1
        self.sub_x.Params.DualReductions = 0

        # 3. Shared state initialised to safe defaults
        Arc = tuple[int, int]
        self.constrs_sub: dict[str, dict[Arc, gp.Constr]] = {}
        self.iteration   = 0
        self.save_cuts   = False
        self.verbose     = False

        # 4. Default log path (outputs/ is git-ignored; created on demand)
        self.log_path = f"outputs/Others/Benders/{name}_inv/log.jsonl"
        logger.info("Default cut log path: %s", self.log_path)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_log_path(self, path: str) -> None:
        """Override the default cut-log file path.

        Args:
            path: Absolute or relative path to the ``.jsonl`` file.
        """
        self.log_path = path

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        objective:     Literal["Internal"] = "Internal",
        maximize:      bool                = True,
        sum_constrain: bool                = True,
    ) -> None:
        """Build master (``y``/``yp``) and subproblem (``x`` LP).

        Args:
            objective:     Objective type.  Only ``"Internal"`` is currently
                           supported (maximise internal triangulation area).
            maximize:      If True, maximise the objective; else minimise.
            sum_constrain: If True, add the cardinality constraints
                           ``Σy = N−2`` and ``Σyp = N−|CH|``.

        Raises:
            ValueError: If *objective* is not ``"Internal"``.
        """
        if objective != "Internal":
            raise ValueError(
                f"OAPInverseBendersModel only supports objective='Internal', "
                f"got {objective!r}."
            )

        logger.info(f"=== Building OAPInverseBendersModel ({self.name}) ===")

        # Step 1 — master problem (InvBendersMasterMixin)
        self.build_inv_master(
            objective=objective,
            maximize=maximize,
            sum_constrain=sum_constrain,
        )

        # Step 2 — subproblem LP (InvBendersSubMixin)
        self.build_inv_sub()

        logger.info("Build complete.")
