# models/OAPBendersModel.py
import logging
from typing import Literal

import gurobipy as gp
import numpy as np
from numpy.typing import NDArray

from models.mixin.benders_analysis_mixin import BendersAnalysisMixin
from models.mixin.benders_cgsp_mixin import BendersCGSPMixin
from models.mixin.benders_ddma_mixin import BendersDDMAMixin
from models.mixin.benders_farkas_mixin import BendersFarkasMixin
from models.mixin.benders_master_mixin import BendersMasterMixin
from models.mixin.benders_mw_mixin import BendersMagnantiWongMixin
from models.mixin.benders_optimize_mixin import BendersOptimizeMixin
from models.mixin.benders_pi_mixin import BendersPiMixin

# --- Importamos la Base y todos nuestros Mixins ---
from models.OAPBaseModel import OAPBaseModel

logger = logging.getLogger(__name__)


# ¡La herencia múltiple en todo su esplendor!
class OAPBendersModel(
    BendersMasterMixin,
    BendersDDMAMixin,            # F3 — before CGSP; shares sub_y/sub_yp
    BendersCGSPMixin,
    BendersMagnantiWongMixin,    # after CGSP — shares dual variable structure
    BendersFarkasMixin,
    BendersPiMixin,
    BendersOptimizeMixin,
    BendersAnalysisMixin,
    OAPBaseModel,
):
    def __init__(self, points: NDArray[np.int64], triangles: NDArray[np.int64], name: str = "OAPBendersModel"):
        """
        Inicializa la estructura de datos compartida y prepara los subproblemas.
        """
        # 1. Inicializa la Base (Crea self.model, self.points, self.CH, etc.)
        super().__init__(points, triangles, name)

        # 2. Modelos adicionales exclusivos de Benders
        self.sub_y = gp.Model(f"{name}_Sub_Y")
        self.sub_yp = gp.Model(f"{name}_Sub_YP")

        # Apagar el output de los subproblemas para que no ensucien la consola
        self.sub_y.setParam("OutputFlag", 0)
        self.sub_yp.setParam("OutputFlag", 0)
        self.model.Params.MIPGapAbs = 1.99

        # 3. Estructuras de datos para la orquestación
        self.constrs_y = {}
        self.constrs_yp = {}
        self.iteration = 0
        self.cortes_añadidos = 0
        self.benders_method = "farkas"  # Valor por defecto
        self._subtour_method = "SCF"    # Valor por defecto; actualizado en build()

        # CGSP (deepest cuts) configuration — defaults to off for backward compat
        self.use_deepest_cuts: bool = False
        self.cut_weights_y: dict | None = None
        self.cut_weights_yp: dict | None = None

        # F2.4 — model caches for the CGSP separation LPs.  ``None`` means
        # "not yet built"; the first call to ``_build_cgsp_y(p)`` populates
        # these and every subsequent call only re-sets the objective.  See
        # BendersCGSPMixin._compute_cgsp_objective and
        # BendersCGSPMixin.invalidate_cgsp_cache.
        self._cgsp_yp_cache: tuple | None = None
        self._cgsp_y_cache: tuple | None = None

        # F3 — DDMA configuration (defaults off for backward compat)
        self.use_ddma: bool = False

        # Magnanti-Wong (pareto-optimal cuts) — defaults to off
        self.use_magnanti_wong: bool = False
        self._core_point: dict | None = None
        self._core_point_strategy: str = "lp_relaxation"

        # Para el logger de Farkas/Pi si deseas guardar archivos JSON
        self.log_path = f"outputs/Others/Benders/{name}/log.json"
        print(f"Ruta por defecto para log de cortes: {self.log_path}")

    def set_log_path(self, path: str) -> None:
        """
        Permite configurar la ruta del log de Farkas/Pi desde fuera de la clase.
        """
        self.log_path = path

    def build(
        self,
        objective: Literal["Fekete", "Internal", "External", "Diagonals"] = "Fekete",
        mode: int = 0,
        maximize: bool = True,
        benders_method: Literal["farkas", "pi"] = "farkas",
        subtour: Literal["SCF", "DFJ"] = "SCF",
        sum_constrain: bool = True,
        crosses_constrain: bool = False,
        strengthen: bool = False,
        plot_strengthen: bool = False,
        use_deepest_cuts: bool = False,
        cut_weights_y: dict | None = None,
        cut_weights_yp: dict | None = None,
        semiplane: Literal[0, 1] = 0,
        use_knapsack: bool = False,
        use_cliques: bool = False,
        use_magnanti_wong: bool = False,
        core_point_strategy: Literal["lp_relaxation", "uniform"] = "lp_relaxation",
        cgsp_norm: Literal["misd", "relaxed_l1"] = "relaxed_l1",
        use_ddma: bool = False,
    ) -> None:
        """
        Orquesta la construcción del Problema Maestro y de los Subproblemas.

        Parameters
        ----------
        use_deepest_cuts : bool
            When True, the callback dispatches to CGSP-based cut generation
            (deepest cuts) instead of Farkas/Pi.  Defaults to False for
            backward compatibility.
        cut_weights_y : dict | None
            Optional L₁ normalisation weights for Y-subproblem CGSP.
            Keys mirror the constraint groups ('alpha', 'beta', 'gamma', 'delta',
            'global', 'r1', 'r2', 'r3', 'pi0').  Defaults to all-ones.
        cut_weights_yp : dict | None
            Optional L₁ normalisation weights for Y'-subproblem CGSP.
            Same key structure as cut_weights_y (suffixed with '_p').
        semiplane : Literal[0, 1], default 0
            Master-side half-plane (semiplane) constraints (Hernandez-Perez §5.2).
            - 0: off (default; preserves backward compatibility with all existing
                 experiments and CLI scripts).
            - 1: V1 — arc-ordering inequality x[i,j] <= x[j, j_next] for every
                 interior->hull arc whose left half-plane is empty of other interior
                 points.  Pure x-space; does NOT affect the Y / Y' subproblems or
                 the Farkas / Pi / CGSP cut derivations (see design note
                 2026-04-30-benders-semiplane-master.md §"Decomposition invariance").
             V2 is reserved for future work.
        use_magnanti_wong : bool, default False
            When True, the callback uses Magnanti-Wong Pareto-optimal cuts.
            Mutually exclusive with use_deepest_cuts=True (raises ValueError).
        core_point_strategy : str, default 'lp_relaxation'
            Strategy to compute x^0 used in Magnanti-Wong:
            - 'lp_relaxation': solve master LP once at build time (accurate).
            - 'uniform': x^0 = 1/N for all arcs (zero cost, less accurate).
        cgsp_norm : Literal["misd", "relaxed_l1"], default "relaxed_l1"
            Normalisation scheme for CGSP deepest-cut weights (F2.5).
            - 'relaxed_l1': column sums of |B| (Hosseini & Turner 2025 §3.3.2).
              Applied automatically when use_deepest_cuts=True and no explicit
              cut_weights_y/yp are supplied.
            - 'misd': unit weights (original MISD behaviour, all weights = 1).

        Notes
        -----
        Default value 0 is binding: every existing benchmark, every cached
        experiment artefact, and every callback in BendersOptimizeMixin must
        behave bit-identically when this kwarg is omitted (NFR-5 / NFR-6).
        """
        logger.info(f"=== Construyendo OAPBendersModel ({benders_method.upper()}) ===")

        # Validate mutually exclusive options
        n_methods = sum([use_deepest_cuts, use_magnanti_wong, use_ddma])
        if n_methods > 1:
            raise ValueError(
                "use_deepest_cuts, use_magnanti_wong, and use_ddma are mutually "
                "exclusive — only one may be True at a time."
            )

        # Guardamos el método elegido para que el callback sepa qué cortes generar
        self.benders_method = benders_method

        # CGSP / deepest-cuts configuration
        self.use_deepest_cuts = use_deepest_cuts
        self.cut_weights_y = cut_weights_y
        self.cut_weights_yp = cut_weights_yp

        # F2.5 — when using deepest cuts, default to Relaxed-ℓ₁ weights unless
        # the caller explicitly supplied weights or chose MISD.
        if use_deepest_cuts and cgsp_norm == "relaxed_l1" and cut_weights_y is None and cut_weights_yp is None:
            # _compute_relaxed_l1_weights requires constrs_y/yp to exist, so it
            # must be called AFTER build_farkas/pi_subproblems below.  Store the
            # strategy string and apply it after step 2.
            self._pending_cgsp_norm = "relaxed_l1"
        else:
            self._pending_cgsp_norm = None

        # 1. Construir el Maestro (Viene de BendersMasterMixin)
        self.build_master(
            objective=objective,
            mode=mode,
            maximize=maximize,
            subtour=subtour,
            crosses_constrain=crosses_constrain,
            semiplane=semiplane,
            use_knapsack=use_knapsack,
            use_cliques=use_cliques,
        )

        # 2. Construir los Subproblemas (Viene de Farkas o Pi Mixin)
        if benders_method == "farkas":
            self.build_farkas_subproblems(
                sum_constrain=sum_constrain, strengthen=strengthen, plot_strengthen=plot_strengthen
            )
        elif benders_method == "pi":
            self.build_pi_subproblems(
                sum_constrain=sum_constrain, strengthen=strengthen, plot_strengthen=plot_strengthen
            )
        else:
            raise ValueError(f"Método de Benders desconocido: {benders_method}")

        # F2.5 — apply Relaxed-ℓ₁ weights now that constrs_y/yp are populated
        if getattr(self, "_pending_cgsp_norm", None) == "relaxed_l1":
            self.cut_weights_y = self._compute_relaxed_l1_weights("y")
            self.cut_weights_yp = self._compute_relaxed_l1_weights("yp")
            # Invalidate any cached CGSP model so it is rebuilt with new weights
            self.invalidate_cgsp_cache()
            self._pending_cgsp_norm = None

        # Magnanti-Wong core point (must be after subproblems + master are built)
        self.use_magnanti_wong = use_magnanti_wong
        if use_magnanti_wong:
            self._core_point_strategy = core_point_strategy
            self._core_point = self._compute_core_point(core_point_strategy)
        else:
            self._core_point = None

        # F3 — DDMA configuration (after subproblems are built)
        self.use_ddma = use_ddma

        logger.info("Construcción completada con éxito.")
