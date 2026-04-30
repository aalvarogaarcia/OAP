# models/OAPBendersModel.py
import logging
from typing import Literal

import gurobipy as gp
import numpy as np
from numpy.typing import NDArray

from models.mixin.benders_analysis_mixin import BendersAnalysisMixin
from models.mixin.benders_cgsp_mixin import BendersCGSPMixin
from models.mixin.benders_farkas_mixin import BendersFarkasMixin
from models.mixin.benders_master_mixin import BendersMasterMixin
from models.mixin.benders_optimize_mixin import BendersOptimizeMixin
from models.mixin.benders_pi_mixin import BendersPiMixin

# --- Importamos la Base y todos nuestros Mixins ---
from models.OAPBaseModel import OAPBaseModel

logger = logging.getLogger(__name__)


# ¡La herencia múltiple en todo su esplendor!
class OAPBendersModel(
    BendersMasterMixin,
    BendersCGSPMixin,
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

        # CGSP (deepest cuts) configuration — defaults to off for backward compat
        self.use_deepest_cuts: bool = False
        self.cut_weights_y: dict | None = None
        self.cut_weights_yp: dict | None = None

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
        sum_constrain: bool = True,
        crosses_constrain: bool = False,
        strengthen: bool = False,
        plot_strengthen: bool = False,
        use_deepest_cuts: bool = False,
        cut_weights_y: dict | None = None,
        cut_weights_yp: dict | None = None,
        semiplane: Literal[0, 1] = 0,
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

        Notes
        -----
        Default value 0 is binding: every existing benchmark, every cached
        experiment artefact, and every callback in BendersOptimizeMixin must
        behave bit-identically when this kwarg is omitted (NFR-5 / NFR-6).
        """
        logger.info(f"=== Construyendo OAPBendersModel ({benders_method.upper()}) ===")

        # Guardamos el método elegido para que el callback sepa qué cortes generar
        self.benders_method = benders_method

        # CGSP / deepest-cuts configuration
        self.use_deepest_cuts = use_deepest_cuts
        self.cut_weights_y = cut_weights_y
        self.cut_weights_yp = cut_weights_yp

        # 1. Construir el Maestro (Viene de BendersMasterMixin)
        self.build_master(
            objective=objective,
            mode=mode,
            maximize=maximize,
            crosses_constrain=crosses_constrain,
            semiplane=semiplane,
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

        logger.info("Construcción completada con éxito.")
