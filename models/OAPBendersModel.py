# models/OAPBendersModel.py
import logging
from typing import Literal

import gurobipy as gp
import numpy as np
from numpy.typing import NDArray

# --- Importamos la Base y todos nuestros Mixins ---
from models.OAPBaseModel import OAPBaseModel
from models.mixin.benders_master_mixin import BendersMasterMixin
from models.mixin.benders_farkas_mixin import BendersFarkasMixin
from models.mixin.benders_pi_mixin import BendersPiMixin
from models.mixin.benders_optimize_mixin import BendersOptimizeMixin
from models.mixin.benders_analysis_mixin import BendersAnalysisMixin

logger = logging.getLogger(__name__)

# ¡La herencia múltiple en todo su esplendor!
class OAPBendersModel(
    BendersMasterMixin, 
    BendersFarkasMixin, 
    BendersPiMixin, 
    BendersOptimizeMixin,
    BendersAnalysisMixin,
    OAPBaseModel
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
        self.sub_y.setParam('OutputFlag', 0)
        self.sub_yp.setParam('OutputFlag', 0)
        
        # 3. Estructuras de datos para la orquestación
        self.constrs_y = {}
        self.constrs_yp = {}
        self.iteration = 0
        self.cortes_añadidos = 0
        self.benders_method = "farkas" # Valor por defecto
        
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
    ) -> None:
        """
        Orquesta la construcción del Problema Maestro y de los Subproblemas.
        """
        logger.info(f"=== Construyendo OAPBendersModel ({benders_method.upper()}) ===")
        
        # Guardamos el método elegido para que el callback sepa qué cortes generar
        self.benders_method = benders_method
        
        # 1. Construir el Maestro (Viene de BendersMasterMixin)
        self.build_master(
            objective=objective, 
            mode=mode, 
            maximize=maximize, 
            crosses_constrain=crosses_constrain
        )
        
        # 2. Construir los Subproblemas (Viene de Farkas o Pi Mixin)
        if benders_method == "farkas":
            self.build_farkas_subproblems(sum_constrain=sum_constrain)
        elif benders_method == "pi":
            self.build_pi_subproblems(sum_constrain=sum_constrain)
        else:
            raise ValueError(f"Método de Benders desconocido: {benders_method}")

        logger.info("Construcción completada con éxito.")