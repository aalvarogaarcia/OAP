# Importamos las clases desde sus respectivos archivos dentro del paquete
from .OAPCompactModel import OAPCompactModel

# Asumo que tu modelo Benders principal está en un archivo llamado OAPBendersModel.py 
# o dentro de una subcarpeta benders. Si está en la raíz de 'models':
from .OAPBendersModel import OAPBendersModel

from .OAPInverseBendersModel import OAPInverseBendersModel

# Opcional pero muy recomendado: Definir __all__
# Esto le dice a Python qué clases se exportan públicamente si alguien hace 'from models import *'
__all__ = [
    "OAPCompactModel",
    "OAPBendersModel",
    "OAPInverseBendersModel",
]