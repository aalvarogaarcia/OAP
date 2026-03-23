from typing import Any, TypeAlias
import numpy as np
import numpy.typing as npt
from collections.abc import Mapping

# Arrays
NumericArray: TypeAlias = npt.NDArray[np.number[Any]]
IndexArray: TypeAlias = npt.NDArray[np.integer[Any]]

# Estructuras
TrianglesAdjList: TypeAlias = Mapping[int, Mapping[int, list[int]]]