from __future__ import annotations

import glob
import os
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import OAPCompactModel
from utils.utils import compute_triangles, read_indexed_instance

DIR = "instance/little-instances"
FILES = sorted(glob.glob(os.path.join(DIR, "*.instance")))
TIME_LIMIT = 30


def as_key(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.10f}"
    return str(v)


def solve_with_mode(filepath: str, family: str, mode: int) -> tuple[Any, int]:
    name = os.path.basename(filepath).replace(".instance", "")
    points = read_indexed_instance(filepath)
    triangles = compute_triangles(points)
    model = OAPCompactModel(points, triangles, name=f"{name}_{family}_m{mode}")
    model.build(
        objective="Fekete",
        maximize=True,
        subtour="SCF",
        sum_constrain=True,
        strengthen=False,
    )
    if family == "bip":
        model._add_bipartition_constraints(mode=mode)
    else:
        model._add_triangle_crossing_constraints(mode=mode)
    model.solve(time_limit=TIME_LIMIT, verbose=False)
    _lp, _gap, ip, _t, _n = model.get_model_stats()
    return ip, int(model.model.Status)


def main() -> None:
    mismatches: list[str] = []
    errors: list[str] = []
    checks = 0

    for filepath in FILES:
        fname = os.path.basename(filepath)
        for family in ("bip", "tri"):
            ips: dict[int, Any] = {}
            status: dict[int, int] = {}
            for mode in (0, 1, 2):
                try:
                    ip, st = solve_with_mode(filepath, family, mode)
                    ips[mode] = ip
                    status[mode] = st
                    checks += 1
                except Exception as exc:  # pragma: no cover
                    errors.append(f"{fname} | {family} | mode={mode} | ERROR: {exc}")

            if len(ips) == 3:
                normalized = {m: as_key(v) for m, v in ips.items()}
                if len(set(normalized.values())) > 1:
                    mismatches.append(f"{fname} | {family} | IPs={ips} | Status={status}")

    print(f"FILES={len(FILES)} CHECKS={checks}")
    if errors:
        print(f"ERRORS={len(errors)}")
        for line in errors[:20]:
            print(line)
    if mismatches:
        print(f"MISMATCHES={len(mismatches)}")
        for line in mismatches[:100]:
            print(line)
    else:
        print("MISMATCHES=0")


if __name__ == "__main__":
    main()
