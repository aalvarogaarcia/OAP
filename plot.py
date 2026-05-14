from models import OAPCompactModel
from utils.utils import compute_triangles, read_indexed_instance

points    = read_indexed_instance("instance/uniform-0000015-1.instance")
triangles = compute_triangles(points)

# Relajado — debe mostrar grupos coloreados por flujo
model = OAPCompactModel(points, triangles, name="test")
model.build(objective="External", maximize=False, subtour="SCF", sum_constrain=True)
model.solve(time_limit=60, relaxed=True, plot=True, verbose=True)

# Entero — comportamiento actual sin cambios
model2 = OAPCompactModel(points, triangles, name="test2")
model2.build(objective="External", maximize=False, subtour="SCF", sum_constrain=True)
model2.solve(time_limit=60, relaxed=False, plot=True, verbose=True)

print(model2)