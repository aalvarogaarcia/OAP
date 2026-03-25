import gurobipy as gp
import numpy as np
from numpy.typing import NDArray
from gurobipy import GRB
from typing import Literal
import matplotlib.pyplot as plt

import networkx as nx
from models.mixin.oap_builder_mixin import OAPBuilderMixin
from models.OAPBaseModel import OAPBaseModel

from utils.utils import (
    compute_convex_hull,
    compute_convex_hull_area,
    segments_intersect,
    point_in_triangle,
    cost_function_area
)

Arc = tuple[int, int]
ArcVarMap = dict[Arc, gp.Var]
McfArc = tuple[int, int, int]
McfVarMap = dict[McfArc, gp.Var]

class OAPCompactModel(OAPBaseModel, OAPBuilderMixin):
    def __init__(self, points: NDArray[np.int64], triangles: NDArray[np.int64], name: str = "OAPCompactModel"):
        super().__init__(points, triangles, name)
        self.N_list = range(len(points))
        self.N = len(points)
        self.CH = compute_convex_hull(points)
        self.V_list = range(len(triangles))
        self.convex_hull_area = compute_convex_hull_area(points)

        self.model = gp.Model(name)

        # Diccionarios de variables
        self.x: ArcVarMap = {}
        self.y: dict[int, gp.Var] = {}
        self.yp: dict[int, gp.Var] = {}
        
        self.c = cost_function_area(points, self.x, 0)
        
        # Variables opcionales (dependientes del modo/subtour)
        self.f: ArcVarMap = {}
        self.u: dict[int, gp.Var] = {}
        self.f_mcf: McfVarMap = {}
        self.z: ArcVarMap = {}
        self.zp: ArcVarMap = {}

    def build(self, 
              objective: Literal["Fekete", "Internal", "External", "Diagonals"] = "Fekete",
              mode: int = 0,
              maximize: bool = True, 
              subtour: Literal["SCF", "MTZ", "MCF"] = "SCF",
              sum_constrain: bool = True,
              semiplane: Literal[0,1,2] = 0,
              use_knapsack: bool = False,
              use_cliques: bool = False
              ) -> None:
        """
        Orquestador principal que construye el modelo paso a paso.
        """
        self._create_variables(subtour, objective, mode)
        self._set_objective(objective, mode, maximize)
        self._add_degree_constraints()
        self._add_subtour_constraints(subtour)
        if sum_constrain:
            self._add_sum_constraints()
        self._add_triangle_ch_constraints()
        self._add_variable_relation_constraints(objective, mode)

        if semiplane > 0:
            self.add_restricciones_semiplano(version=semiplane)
            
        if use_knapsack:
            self.inyectar_cortes_knapsack_locales()
            
        if use_cliques:
            self.inyectar_cliques_de_cruce()

        self.model.update()

        self.model.update()  # Asegura que todas las variables y restricciones estén registradas en el modelo antes de optimizar
        

    def solve(self, 
              time_limit: int = 7200, 
              verbose: bool = False, 
              relaxed: bool = False, 
              plot: bool = False) -> None:
       """
       Ejecuta la optimización del modelo, aplica relajación si es necesario 
       y procesa los resultados.
       """
       if verbose:
           print("Constraints added. \nOptimizing model...")
       
       # --- Configuración de Parámetros ---
       self.model.setParam('OutputFlag', 1 if verbose else 0)
       self.model.setParam('TimeLimit', time_limit)
       self.model.Params.MIPGap = 0.00001
       self.model.Params.NodeLimit = GRB.INFINITY
       self.model.Params.SolutionLimit = GRB.MAXINT
       
       self.model.update()

       # --- Relajación Lineal (LP Relaxation) ---
       if relaxed:
           for v in self.model.getVars():
               if v.VType != GRB.CONTINUOUS:
                   v.VType = GRB.CONTINUOUS
           self.model.update()

       # --- Optimización ---
       self.model.optimize()

       # --- Extracción de Resultados ---
       self.x_results = []

       if self.model.SolCount > 0:
           # Iteramos directamente sobre nuestro diccionario de variables self.x
           self.x_results = [arc for arc, var in self.x.items() if var.X > 0.5]

       # --- Visualización ---
       if plot:
           if self.model.SolCount > 0:
               title = "Optimal Tour" if self.model.Status == GRB.OPTIMAL else "Best Found Tour"
               self.plot(title=title)
           elif verbose:
               print("No feasible solution found to plot.")

    def plot(self, title: str = "Solution") -> None:
        """Dibuja la solución del modelo utilizando los resultados almacenados en la clase."""
        if not hasattr(self, 'x_results') or not self.x_results:
            print("No results to plot. Please solve the model first and ensure a solution was found.")
            return

        G = nx.DiGraph()
        G.add_edges_from(self.x_results)
        
        plt.figure(figsize=(8, 8))
        plt.title(title)
        plt.scatter(self.points[:, 0], self.points[:, 1], color='blue')
        
        # Dibuja las aristas del tour
        for edge in G.edges():
            pt1 = self.points[edge[0]]
            pt2 = self.points[edge[1]]
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r-', alpha=0.7)

        # Dibuja la envolvente convexa (ya la calculamos en el __init__)
        hull_set = set(self.CH)
        for i in range(len(self.CH)):
            pt1 = self.points[self.CH[i]]
            pt2 = self.points[self.CH[(i + 1) % len(self.CH)]]
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'g-.', alpha=0.5)
        
        # Etiqueta los puntos con sus índices
        for i, pt in enumerate(self.points):
            if i in hull_set:
                # Puntos de la envolvente convexa en rojo y negrita
                plt.annotate(str(i), (pt[0], pt[1]), 
                            textcoords="offset points", xytext=(5, 5),
                            fontsize=10, color='red', fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            else:
                # Puntos regulares en negro
                plt.annotate(str(i), (pt[0], pt[1]), 
                            textcoords="offset points", xytext=(5, 5),
                            fontsize=9, color='black',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))    
        
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    def __str__(self) -> str:
        """Define lo que se muestra al hacer print() de la instancia de la clase."""
        
        # 1. Si el modelo aún no se ha construido o resuelto, evitamos errores
        if self.model.NumVars == 0:
            return "OAPCompactModel (Estado: No construido)"
        if not hasattr(self, 'x_results') or not self.x_results:
            return "Modelo sin solución factible para mostrar."
        lp, gap, ip, elapsed_time, nodes = self.get_model_stats()
        tour = self.get_tour()

        # 3. Construimos el texto línea por línea
        lineas = [
            "-" * 30,
            "--- Valores del modelo matemático compacto ---",
            "-" * 30,
            f"Columnas modelo original: {self.model.NumVars}",
            f"Filas modelo original: {self.model.NumConstrs}",
            f"Area de la envolvente convexa: {self.convex_hull_area:.4f}",
            "",
            "-" * 30,
            "--- Valores del modelo IP y Relajado ---",
            "-" * 30,
            f"Instance: {self.model.ModelName}",
            f"IP Objective Value: {ip:.2f}",
            f"LP Objective Value: {lp:.2f}",
            f"Optimality Gap: {gap:.2f}%",
            f"Elapsed Time: {elapsed_time:.2f} seconds",
            f"Number of Nodes Explored: {nodes}",
            "",
            "-" * 30,
            "--- Tour obtenido ---",
            str(tour),
            "-" * 30,
            "",
            f"Resultados: LP={lp:.2f}, Gap={gap:.2f}%, IP={ip:.2f}, Time={elapsed_time:.2f}s, Nodes={nodes}"
        ]
        
        return "\n".join(lineas)

    def add_restricciones_semiplano(self, version: int = 1):
        """Agrega restricciones de semiplano (V1 o V2) al modelo."""
        # Filtrar índices que no están en la envolvente convexa
        A_pp = [i for i in range(self.N) if i not in self.CH]
        constrains = []

        if version == 1:
            for i in A_pp:
                for j in self.CH: 
                    if (i, j) not in self.x:
                        continue
                        
                    semiplano_izquierdo_limpio = True
                    for k in A_pp:
                        if k == i: 
                            continue
                        
                        x_i, y_i = self.points[i]
                        x_j, y_j = self.points[j]
                        x_k, y_k = self.points[k]
                        
                        D = (x_j - x_i) * (y_k - y_i) - (y_j - y_i) * (x_k - x_i)
                        
                        if D > 0:
                            semiplano_izquierdo_limpio = False
                            break
                            
                    if semiplano_izquierdo_limpio:
                        index = np.where(self.CH == j)[0][0]
                        j_siguiente = self.CH[(index + 1) % len(self.CH)]
                        
                        if (j, j_siguiente) in self.x:
                            constrains.append(
                                self.model.addConstr(
                                    self.x[i,j] <= self.x[j, j_siguiente],
                                    name=f"semiplano_{i}_{j}"
                                ))
            print(f"Añadidas {len(constrains)} restricciones de semiplano V1.")

        elif version == 2:
            for i in A_pp:
                for j in self.CH:    
                    if (i, j) not in self.x:
                        continue
                    
                    S_left = []
                    for k in A_pp:
                        if k == i: 
                            continue
                        x_i, y_i = self.points[i]
                        x_j, y_j = self.points[j]
                        x_k, y_k = self.points[k]
                        D_k = (x_j - x_i) * (y_k - y_i) - (y_j - y_i) * (x_k - x_i)

                        if D_k > 0:
                             S_left.append(k)

                    idx_j = np.where(self.CH == j)[0][0]
                    j_siguiente = self.CH[(idx_j + 1) % len(self.CH)]

                    if len(S_left) == 0:
                        nodo_actual_ch = j
                        for step in range(1, len(self.CH)):
                            idx_siguiente = (idx_j + step) % len(self.CH)
                            nodo_siguiente_ch = self.CH[idx_siguiente]
                            
                            x_sig, y_sig = self.points[nodo_siguiente_ch]
                            D_sig = (x_j - x_i) * (y_sig - y_i) - (y_j - y_i) * (x_sig - x_i)
                            
                            if D_sig > 0: 
                                if (nodo_actual_ch, nodo_siguiente_ch) in self.x:
                                    constrains.append(
                                        self.model.addConstr(
                                            self.x[i, j] <= self.x[nodo_actual_ch, nodo_siguiente_ch],
                                            name=f"semiplano_cadena_{i}_{j}_fuerza_{nodo_actual_ch}_{nodo_siguiente_ch}"
                                        ))
                                nodo_actual_ch = nodo_siguiente_ch
                            else:
                                break
                    else:
                        expr_escape = gp.LinExpr()
                        if (j, j_siguiente) in self.x:
                            expr_escape.addTerms(1.0, self.x[j, j_siguiente])

                        for k in S_left:
                            if (j, k) in self.x:
                                expr_escape.addTerms(1.0, self.x[j, k])

                        constrains.append(
                            self.model.addConstr(
                                self.x[i, j] <= expr_escape,
                                name=f"bolsillo_{i}_{j}_soporta_{len(S_left)}_puntos"
                            ))
            print(f"Añadidas {len(constrains)} restricciones de semiplano V2.")

        elif version == 3:
            for i in A_pp:
                for j in self.CH:    
                    if (i, j) not in self.x:
                        continue
                    
                    S_left = []
                    for k in A_pp:
                        if k == i: 
                            continue
                        x_i, y_i = self.points[i]
                        x_j, y_j = self.points[j]
                        x_k, y_k = self.points[k]
                        D_k = (x_j - x_i) * (y_k - y_i) - (y_j - y_i) * (x_k - x_i)

                        if D_k > 0:
                             S_left.append(k)

                    idx_j = np.where(self.CH == j)[0][0]
                    j_siguiente = self.CH[(idx_j + 1) % len(self.CH)]

                    if len(S_left) == 0:
                        nodo_actual_ch = j
                        for step in range(1, len(self.CH)):
                            idx_siguiente = (idx_j + step) % len(self.CH)
                            nodo_siguiente_ch = self.CH[idx_siguiente]
                            
                            x_sig, y_sig = self.points[nodo_siguiente_ch]
                            D_sig = (x_j - x_i) * (y_sig - y_i) - (y_j - y_i) * (x_sig - x_i)
                            
                            if D_sig > 0: 
                                if (nodo_actual_ch, nodo_siguiente_ch) in self.x:
                                    constrains.append(
                                        self.model.addConstr(
                                            self.x[i, j] <= self.x[nodo_actual_ch, nodo_siguiente_ch],
                                            name=f"semiplano_cadena_{i}_{j}_fuerza_{nodo_actual_ch}_{nodo_siguiente_ch}"
                                        ))
                                nodo_actual_ch = nodo_siguiente_ch
                            else:
                                break
                    else:
                        expr_escape = gp.LinExpr()
                        if (j, j_siguiente) in self.x:
                            expr_escape.addTerms(1.0, self.x[j, j_siguiente])

                        for k in S_left:
                            if (j, k) in self.x:
                                expr_escape.addTerms(1.0, self.x[j, k])

                        constrains.append(
                            self.model.addConstr(
                                self.x[i, j] <= expr_escape,
                                name=f"bolsillo_{i}_{j}_soporta_{len(S_left)}_puntos"
                            ))
            print(f"Añadidas {len(constrains)} restricciones de semiplano V3.")



    def inyectar_cortes_knapsack_locales(self):
        """Inyecta restricciones de mochila que limitan la contribución fraccionaria."""
        cortes_añadidos = 0
        for i in range(self.N):
            max_beneficio_real = 0.0
            
            for j1 in range(self.N):
                if j1 == i or (i, j1) not in self.x: 
                    continue
                    
                for j2 in range(j1 + 1, self.N):
                    if j2 == i or (i, j2) not in self.x: 
                        continue

                    es_pareja_legal = True
                    for k in range(self.N):
                        if k in [i, j1, j2]:
                             continue
                        if point_in_triangle(self.points[k], self.points[j1], self.points[i], self.points[j2]):
                            es_pareja_legal = False
                            break
                    
                    if es_pareja_legal:
                        beneficio_pareja = self.c.get((i, j1), 0) + self.c.get((i, j2), 0)
                        if beneficio_pareja > max_beneficio_real:
                            max_beneficio_real = beneficio_pareja
                            
            expr_knapsack = gp.LinExpr()
            for j in range(self.N):
                if j != i and (i, j) in self.x:
                    expr_knapsack.addTerms(self.c.get((i, j), 0), self.x[i, j])
                    
            if expr_knapsack.size() > 0:
                self.model.addConstr(expr_knapsack <= max_beneficio_real, name=f"knapsack_local_{i}")
                cortes_añadidos += 1
                
        print(f"Inyectados {cortes_añadidos} Cortes Knapsack Locales.")

    def inyectar_cliques_de_cruce(self):
        """Busca grupos de arcos que se cruzan TODOS entre sí (Cliques)."""
        print("Construyendo grafo de intersecciones para Cliques...")
        aristas = [(i, j) for (i, j) in self.x.keys() if i < j]

        G_cruces = nx.Graph()
        G_cruces.add_nodes_from(aristas)
        
        for idx, e1 in enumerate(aristas):
            for e2 in aristas[idx+1:]:
                if e1[0] in e2 or e1[1] in e2:
                    continue
                    
                p1, p2 = self.points[e1[0]], self.points[e1[1]]
                p3, p4 = self.points[e2[0]], self.points[e2[1]]
                
                if segments_intersect(p1, p2, p3, p4):
                    G_cruces.add_edge(e1, e2)
                    
        cliques = list(nx.find_cliques(G_cruces))
        cortes_añadidos = 0
        
        for clique in cliques:
            if len(clique) >= 3:
                expr_clique = gp.LinExpr()
                for e in clique:
                    if e in self.x:
                        expr_clique.addTerms(1.0, self.x[e[0], e[1]])
                    if (e[1], e[0]) in self.x:
                        expr_clique.addTerms(1.0, self.x[e[1], e[0]])
                
                self.model.addConstr(expr_clique <= 1, name=f"clique_cruce_{cortes_añadidos}")
                cortes_añadidos += 1
                
        print(f"Inyectados {cortes_añadidos} Cortes de Clique de Cruces.")