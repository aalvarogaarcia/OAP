# **Optimal Area Polygonization (OAP) \- MILP Framework** {#optimal-area-polygonization-(oap)---milp-framework}

This repository contains an ongoing research implementation for solving the **Optimal Area Polygonization** problem using Mixed-Integer Linear Programming (MILP).

## **📑 Table of Contents**

1. [The Problem](https://www.google.com/search?q=%23-the-problem)  
2. [Methodology](https://www.google.com/search?q=%23-methodology)  
3. [Repository Structure](https://www.google.com/search?q=%23-repository-structure)  
4. [Project Status](https://www.google.com/search?q=%23-project-status)  
5. [How to Cite](https://www.google.com/search?q=%23-how-to-cite)  
6. [References](https://www.google.com/search?q=%23-references)

## **📌 The Problem**

Given a set of points $S$ in a 2D Euclidean plane, the objective is to find a **simple polygon** (a polygon that does not self-intersect) whose vertices are exactly the points in $S$, such that it:

* **Minimizes** the enclosed area (**MIN-OAP**).  
* **Maximizes** the enclosed area (**MAX-OAP**).

Both versions of the problem are **NP-hard** and require advanced combinatorial optimization techniques to guarantee optimality for moderate to large-scale point sets.

[Back to index ↑](#optimal-area-polygonization-\(oap\)---milp-framework)

## **🛠 Methodology**

The project tackles the problem using two mathematical architectures implemented with the **Gurobi** solver:

### **1\. Compact Models (`models/gurobi.py`)**

These utilize a monolithic formulation based on:

* **Flow Theory:** Ensures connectivity and subtour elimination (ATSP style).  
* **Triangulation:** The area is calculated by selecting triangles from a precomputed triangulation of the convex hull.  
* **Simplicity:** Logical constraints that prevent selected triangles from overlapping or edges from crossing.

### **2\. Benders Decomposition (`models/benders.py`)**

To improve scalability, the problem logic is separated:

* **Master Problem:** Operates on the edge selection space (graph-based).  
* **Subproblems (SP\_Y / SP\_YP):** Verify the feasibility of the triangulation and the simplicity of the polygon.  
* **Benders Cuts:** Implementation of *Lazy Constraints* that utilize **Farkas Rays** to identify and prohibit edge configurations that do not allow for a valid polygon.

[Back to index ↑](#optimal-area-polygonization-\(oap\)---milp-framework)

## **📂 Repository Structure**

The project is organized modularly to separate optimization logic from analysis utilities:

```
OAP/
├── main.py              # Main entry point for batch executions
├── models/              # Optimization logic
│   ├── __init__.py      # Module initializer
│   ├── gurobi.py        # Compact models and triangulation logic
│   └── benders.py       # Decomposition framework and callbacks
├── utils/               # Support tools
│   ├── __init__.py
│   ├── utils.py         # Geometric functions and data handling
│   ├── model_stats.py   # Extraction of Gurobi metrics (Gap, Nodes, etc.)
│   └── analyze_benders.py # Post-mortem analysis of cuts and Farkas rays
├── instance/            # Sample .instance files
├── outputs/             # Results (Excel, LaTeX, Logs)
├── CITATION.cff         # Citation metadata
└── README.md            # Main documentation
```

[Back to index ↑](#optimal-area-polygonization-\(oap\)---milp-framework)

## **🚧 Project Status**

This repository is a **Work in Progress (WIP)**.

* **Implemented:** Functional compact models and base Benders structure.  
* **In Development:** Refinement of cut depth (Deepest Cuts) and subproblem optimization.  
* **Future:** Integration of a visual interface (likely Streamlit) to monitor polygon generation in real-time.

**Notice:** As the project is in a research phase, the Command Line Interface (CLI) and detailed user manuals will be published once a stable version is reached.

[Back to index ↑](#optimal-area-polygonization-\(oap\)---milp-framework)

## **🎓 How to Cite**

If you use this framework in your research, please cite it using the metadata provided in the `CITATION.cff` file. You can also use the "Cite this repository" button in the GitHub sidebar.

```
García, Á. (2024). Optimal Area Polygonization (OAP) - MILP Framework (Version 0.1.0-alpha). 
Available at: [https://github.com/aalvarogaarcia/OAP](https://github.com/aalvarogaarcia/OAP)
```

[Back to index ↑](#optimal-area-polygonization-\(oap\)---milp-framework)

## **📚 References**

This work is based on the following research:

* **Hernández-Pérez, H., Riera-Ledesma, J., Rodríguez-Martín, I., & Salazar-González, J. J.** *"Optimal area polygonisation problems: Mixed integer linear programming models"*.  
* **Hosseini, M., & Turner, J.** *"Deepest Cuts for Benders Decomposition"*.

[Back to index ↑](#optimal-area-polygonization-\(oap\)---milp-framework)