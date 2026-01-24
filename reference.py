# Instalamos las librerías que vamos a usar
import array
import random 
import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx

# !pip install ortools
from ortools.linear_solver import pywraplp
from scipy.spatial import ConvexHull, convex_hull_plot_2d



''' Leemos los datos de un fichero '''

EPS = 0.001 # epsilon

def lectura_fichero(nombre_fichero):
  lista=[]
  lista_limpia=[]
  f = open(nombre_fichero, 'r')
  lines = f.readlines() # Lee línea a línea
  for i in lines:
    lista.append(i)
  for k in range(2,len(lista)):
    limpiar=lista[k].strip('\n').split()
    lista_limpia.append(limpiar)
  
  # Imprimimos por pantalla nuestra lista con todos los datos del fichero
  #print('Lista sin espacios:')
  #print(lista_limpia)

  # Declaramos las variables y los vectores que vamos a usar en el resto del problema como globales
  global nP  # Número de puntos
  global nCH # Número de puntos en la envolvente convexa
  global nT  # Número de triángulos
  global nSC # Número de segmentos que se cruzan
  global nTI # Número de triángulos incompatibles
  global area_CH # Área de la envolvente convexa
  global N   # Conjunto de puntos
  global N0  # Conjunto de puntos sin el punto 0
  global V   # Conjunto de triángulos
  global coord
  global CH
  global triangulos
  global areas
  global SC
  global triangulos_incompatibles
  global ta_arc
  global estaEnCH

  # De la primera línea, recogemos los siguientes datos:
  nP = int(lista_limpia[0][0])
  nCH = int(lista_limpia[0][1])
  nT = int(lista_limpia[0][2])   
  nSC = int(lista_limpia[0][3])  
  nTI = int(lista_limpia[0][4]) 
  area_CH =  float(lista_limpia[0][5])
  # Imprimimos por pantalla para comprobar si los datos se han guardado correctamente
  # print('\n')
  # print('El número de puntos es', nP)
  # print('El número de puntos en la envolvente convexa es', nCH)
  # print('El número de triángulos es', nT)
  # print('El número de segmentos que se cruzan es', nSC)
  # print('El número de triángulos incompatibles es', nTI)
  # print('El área de la envolvente convexa es', area_CH)
  # print('\n')

  # Guardamos las coordenadas de los puntos en una matrix de tamaño (nP x 2)
  coord = np.zeros((nP,2))
  first_line = 2
  for i in range(nP) :
    coord[i][0] = float(lista_limpia[first_line + i][1])
    coord[i][1] = float(lista_limpia[first_line + i][2])
  # print('\n')
  # print('Las coordenadas de los', nP, 'puntos son \n', coord)

  # Guardamos los puntos que forman la envolvente convexa
  CH = np.zeros(nCH, dtype = int) # Puntos que forman la envolvente convexa
  first_line = 2+nP+1
  for i in range(nCH) :
    CH[i] = int(lista_limpia[first_line + i][1])
  # print('\n')
  # print('Los puntos que forman la envolvente convexa son \n', envolvente_convexa)

  # Guardamos las coordenadas de los triángulos en una matriz de tamaño (nT x 3) y las áreas de cada uno de esos triángulos en un vector
  triangulos = np.zeros((nT,3), dtype=int) 
  areas =  np.zeros(nT) 
  first_line = 2+nP+1+nCH+1 
  # También creamos una lista de listas para crear los triángulos adyacentes a dos puntos dados (orientados)
  ta_arc = []

  for i in range(nP):
    ta_arc.append([])
    for j in range(nP):
      ta_arc[i].append([])

  # print(ta_arc)

  for t in range(nT) :
    i = int(lista_limpia[first_line + t][1])
    j = int(lista_limpia[first_line + t][2])
    k = int(lista_limpia[first_line + t][3])
    triangulos[t][0] = i
    triangulos[t][1] = j
    triangulos[t][2] = k
    areas[t] = float(lista_limpia[first_line + t][4])
  # print(i , " ", j, " ", k)
    if areas[t] > 0:
      ta_arc[i][j].append(t)
      ta_arc[j][k].append(t)
      ta_arc[k][i].append(t)
    else:
      ta_arc[j][i].append(t)
      ta_arc[k][j].append(t)
      ta_arc[i][k].append(t)
  # print(ta_arc)

  # Guardamos los puntos de los segmentos que se cruzan (i,j) y (k,l)
  SC = np.zeros((nSC,4), dtype=int) 
  first_line = 2+nP+1+nCH+1+nT+1
  for i in range(nSC) :
    SC[i][0] = int(lista_limpia[first_line + i][2])
    SC[i][1] = int(lista_limpia[first_line + i][1])
    SC[i][2] = int(lista_limpia[first_line + i][4])
    SC[i][3] = int(lista_limpia[first_line + i][3])
  # print('\n')
  # print('Los segmentos que se cruzan son los siguientes \n', SC)

  # Guardamos un vector que nos dice si un punto está o no en la CH
  estaEnCH = np.zeros(nP, dtype=int)
  for i in range(nCH) :
    estaEnCH[CH[i]] = 1


  # Guardamos los triángulos que son incompatibles (triángulo_incompatible1,triángulo_incompatible2)
  triangulos_incompatibles = np.zeros((nTI,2), dtype=int) 
  first_line = 2+nP+1+nCH+1+nT+1+nSC+1
  for i in range(nTI) :
    triangulos_incompatibles[i][0] = int(lista_limpia[first_line + i][1])
    triangulos_incompatibles[i][1] = int(lista_limpia[first_line + i][2])
  # print('\n')
  # print('Los triángulos incompatibles son los siguientes \n', triangulos_incompatibles)

  # Definimos otras variables que nos van a hacer falta
  N = range(nP)
  N0 = range(1,nP)
  V = range(nT)


# Leemos el primer fichero de la lista para poder ejecutar el programa
# lectura_fichero(lista_ficheros[0]) 

''' Distribución de puntos en el plano '''

def dibuja(selected):           
  plt.figure(figsize=(8,8))
  plt.xlim(0,200)
  plt.ylim(0,200)
  plt.xlabel("Coordenada X", fontsize='16')
  plt.ylabel("Coordenada Y", fontsize='16')
  #plt.title('Distribución de los puntos en el plano \n', fontsize='18')

  for i in N : 
    plt.plot(coord[i][0], coord[i][1],'ro')
    plt.annotate('%i'%(i), (coord[i][0],coord[i][1]+2),fontsize='12',color='black')
  for (i,j) in selected:
    # Se muestran los arcos una vez se resuelva el problema
    plt.plot([coord[i][0],coord[j][0]], [coord[i][1],coord[j][1]], 'y-') 
 
  plt.show()

#dibuja({})  

def computa_matriz_distancias(coordenadas):
  n = len(coordenadas)
  distancias = np.zeros((n,n))
  for i in range(n):
    for j in range(n):
      distancias[i][j] = math.sqrt((coord[i][0]-coord[j][0])**2 + (coord[i][1]-coord[j][1])**2)
  return distancias

def computa_distancia_ruta(dist, selected):
  distancia = 0
  for (i,j) in selected:
    distancia += dist[i,j]
  return distancia

''' MODELO 
A esta función le pasamos los siguientes parámetros
objetive:  sentido de la función objetivo que puede ser Maximize o Minimize
lp_solver: solver que utiliza para resolver el modelo (GUROBI o CBC)
model:     Modelo que utiliza para resolver el problema
           '2' MT2 considera las áreas dentro del polígono y 
           '3' MT3 considera las áreas entre el polígono y la envolvente convexa
           '4' MT4 considera las áreas dentro y fuera de la envolvente convexa
seg_cruzan: 'True' si se fortalece la restricción con segmentos que se cruzan
directed_x: 'True' si se consideran las variables dirigidas 
y_binarias: 'True' si las variables y son binarias
num_triangles: 'True' si se considera una restricción con el número de triángulos
write_lp: 'True' si se quiere escribir el modelo en un fichero .lp
'''
def modelo(objetive, lp_solver, model, directed_x, seg_cruzan, y_binarias, num_triangles, write_lp = False):
  selected = {} 
  if lp_solver == 'GUROBI' :
    solver = pywraplp.Solver('Modelo', pywraplp.Solver.GUROBI_MIXED_INTEGER_PROGRAMMING)
    print("Solver selected:", lp_solver)
  else :
    solver = pywraplp.Solver('Modelo', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

  solver.SetTimeLimit(7200000)
  # solver.parameters.max_time_in_seconds = 10.0

  # Definimos las variables de decisión
  # 1 - Si el arco (i,j) está en el perímetro
  if directed_x == 0 :
    # Consideramos tambien las variables no dirigidas x[i,j] = xd[i,j] + xd[j,i] para nuestras cuentas
    # xd = { (i,j) : solver.NumVar(0, 1, 'xd[%i,%i]' % (i,j)) for i in N for j in N if j!=i }
    x = { (i,j) : solver.BoolVar('x[%i,%i]' % (i,j)) for i in N for j in N if i<j }
  else :
    xd = { (i,j) : solver.BoolVar('xd[%i,%i]' % (i,j)) for i in N for j in N if j!=i }


  #vamos a utilziar unas variables f para controlar los subciclos
  f = { (i,j) : solver.NumVar(0, nP-1, 'f[%i,%i]' % (i,j)) for i in N for j in N if j!=i }
  # Posición-1 del punto i en la ruta
  #u = { i : solver.NumVar(0.0, nP-2, 'u[%i]' % i) for i in N0 }

  # Definimos las variables "y" e "yp" para los triángulos
  # Por comodidad vamos a definir variables "w" e "wp" que son la suma de todos los triángulos orientados que comparten un arco 
  # 1 - Si el triángulo t está en la triangulación solución
  if y_binarias == 1 :
    if model == 2 or model == 4 :
      y = { (i) : solver.BoolVar('y[%i]' % (i)) for i in V }
      w = { (i,j) : solver.BoolVar('w[%i,%i]' % (i,j)) for i in N for j in N if j!=i } 
    if model == 3 or model == 4:
      yp = { (i) : solver.BoolVar('yp[%i]' % (i)) for i in V }
      wp = { (i,j) : solver.BoolVar('wp[%i,%i]' % (i,j)) for i in N for j in N if j!=i } 
  else :
    if model == 2 or model == 4 :
      y = { (i) : solver.NumVar(0, 1, 'y[%i]' % (i)) for i in V }
      w = { (i,j) : solver.NumVar(0, 1, 'w[%i,%i]' % (i,j)) for i in N for j in N if j!=i } 
    if model == 3 or model == 4:
      yp = { (i) : solver.NumVar(0, 1, 'yp[%i]' % (i)) for i in V }
      wp = { (i,j) : solver.NumVar(0, 1, 'wp[%i,%i]' % (i,j)) for i in N for j in N if j!=i } 


  # Escribimos el modelo matemático

  # Función objetivo
  if objetive == 'Maximize' :
    if model == 2 or model == 4 :
      solver.Maximize(solver.Sum( abs(areas[t])*y[t] for t in V )) 
    else :
      solver.Maximize(area_CH - solver.Sum( abs(areas[t])*yp[t] for t in V )) 
  else :
    if model == 2 or model == 4 :
      solver.Minimize(solver.Sum( abs(areas[t])*y[t] for t in V )) # Maximize o Minimize 
    else :
      solver.Minimize(area_CH - solver.Sum( abs(areas[t])*yp[t] for t in V )) 
    

  # RESTRICCIONES
  
  # Restricción del número de triángulos
  if num_triangles == 1 :
    if model == 2 or model == 4 :
      solver.Add( solver.Sum( y[t] for t in V) == nP - 2 ) 
    if model == 3 or model == 4 :
      solver.Add( solver.Sum( yp[t] for t in V) == nP - nCH ) 

  if directed_x == 1 :
    # Restricciones (2). Restricciones de grado del ATSP
    [ solver.Add( solver.Sum( xd[i,j] for j in N if j!=i) == 1 )  for i in N ]
    [ solver.Add( solver.Sum( xd[j,i] for j in N if j!=i) == 1 )  for i in N ]
  else :
    # [ solver.Add( x[i,j] == xd[i,j] + xd[j,i]) for i in N for j in N if i<j ]
    # Restricciones (31). Restricciones de grado del TSP     
    [ solver.Add( solver.Sum( x[i,j] for j in N if i<j) + solver.Sum( x[j,i] for j in N if j<i) == 2 )  for i in N ]
  
  # Restricciones de subciclos (3) (o equivalentes)  
  # [ solver.Add( u[j] >= u[i] + xd[i,j] - (nP-2)*(1-xd[i,j]) ) for i in N0 for j in N0 if j!=i ]

  [ solver.Add( solver.Sum(f[j,i]-f[i,j] for j in N if j!=i) == 1 )  for i in N0 ]
  if directed_x == 1 :
    [ solver.Add(f[i,j] <= (nP-1)*xd[i,j])    for i in N for j in N if i!=j ]
  else :
    [ solver.Add( f[i,j] + f[j,i] <= (nP-1)*x[i,j])    for i in N for j in N if i<j ]
    
 

  # Por comodidad vamos a utilizar estas variables como la suma de todos los triángulos orientados
  # que comparten un arco
  if model == 2 or model == 4 :
    [ solver.Add( w[i,j] == solver.Sum(y[t] for t in ta_arc[i][j])) for i in N for j in N if j!=i ]
  if model == 3 or model == 4 :
    [ solver.Add( wp[i,j] == solver.Sum(yp[t] for t in ta_arc[i][j])) for i in N for j in N if j!=i ]

  

  # Restricciones de cruces (5)
  if seg_cruzan == 1 :
    if directed_x == 1 :
      [ solver.Add( xd[SC[i][0],SC[i][1]] + xd[SC[i][1],SC[i][0]] + xd[SC[i][2],SC[i][3]] + xd[SC[i][3],SC[i][2]] <= 1 ) 
                    for i in range(nSC) ]   
    else :
      #Cada fila de SC tiene dos pares de nodos donde hemos colocado primero el menor
      [ solver.Add( x[SC[i][0],SC[i][1]] + x[SC[i][2],SC[i][3]] <= 1 ) for i in range(nSC) ]
    # También se pueden poner con las variables w 
    # [ solver.Add( w[SC[i][0],SC[i][1]] + w[SC[i][2],SC[i][3]] <= 1 )
    # [ solver.Add( w[SC[i][0],SC[i][1]] + w[SC[i][3],SC[i][2]] <= 1 )
    # [ solver.Add( w[SC[i][1],SC[i][0]] + w[SC[i][2],SC[i][3]] <= 1 )
    # [ solver.Add( w[SC[i][1],SC[i][0]] + w[SC[i][3],SC[i][2]] <= 1 )

  # Restricciones que relacionan las variables del tour con los triángulos
  # Primero ponemos las que están en la envolvente convexa
  if model == 2 or model == 4:
    if directed_x == 1 : # Restricciones (20)
      [ solver.Add( w[CH[i],CH[i+1]] == xd[CH[i],CH[i+1]] ) for i in range(nCH-1)]
      solver.Add( w[CH[nCH-1],CH[0]] == xd[CH[nCH-1],CH[0]])
    else :  # Restricciones (35)
      [ solver.Add( w[CH[i],CH[i+1]] == x[CH[i],CH[i+1]]) for i in range(nCH-1) if CH[i]<CH[i+1] ]
      [ solver.Add( w[CH[i],CH[i+1]] == x[CH[i+1],CH[i]]) for i in range(nCH-1) if CH[i]>CH[i+1] ]
      if CH[nCH-1] < CH[0] :
        solver.Add( w[CH[nCH-1],CH[0]] == x[CH[nCH-1],CH[0]])
      else :        
        solver.Add( w[CH[nCH-1],CH[0]] == x[CH[0],CH[nCH-1]])
  if model == 3 or model == 4:
    if directed_x == 1 : # Restricciones (24)
      [ solver.Add( wp[CH[i],CH[i+1]] == 1 - xd[CH[i],CH[i+1]] ) for i in range(nCH-1)]
      solver.Add( wp[CH[nCH-1],CH[0]] == 1 - xd[CH[nCH-1],CH[0]])
    else :  # Restricciones (38)
      [ solver.Add( wp[CH[i],CH[i+1]] == 1 - x[CH[i],CH[i+1]]) for i in range(nCH-1) if CH[i]<CH[i+1] ]
      [ solver.Add( wp[CH[i],CH[i+1]] == 1 - x[CH[i+1],CH[i]]) for i in range(nCH-1) if CH[i]>CH[i+1] ]
      if CH[nCH-1] < CH[0] :
        solver.Add( wp[CH[nCH-1],CH[0]] == 1 - x[CH[nCH-1],CH[0]])
      else :        
        solver.Add( wp[CH[nCH-1],CH[0]] == 1 - x[CH[0],CH[nCH-1]])
  # Luego las que no están en la envolvente convexa
  if model == 2 or model == 4:
    if directed_x == 1 : # Restricciones (21)
      [ solver.Add( w[i,j] - w[j,i] == xd[i,j] - xd[j,i] ) for i in N for j in N if i !=j and (estaEnCH[i] == 0 or estaEnCH[j] == 0)]
    else :  # Restricciones (36)
      [ solver.Add( w[i,j] - w[j,i] >= -x[i,j] ) for i in N for j in N if i<j and (estaEnCH[i] == 0 or estaEnCH[j] == 0)]
      [ solver.Add( w[i,j] - w[j,i] <=  x[i,j] ) for i in N for j in N if i<j and (estaEnCH[i] == 0 or estaEnCH[j] == 0)]
  if model == 3 or model == 4:
    if directed_x == 1 : # Restricciones (25)
      [ solver.Add( wp[i,j] - wp[j,i] == xd[j,i] - xd[i,j] ) for i in N for j in N if i !=j and (estaEnCH[i] == 0 or estaEnCH[j] == 0)]
    else :  # Restricciones (39)
      [ solver.Add( wp[i,j] - wp[j,i] >= -x[i,j] ) for i in N for j in N if i<j and (estaEnCH[i] == 0 or estaEnCH[j] == 0)]
      [ solver.Add( wp[i,j] - wp[j,i] <=  x[i,j] ) for i in N for j in N if i<j and (estaEnCH[i] == 0 or estaEnCH[j] == 0)]
  # Finalmente otras acotaciones para las variables que no están en la envolvente convexa
  if model == 2 or model == 4:
    if directed_x == 1 : # Restricciones (22)
      [ solver.Add( w[i,j] >= xd[i,j] ) for i in N for j in N if i !=j and (estaEnCH[i] == 0 or estaEnCH[j] == 0)]
      [ solver.Add( w[i,j] <= 1 - xd[j,i] ) for i in N for j in N if i !=j and (estaEnCH[i] == 0 or estaEnCH[j] == 0)]
    else :  # Restricciones (37)
      [ solver.Add( w[i,j] + w[j,i] >= x[i,j] ) for i in N for j in N if i<j and (estaEnCH[i] == 0 or estaEnCH[j] == 0)]
      [ solver.Add( w[i,j] + w[j,i] <= 2 - x[i,j] ) for i in N for j in N if i<j and (estaEnCH[i] == 0 or estaEnCH[j] == 0)]
  if model == 3 or model == 4:
    if directed_x == 1 : # Restricciones (26)
      [ solver.Add( wp[i,j] >= xd[j,i] ) for i in N for j in N if i !=j and (estaEnCH[i] == 0 or estaEnCH[j] == 0)]
      [ solver.Add( wp[i,j] <= 1 - xd[i,j] ) for i in N for j in N if i !=j and (estaEnCH[i] == 0 or estaEnCH[j] == 0)]
    else :  # Restricciones (40)
      [ solver.Add( wp[i,j] + wp[j,i] >= x[i,j] ) for i in N for j in N if i<j and (estaEnCH[i] == 0 or estaEnCH[j] == 0)]
      [ solver.Add( wp[i,j] + wp[j,i] <= 2 - x[i,j] ) for i in N for j in N if i<j and (estaEnCH[i] == 0 or estaEnCH[j] == 0)]

  # Creo que es necesario esto para garantizar la orientación 
  # [ solver.Add( u[CH[i]] <= u[CH[i+1]] - 1) for i in range(1,nCH-1) ]

  ## Imprimir el modelo en formato LP para debugger
  # print(solver.ExportModelAsLpFormat(False))

  # Resolvemos el problema
  global status
  status = solver.Solve()

  global tiempo
  global area_total
  tiempo = solver.WallTime()/1000
  area_total = 0
  if status == pywraplp.Solver.OPTIMAL :
    area_total = solver.Objective().Value()
    print('Area: ', area_total, '  Tiempo: ', tiempo)
    if directed_x == 0 :
      selected = [(i,j) for i in N for j in N if i<j if x[i,j].solution_value() > EPS]
    else :
      selected = [(i,j) for i in N for j in N if i!=j if xd[i,j].solution_value() > EPS]

    G = nx.Graph()
    G.add_edges_from( selected )
    Components = list(nx.connected_components(G))
    if len(Components) > 1 :

      if model == 2 or model == 4 :
        for t in V :
          if y[t].solution_value() > EPS :
            print(f"Triángulo interior {t} ({triangulos[t][0]},{triangulos[t][1]},{triangulos[t][2]}), Area: {areas[t]}, y[t]={y[t].solution_value()}")
      if model == 3 or model == 4 :
        for t in V :
          if yp[t].solution_value() > EPS :
            print(f"Triángulo exterior {t} ({triangulos[t][0]},{triangulos[t][1]},{triangulos[t][2]}), Area: {areas[t]}, y[t]={yp[t].solution_value()}")
      # print('Area: ', area_total, ' Distancia: ', computa_distancia_ruta(dist, selected), 'Tiempo: ', tiempo)
      # print(selected)
      dibuja(selected)

    if write_lp:
      lp_filename = 'outputs/Others/' + nombre_fichero.split('.')[0].split('/')[-1] + '_lp_' + objetive + '_m' + str(model) + '_directedx' + str(directed_x) + '_cross' + str(seg_cruzan) + '_binariey' + str(y_binarias) + '_nT' + str(num_triangles) + '.lp'
      print(f"Saving LP model to '{lp_filename}'")
      lp_model_str = solver.ExportModelAsLpFormat(False)
      with open(lp_filename, 'w') as lp_file:
          lp_file.write(lp_model_str)
      print("LP model saved.")

  elif status == pywraplp.Solver.FEASIBLE :
    area_total = solver.Objective().Value()
    print('Area: ', area_total, '  Tiempo: ', tiempo)
    #print('Area: ', area_total, ' Distancia: ', computa_distancia_ruta(dist, selected), 'Tiempo: ', tiempo)
    # if directed_x == 0 :
    #   selected = [(i,j) for i in N for j in N if i!=j if x[i,j].solution_value() > EPS]
    # else :
    #     selected = [(i,j) for i in N for j in N if i!=j if xd[i,j].solution_value() > EPS]
    #dibuja(selected)



  else:
    print('Este problema no tiene solución factible')


def tsp(objetive, lp_solver, seg_cruzan):
  selected = {} 
  if lp_solver == 'GUROBI' :
    solver = pywraplp.Solver('Modelo', pywraplp.Solver.GUROBI_MIXED_INTEGER_PROGRAMMING)
  else :
    solver = pywraplp.Solver('Modelo', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

  solver.SetTimeLimit(7200000)
  # solver.parameters.max_time_in_seconds = 10.0

  # Definimos las variables de decisión
  
  # 1 - Si el segmento (i,j) está en el perímetro
  x = { (i,j) : solver.BoolVar('x[%i,%i]' % (i,j)) for i in N for j in N if j!=i } 
  # Posición-1 del punto i en la ruta
  u = { i : solver.NumVar(0.0, nP-2, 'u[%i]' % i) for i in N0 }




  # Escribimos el modelo matemático

  # Función objetivo
  if objetive == 'Maximize' :
      solver.Maximize(solver.Sum(dist[i,j] * x[i,j]  for i in N for j in N if j!=i )) 
  else :
      solver.Minimize(solver.Sum(dist[i,j] * x[i,j]  for i in N for j in N if j!=i ))
    


  if seg_cruzan == 0 :
    [ solver.Add( x[i,j] + x[j,i] <= 1 ) for i in N for j in N if i<j ]
  else :
    [ solver.Add( x[CH[i],CH[j]] == 0 ) for i in range(nCH) for j in range(nCH) if (i-j > 1 and (i!=nCH-1 or j!=0 )) or (j-i > 1 and (i!=0 or j!=nCH-1 ))]
    [ solver.Add( x[SC[i][0],SC[i][1]] + x[SC[i][1],SC[i][0]] + x[SC[i][2],SC[i][3]] + x[SC[i][3],SC[i][2]] <= 1 ) 
                    for i in range(nSC) ]

  # Restricción 5
  [ solver.Add( solver.Sum( x[i,j] for j in N if j!=i) == 1 )  for i in N ]
  [ solver.Add( solver.Sum( x[j,i] for j in N if j!=i) == 1 )  for i in N ]
  # Restricción 6
  [ solver.Add( u[j] >= u[i] + x[i,j] - (nP-2)*(1-x[i,j]) + (nP-3)*x[j,i] ) for i in N0 for j in N0 if j!=i ]

  # Resolvemos el problema
  global status

  status = solver.Solve()
   
  global tiempo
  global area_total
  tiempo = solver.WallTime()/1000
  area_total = 0
  if status == pywraplp.Solver.OPTIMAL :
    area_total = solver.Objective().Value()
    selected = [(i,j) for i in N for j in N if i!=j if x[i,j].solution_value() > EPS]
    print('Area: ', area_total, ' Distancia: ', computa_distancia_ruta(dist, selected), 'Tiempo: ', tiempo)
    print(selected)
    dibuja(selected)
  elif status == pywraplp.Solver.FEASIBLE :
    area_total = solver.Objective().Value()
    print('Area: ', area_total, ' Distancia: ', computa_distancia_ruta(dist, selected), 'Tiempo: ', tiempo)
    #selected = [(i,j) for i in S for j in S if i!=j if x[i,j].solution_value() > EPS]
    #dibuja(selected)
  else:
    print('Este problema no tiene solución factible')



valores_optimos = [ 
    [ [9096, 2736, 5766, 6882, 2832, 4766, 3276, 5608, 4958, 8018],   # MinArea tamaño 10
      [7668, 5512, 6850, 7572, 3764, 4622, 5092, 4202, 7942, 4244] ], # MinArea tamaño 15
    [ [17532, 12140, 18296, 16918, 18162, 9832, 15828, 17534, 13900, 23282], # MaxArea tamaño 10
      [26248, 21592, 22368, 27324, 22918, 17076, 19760, 18558, 21028, 23562] ] # MaxArea tamaño 15
  ]


fichero_resultados = "results.txt"

ruta_instancias = "outputs/Pre-files"


''' Programa principal  '''
f = open(fichero_resultados, 'w')
f.write("Objetivo\tSolver\tModelo\tNombre\tn\tdirect\tcruzan\ty_bin\tnum_tri\tstatus\tTiempo\tArea\tError\n")
f.close()

#opciones_objetivo = ['Maximize', 'Minimize']
#solucionadores = ['GUROBI']   # se puede incluir CBC y GUROBI
#modelos = [2, 3, 4]
#tamanios = [10]
#tamanios = [10,15,20]
#semillas = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
# semillas = ['04', '05', '06', '07', '08', '09', '10']

## opciones = [ directed, cruces, y_binarias, num_triangulos]
# Combinación para probar si los cruces son necesarios o no
# opciones = [ 
#              [1, 0, 1, 1],  # Dirigido, sin cruces y binarias  
#              [0, 0, 1, 1],  # No dirigido, sin cruces y binarias  
#              [1, 0, 1, 0],  # Dirigido, sin cruces y binarias  
#              [0, 0, 1, 0]  # No dirigido, sin cruces y binarias  
#            ]

# Combinación para probar la relajación de las variables y
# opciones = [ 
#              [1, 1, 0, 1],  
#              [0, 1, 0, 1],  
#              [1, 1, 0, 0],  
#              [0, 1, 0, 0]  
#            ]

# Combinación sin cruces y con la relajación de las variables y
#opciones = [ 
#             [1, 0, 0, 1],  
#             [1, 0, 0, 0],  
#             [0, 0, 0, 0]  
#             [0, 0, 0, 1],  
#           ]

opciones_objetivo = ['Maximize']
solucionadores = ['GUROBI']   # se puede incluir CBC y GUROBI
modelos = [4]
tamanios = [25]
semillas = ['0000025']#,
opciones = [  
     [1, 0, 1, 1],  # Dirigido, sin cruces, binarias y con número de triángulos  
    [1, 0, 0, 0],  # Dirigido, sin cruces, relajación de y y sin número de triángulos
  #  [0, 1, 0, 0],
  #  [0, 0, 1, 0],
  #  [0, 0, 0, 1],
    [1, 1, 1, 1], # Dirigido, con cruces, binarias y con número de triángulos
    [1, 0, 1, 0], # Dirigido, sin cruces, binarias y sin número de triángulos
    [1, 1, 0, 0], # Dirigido, con cruces, relajación de y y sin número de triángulos
  #  [0, 1, 1, 0],
    [1, 0, 0, 1], # Dirigido, sin cruces, relajación de y y con número de triángulos
    [0, 1, 0, 1], # No dirigido, con cruces, relajación de y y con número de triángulos
  #  [0, 0, 1, 1]
    ]
# Combinación para probar la relajación de las variables y
#opciones = [ 
#              [1, 1, 0, 1],  
#              [0, 1, 0, 1],  
#              [1, 1, 0, 0],  
#              [0, 1, 0, 0]  
#            ]

# Combinación sin cruces y con la relajación de las variables y
#opciones = [ 
#             [1, 0, 0, 1],  
#             [1, 0, 0, 0],  
#             [0, 0, 0, 0],  
#             [0, 0, 0, 1],  
#           ]

write_lp = True

for sol in solucionadores :
  for tam in tamanios :
    for s in semillas : 
      nombre_fichero = 'n' + str(tam) + 's' + s + '.pre' 
      lectura_fichero(ruta_instancias + '/' + nombre_fichero)
      #print('\n')
      print('---------',nombre_fichero, '--------- ')
      for obj in opciones_objetivo :
        for m in modelos :
          for opt in opciones :
            print('Solving ' + obj + ' ' + sol + ' model ' + str(m) + ' ' + str(opt))
            modelo(obj, sol, m, opt[0], opt[1], opt[2], opt[3], write_lp)
            f = open(fichero_resultados, 'a')
            f.write(obj)
            f.write("\t")
            f.write(sol)
            f.write("\t")
            f.write(str(m))
            f.write("\t")
            f.write(nombre_fichero)
            f.write("\t")
            f.write(str(nP))
            f.write("\t")
            f.write(str(opt[0]))
            f.write("\t")
            f.write(str(opt[1]))
            f.write("\t")
            f.write(str(opt[2]))
            f.write("\t")
            f.write(str(opt[3]))
            f.write("\t")
            f.write(str(status))
            f.write("\t")
            f.write(str(tiempo))
            f.write("\t")
            f.write(str(area_total))
            f.write("\t")
            index_obj = 1
            if(obj == 'Minimize'):
              index_obj = 0
            index_tam = int((tam-10)/5)
            index_sem = int(s) - 1  
            #f.write(str(valores_optimos[index_obj][index_tam][index_sem]))
            f.write("\n")
            f.close()

# Valores óptimos de las áreas MinArea y MaxArea
# uniform-0000010-1	    9096	17532
# uniform-0000010-2	    2736	12140
# uniform-0000010-3	    5766	18296
# uniform-0000010-4	    6882	16918
# uniform-0000010-5	    2832	18162
# uniform-0000010-6	    4766	9832
# uniform-0000010-7	    3276	15828
# uniform-0000010-8	    5608	17534
# uniform-0000010-9	    4958	13900
# uniform-0000010-10	8018	23282
# uniform-0000015-1	    7668	26248
# uniform-0000015-2	    5512	21592
# uniform-0000015-3	    6850	22368
# uniform-0000015-4	    7572	27324
# uniform-0000015-5	    3764	22918
# uniform-0000015-6	    4622	17076
# uniform-0000015-7	    5092	19760
# uniform-0000015-8	    4202	18558
# uniform-0000015-9	    7942	21028
# uniform-0000015-10	4244	23562

