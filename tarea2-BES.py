import random as rnd
import math

#################################################################################################
#############################      PROBLEM      #################################################
#################################################################################################

#Se definen variables del problema
class Problem:
  def __init__(self):
    self.dimension    = 6
    self.valorization = [65,90,40,60,20]
    self.cost         = [150,300,40,100,10]
    self.domains = [
      self.domain(0, 13),   # dom(x1) = {0, 1, ... ,12}
      self.domain(0, 7),    # dom(x2) = {0, 1, ... ,6}
      self.domain(0, 26),   # dom(x3) = {0, 1, ... ,25}
      self.domain(0, 5),    # dom(x4) = {0, 1, ... ,4}
      self.domain(0, 31),   # dom(x5) = {0, 1, ... ,30}
      self.domain(0, 2)     # dom(y)  = {0, 1}
    ]

  #Creación de listas de dominios
  def domain(self, min, max):
    return list(range(min, max))
  
  #Verificacion de restricciones binarias
  def checkConstraint(self, x):
    if not (x[1] <= 10 * (1 - x[5])):
      return False
    elif not (x[2] <= 25 * x[5]):
      return False
    elif not (x[0] + x[1] <= 20):
      return False
    elif not (((150 * x[0]) + (300 * x[1])) <= 1800):
      return False
    elif not ((1000 * x[0]) + (2000 * x[1]) + (1500 * x[2]) + (2500 * x[3]) + (300 * x[4])) <= 50000:
      return False
    #si todas las restricciones se cumplen, retornamos True
    return True

  #Evaluacion de la funcion multiobjetivo
  def eval(self, x):
    maxValue = 0
    minValue = 0
    for i in range(self.dimension-1):
      maxValue = maxValue + x[i] * self.valorization[i]
      minValue = minValue + x[i] * self.cost[i]

    return (0.5 * ((maxValue) / 2153.33)) + (0.5 * ((3006.67 - minValue) / 3006.67))
  
################################################################################################
##########################           AGENTE           ##########################################
################################################################################################

class Agent(Problem):
  def __init__(self):
    self.p = Problem()
    self.x = []
    self.y = self.p.dimension - 1
    self.yr = 0
    self.xr = 0
    self.yi = 0
    self.xi = 0
    self.x1 = 0
    self.y1 = 0

    for domain in (self.p.domains):
      self.x.append(rnd.choice(domain))
      #self.x.append(rnd.randint(domain[0],max(domain) + 1))  # +1 para que tome el ult

  def isFeasible(self):
    return self.p.checkConstraint(self.x)

  # Dependiendo del problema (min o max) (en este caso es max)
  def isBetterThan(self, g):
    return self.fit() > g.fit()

  def fit(self):
    return self.p.eval(self.x)

  #### Funciones de movimiento ####################################################################################################################################
  def moveSelectStage(self, pBest, pMean, alpha):
    for j in range(self.p.dimension):
      self.x[j] = self.toInteger((pBest.x[j] + alpha * rnd.random() * (pMean.x[j] - self.x[j])), min(self.p.domains[j]), max(self.p.domains[j]))
      #  Pi     =                  Pbest     + alpha *       rand   * (  Pmean    - Pi)
  
  def moveSearchInSpace(self, pMean, pNext):
    for j in range(self.p.dimension):
      self.x[j] = self.toInteger((self.x[j] + self.yi * (self.x[j] - pNext.x[j]) + self.xi * (self.x[j] - pMean.x[j])),min(self.p.domains[j]), max(self.p.domains[j]))
      # Pi      =                       Pi  +    yi   * (   Pi     -   P(i+1)  ) +    xi   * (    Pi    - Pmean)
  
  def moveSwoop(self, pBest, pMean, c1, c2):
    for j in range(self.p.dimension - 1):
      self.x[j] = self.toInteger((rnd.random() * pBest.x[j] + self.x1 * (self.x[j] - c1 * pMean.x[j]) + self.y1 * (self.x[j] - c2 * pMean.x[j])),min(self.p.domains[j]), max(self.p.domains[j]))
      # Pi      =                       rand   *  pBest     +   x1    * (    Pi     - c1 * pMean    ) +   y1    *     Pi     - c2 * pMean     )
  ####################################################################################################################################################################

  def toInteger(self, x, domain_min, domain_max):
    sigmoid_value = 1 / (1 + math.exp(-x))
    normalized_value = sigmoid_value / (1 + math.exp(-domain_max)) * rnd.random() 
    adjusted_value = normalized_value * (domain_max - domain_min) + domain_min
    return math.ceil(adjusted_value) if normalized_value > rnd.random() else math.floor(adjusted_value)

  def __str__(self) -> str:
    return f"fit:{self.fit()} x:{self.x}"

  def copy(self, a):
    self.x = a.x.copy()
    self.xi = a.xi
    self.yi = a.yi
    self.xr = a.xr
    self.yr = a.yr
    self.x1 = a.x1
    self.y1 = a.y1

#################################################################################################
#############################      SWARM        #################################################
#################################################################################################

class Swarm:
  def __init__(self):
    self.maxIter  = 70
    self.nAgents  = 25
    self.swarm    = []
    self.pBest    = Agent()
    self.pMean    = Agent()  
    self.R        = rnd.uniform(0.5, 2)
    self.a        = rnd.uniform(5, 10)
    self.alpha    = rnd.uniform(1.5, 2)
    self.c1       = rnd.randint(1,2)
    self.c2       = rnd.randint(1,2)
    self.maxXr    = 0
    self.maxYr    = 0

  # Funcion para actualizar el agente pMean al inicio de cada iteración ( pMean es requerido por todos los movimientos)
  def updateMean(self):
    for j in range(len(self.pMean.x)):
      sum = 0
      for i in range(self.nAgents):
        sum = sum + self.swarm[i].x[j]
      self.pMean.x[j] = sum / self.nAgents
 
  # Funcion para actualizar valores del agente requeridos para aplicar el segundo movimiento a todos los agentes
  def updateValuesForMove2(self):
    for i in range(self.nAgents):
      theta = self.a * math.pi * rnd.random()
      ri = theta * self.R * rnd.random()
      self.swarm[i].xr = ri * math.sin(theta)
      self.swarm[i].yr = ri * math.cos(theta)

      if self.swarm[i].xr > self.maxXr:
        self.maxXr = self.swarm[i].xr         
      if self.swarm[i].yr > self.maxYr:
        self.maxYr = self.swarm[i].yr

    for i in range(self.nAgents):
      self.swarm[i].xi = self.swarm[i].xr / self.maxXr
      self.swarm[i].yi = self.swarm[i].yr / self.maxYr

  # Funcion para actualizar valores del agente requeridos para aplicar el tercer movimiento a todos los agentes
  def updateValuesForMove3(self):
    for i in range(self.nAgents):
      theta = self.a * math.pi * rnd.random()
      self.swarm[i].xr = theta * math.sinh(theta)
      self.swarm[i].yr = theta * math.cosh(theta)

      if self.swarm[i].xr > self.maxXr:
        self.maxXr = self.swarm[i].xr         
      if self.swarm[i].yr > self.maxYr:
        self.maxYr = self.swarm[i].yr

      for i in range(self.nAgents):
        self.swarm[i].x1 = self.swarm[i].xr / self.maxXr
        self.swarm[i].y1 = self.swarm[i].yr / self.maxYr

  def solve(self):
    self.initRand()
    self.evolve()
    self.bestToConsole()

  def initRand(self):
    #print("  -->  initRand  <-- ")
    for i in range(self.nAgents):
      while True:
        a = Agent()
        if a.isFeasible():
          break
      self.swarm.append(a)
    self.pBest.copy(self.swarm[0])
    for i in range(1, self.nAgents):
      if self.swarm[i].isBetterThan(self.pBest):
        self.pBest.copy(self.swarm[i])

    #self.swarmToConsole()
    #self.bestToConsole()

  # Funcion evolucionar : en los 3 movimientos, por cada agente de la población es copiado en un agente auxiliar, 
  # se aplica el movimiento hasta que sea factible y si es mejor que el original, se actualiza el agente, 
  # si es mejor que el pBest se actualiza el pBest

  def evolve(self):
    #print("  -->  evolve  <-- ")
    t = 1
    a = Agent()
    while t <= self.maxIter:
      self.updateMean()
      # MOVEMENT 1 : SELECT SPACE ####################################################################################
      for i in range(self.nAgents):
        #print("i: ", i)
        a.copy(self.swarm[i])
        while True:
          a.moveSelectStage(self.pBest, self.pMean, self.alpha)
          if a.isFeasible():
            break
        if a.isBetterThan(self.swarm[i]):
          self.swarm[i].copy(a)
          if a.isBetterThan(self.pBest):
            self.pBest.copy(a)
      # MOVEMENT 2 : SEARCH IN SPACE #################################################################################
      self.updateValuesForMove2()
      for i in range(self.nAgents):
        a.copy(self.swarm[i])
        while True:
          if i < self.nAgents-1:
            a.moveSearchInSpace(self.pMean, self.swarm[i + 1])
          else:
            a.moveSearchInSpace(self.pMean, self.swarm[0])
          if a.isFeasible():
            break
        if a.isBetterThan(self.swarm[i]):
          self.swarm[i].copy(a)
          if a.isBetterThan(self.pBest):
            self.pBest.copy(a)
      # MOVEMENT 3 : SWOOP ###########################################################################################
      self.updateValuesForMove3()
      for i in range(self.nAgents):
        a.copy(self.swarm[i])
        while True:
          a.moveSwoop(self.pBest,self.pMean,self.c1,self.c2)
          if a.isFeasible():
            break
        if a.isBetterThan(self.swarm[i]):
          self.swarm[i].copy(a)
          if a.isBetterThan(self.pBest):
            self.pBest.copy(a)
      #self.swarmToConsole()
      #self.bestToConsole()
      t = t + 1

  def swarmToConsole(self):
    print(" -- Swarm --")
    for i in range(self.nAgents):
      print(f"{self.swarm[i]}")

  def bestToConsole(self):
    #print(" -- Best --")
    print(f"{self.pBest}")

#import time
for i in range(30):
#  inicio = time.time()
  try:
    Swarm().solve()
  except Exception as e:
    print(f"{e} \nCaused by {e.__cause__}, {e.__le__}")
#fin = time.time()
#  print(fin-inicio)
print("OK")