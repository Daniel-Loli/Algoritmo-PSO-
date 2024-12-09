import numpy as np
import matplotlib.pyplot as plt
import random

# --- Generación de Datos Realistas ---
NUM_CLIENTES = 15
NUM_VEHICULOS = 3
CAPACIDAD_VEHICULO = 1000  # Capacidad máxima de los vehículos en kg
ALMACEN = (50, 50)  # Ubicación de la base o almacén central

# Generar puntos de entrega y demandas de cada cliente
np.random.seed(42)
coordenadas_clientes = np.random.randint(0, 100, size=(NUM_CLIENTES, 2))  # Coordenadas (x, y)
demandas_clientes = np.random.randint(50, 300, size=NUM_CLIENTES)  # Peso de la carga

# --- Funciones Auxiliares ---

def calcular_distancia(p1, p2):
    """ Calcula la distancia euclidiana entre dos puntos """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calcular_costo_ruta(ruta, demandas):
    """ Calcula la distancia total recorrida para la ruta """
    distancia_total = 0
    capacidad_utilizada = 0
    punto_actual = ALMACEN
    
    for cliente in ruta:
        distancia_total += calcular_distancia(punto_actual, coordenadas_clientes[cliente])
        capacidad_utilizada += demandas[cliente]
        
        if capacidad_utilizada > CAPACIDAD_VEHICULO: 
            penalizacion = 1000 * (capacidad_utilizada - CAPACIDAD_VEHICULO)
            return distancia_total + penalizacion
        
        punto_actual = coordenadas_clientes[cliente]
    
    distancia_total += calcular_distancia(punto_actual, ALMACEN)  # Regreso al almacén
    return distancia_total

# --- Clase Partícula ---

class Particula:
    def __init__(self):
        self.posicion = list(np.random.permutation(NUM_CLIENTES))
        self.velocidad = np.zeros(NUM_CLIENTES)
        self.mejor_posicion = self.posicion.copy()
        self.mejor_costo = calcular_costo_ruta(self.posicion, demandas_clientes)
        self.costo_actual = self.mejor_costo

    def actualizar_velocidad(self, mejor_global):
        w, c1, c2 = 0.7, 1.5, 1.5
        r1, r2 = np.random.rand(), np.random.rand()
        self.velocidad = (w * self.velocidad 
                          + c1 * r1 * (np.array(self.mejor_posicion) - np.array(self.posicion)) 
                          + c2 * r2 * (np.array(mejor_global) - np.array(self.posicion)))
    
    def actualizar_posicion(self):
        for i in range(len(self.posicion)):
            if np.random.rand() < abs(self.velocidad[i]):
                swap_idx = random.randint(0, len(self.posicion) - 1)
                self.posicion[i], self.posicion[swap_idx] = self.posicion[swap_idx], self.posicion[i]
        
        self.costo_actual = calcular_costo_ruta(self.posicion, demandas_clientes)
        
        if self.costo_actual < self.mejor_costo:
            self.mejor_posicion = self.posicion.copy()
            self.mejor_costo = self.costo_actual

# --- Algoritmo PSO ---

resultados_iteraciones = []  # Lista para almacenar los resultados de cada iteración

def pso(num_particulas=50, num_iteraciones=100):
    particulas = [Particula() for _ in range(num_particulas)]
    mejor_global = min(particulas, key=lambda p: p.mejor_costo).mejor_posicion
    mejor_costo_global = min(particulas, key=lambda p: p.mejor_costo).mejor_costo
    
    plt.ion()  # Modo interactivo de matplotlib
    
    for iteracion in range(num_iteraciones):
        for particula in particulas:
            particula.actualizar_velocidad(mejor_global)
            particula.actualizar_posicion()
        
        mejor_particula = min(particulas, key=lambda p: p.costo_actual)
        if mejor_particula.costo_actual < mejor_costo_global:
            mejor_global = mejor_particula.posicion.copy()
            mejor_costo_global = mejor_particula.costo_actual
        
        # Almacenar resultados de la iteración
        resultados_iteraciones.append({
            'iteracion': iteracion + 1,
            'mejor_costo': mejor_costo_global,
            'mejor_ruta': mejor_global.copy()
        })
        
        print(f"Iteración {iteracion+1}/{num_iteraciones} - Mejor costo: {mejor_costo_global}")
        
        # --- Visualización Dinámica ---
        plt.clf()
        visualizar_ruta(mejor_global)
        plt.pause(0.1)
    
    plt.ioff()  # Desactivar modo interactivo
    return mejor_global, mejor_costo_global

# --- Visualización de la Ruta ---

def visualizar_ruta(ruta):
    plt.scatter(*ALMACEN, c='red', s=200, label='Almacén')
    for i, (x, y) in enumerate(coordenadas_clientes):
        plt.scatter(x, y, c='blue', s=100)
        plt.text(x + 1, y + 1, f'C{i}', fontsize=12, color='darkblue')
    
    x_coords, y_coords = [ALMACEN[0]], [ALMACEN[1]]
    for cliente in ruta:
        x, y = coordenadas_clientes[cliente]
        x_coords.append(x)
        y_coords.append(y)
    x_coords.append(ALMACEN[0])
    y_coords.append(ALMACEN[1])
    
    mitad = len(x_coords) // 2
    
    # Visualizar la ida (del almacén al último cliente)
    plt.plot(x_coords[:mitad + 1], y_coords[:mitad + 1], linestyle='-', color='green', linewidth=2, label='Ida')
    
    # Visualizar la vuelta (del último cliente al almacén)
    plt.plot(x_coords[mitad:], y_coords[mitad:], linestyle='-', color='red', linewidth=2, label='Vuelta')
    
    plt.title('Ruta en construcción')
    plt.legend()

# --- Ejecución del PSO ---

mejor_ruta, mejor_costo = pso(num_particulas=30, num_iteraciones=50)
print("\nMejor ruta encontrada:", mejor_ruta)
print("Costo de la mejor ruta:", mejor_costo)
