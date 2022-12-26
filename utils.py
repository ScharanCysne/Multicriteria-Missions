import copy
import math
import numpy as np
import pygame
import random
import matplotlib.pyplot as plt

from math      import exp 
from constants import *

def distance(x0, y0, x1, y1):
    return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)

def distance(drone_0, drone_1):
    pos_x_0, pos_y_0 = drone_0.location
    pos_x_1, pos_y_1 = drone_1.location
    return np.sqrt((pos_x_0 - pos_x_1)**2 + (pos_y_0 - pos_y_1)**2)

def normalFunction(omega, center, position):
    f = exp( -omega*((position.x - center.x) + (position.y - center.y)))
    return f

def bivariateFunction(alpha, beta, center, position):
    '''
        Calculates the bivariate function
        
        position: (x,y)
        center of the function: (xc,yc)
        control variables: Alpha and Beta will control the stringthof the vectors in x and y directions
        return: point in the bivariate function
    '''
    k = 10
    f = exp( -alpha*(position.x - center.x)/k**2 - beta*(position.y - center.y)/k**2 )
    return f
 
def derivativeBivariate(alpha, beta, center, position):
    '''
        Calculates the bivariate function
        
        position: (x,y)
        center of the function: (xc,yc)
        control variables: Alpha and Beta will control the stringthof the vectors in x and y directions
        return: point in the derivative of the bivariate function
    '''
    f = bivariateFunction(alpha, beta, center, position)
    dx = f * (-2*alpha*(position.x-center.x))
    dy = f * (-2*beta*(position.y-center.y))
    return pygame.math.Vector2(dx,dy)

def constrain_ang(ang, min_angle, max_angle):
    ang = min(ang, max_angle)
    ang = max(ang, min_angle)
    return ang

def limit(v2, max):
    """
        Limits magnitude of vector2

        :param v2: Vector2 to be normalized
        :type v2: pygame.Vector2
        :param max: maximum length of vector
        :type max: int
        :return v: returns vector 
        :rtype v: vector2
    """
    v = copy.deepcopy(v2)
    if v.length() > max:
        v.scale_to_length(max)
    return v

def constrain(v2, w, h):
    """
        Constrains movement of drone inside the canvas

        :param v2: Vector2 to be constrained
        :type v2: pygame.Vector2
        :param w: maximum width
        :type w: int
        :param h: maximum height
        :type h: int
        :return v2: returns vector within the limits
        :rtype v2: vector2
    """
    v2.x = min(max(v2.x, 0), w)
    v2.y = min(max(v2.y, 0), h)
    return v2


class FlowField():
    def __init__(self, resolution):
        self.cols = SCREEN_WIDTH // resolution    # Cols of the grid
        self.rows = SCREEN_HEIGHT // resolution   # Rows of the grid
        self.resolution = resolution              # Resolution of grid relative to window width and height in pixels
        
        # create grid
        self.field = [[pygame.math.Vector2(random.uniform(0,1),random.uniform(0,1)) for _ in range(self.cols)] for _ in range(self.rows)]  
        
    def draw(self, screen):
        blockSize = self.resolution #Set the size of the grid block
        for x in range(0, SCREEN_WIDTH, blockSize):
            for y in range(0, SCREEN_HEIGHT, blockSize):
                rect = pygame.Rect(x, y, blockSize, blockSize)
                pygame.draw.rect(screen, (200, 200, 200), rect, 1)

def intersection(drone_i, drone_j):
    d = (drone_i.location - drone_j.location).magnitude()
    d1 = d / 2
    return 2 * OBSERVABLE_RADIUS**2 * math.acos(d1/OBSERVABLE_RADIUS) - d1 * math.sqrt(OBSERVABLE_RADIUS**2 - d1**2)

def plot_results(NUM_DRONES, NUM_TIMESTEPS, robustness_level, algebraic_connectivity, area_coverage):
    x = list(range(NUM_TIMESTEPS))

    # Robustness Level
    y = robustness_level.mean(axis=0)
    plt.figure(f"Coverage Mission | {NUM_DRONES} Drones | Mean Robustness Level", figsize=(8, 2))
    plt.plot(x, robustness_level.mean(axis=0))
    # Compute and plot rolling mean with window of size EPISODE_WINDOW
    plt.xlim(0, x[-1])
    plt.title(f"Coverage Mission | {NUM_DRONES} Drones | Mean Robustness Level")
    plt.xlabel("Timestep")
    plt.ylabel("Robustness Level")
    plt.tight_layout()
    plt.savefig(f"RL_{NUM_DRONES}")

    # Algebraic Connectivity
    plt.figure(f"Coverage Mission | {NUM_DRONES} Drones | Mean Algebraic Connectivity", figsize=(8, 2))
    plt.plot(x, algebraic_connectivity.mean(axis=0))
    # Compute and plot rolling mean with window of size EPISODE_WINDOW
    plt.xlim(0, x[-1])
    plt.title(f"Coverage Mission | {NUM_DRONES} Drones | Mean Algebraic Connectivity")
    plt.xlabel("Timestep")
    plt.ylabel("Algebraic Connectivity")
    plt.tight_layout()
    plt.savefig(f"MAC_{NUM_DRONES}_AttacksB")

    # Area Coverage
    plt.figure(f"Coverage Mission | {NUM_DRONES} Drones | Mean Area Coverage Percentage", figsize=(8, 2))
    plt.plot(x, area_coverage.mean(axis=0) / (NUM_DRONES * np.pi * OBSERVABLE_RADIUS ** 2))
    # Compute and plot rolling mean with window of size EPISODE_WINDOW
    plt.xlim(0, x[-1])
    plt.title(f"Coverage Mission | {NUM_DRONES} Drones | Mean Area Coverage Percentage")
    plt.xlabel("Timestep")
    plt.ylabel("Mean Percentage")
    plt.tight_layout()
    plt.savefig(f"AC_{NUM_DRONES}B")