# Simulation Parameters
NUM_DRONES = 10                 # Number of simultaneous drones
SIZE_TRACK = 100
NUM_OBSTACLES = 50
OBSERVABLE_RADIUS = 16
AVOID_DISTANCE = 2

UPPER_X = 500
LOWER_X = 0
UPPER_Y = 50
LOWER_Y = 0

# Display Parameters
SCREEN_WIDTH = 1800
SCREEN_HEIGHT = 180
RESOLUTION = 30                 # of grid
SIZE_OBSTACLES = 3
SIZE_DRONE = 3
RATIO = SCREEN_WIDTH / UPPER_X  # Factor to convert from pixels to meters
TIME_MAX_SIMULATION = 15        # Time to stop simulation in case the conditions are not completed

# Sample Time Parameters
FREQUENCY = 60                  # simulation frequency
SAMPLE_TIME = 1.0 / FREQUENCY   # simulation sample time

# Behavior Parameters
FORWARD_SPEED = 0.6               # default linear speed when going forward
SEEK_FORCE = 0.5                  # max seek force
THRESHOLD_TARGET = SCREEN_WIDTH - 50 
MASS = 10                       # Drone Mass, used to calculate force

# Colors
WHITE = (255,255,255)
BLACK = (0,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)
RED = (255,0,0)
LIGHT_RED = (250,255,114)
LIGHT_GRAY = (232, 232, 232)
LIGHT_YELLOW = (255,255,102)

# keys
MOUSE_LEFT = 0
MOUSE_RIGHT = 2

# Execution Parameters
PENALTY_STEP = -1
PENALTY_DISCONNECTED = -100
TIMESTEPS_PER_EPISODE = TIME_MAX_SIMULATION * FREQUENCY
TRAINING = 1
EVALUATION = 0