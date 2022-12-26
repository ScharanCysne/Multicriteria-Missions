import pygame

from constants import *

class Interface(object):
    def __init__(self):
        pygame.init()
        self.resolution = RESOLUTION
        self.size = SCREEN_WIDTH, SCREEN_HEIGHT 
        self.font20 = pygame.font.SysFont(None, 20)
        self.font24 = pygame.font.SysFont(None, 24)
        self.screen = pygame.display.set_mode(self.size)
        self.clock = pygame.time.Clock()

        # Title
        self.title = self.font24.render('Deep RL for UAVs in Coverage Missions', True, BLACK)
        # Flow Chart
        #self.flow = FlowField(resolution)
        # Drones' start srea
        self.start_area = pygame.Surface((50*RATIO, SCREEN_HEIGHT))
        self.start_area.set_alpha(50)
        pygame.draw.rect(self.start_area, BLUE, self.start_area.get_rect(), 1)
        # Drones' end area
        self.end_area = pygame.Surface((50, SCREEN_HEIGHT))
        self.end_area.set_alpha(50)
        pygame.draw.rect(self.end_area, BLUE, self.end_area.get_rect(), 1)
        # Simulation Time
        self.sim_time = self.font24.render(f"Time: 0.00s", True, BLACK)
        self.timesteps = 0


    def draw(self, agents, obstacles, env_state, num_agents, out_time, time_executing, record=False):
        self.update_screen(agents, obstacles, env_state, num_agents, out_time, time_executing, record)


    def update_screen(self, agents=dict(), obstacles=[], state=None, num_agents=0, out_time=[], time_executing=0, record=False):
        self.timesteps += 1
        # Background
        self.screen.fill(LIGHT_GRAY)                             
        # Starting area
        self.screen.blit(self.start_area, (0, 0))                
        # Ending area
        self.screen.blit(self.end_area, (SCREEN_WIDTH - 50, 0))   
        # Flow Chart
        #self.flow.draw(self.screen)                              
        # Drone field of vision
        self.draw_observable_area(agents)       
        # Obstacles
        self.draw_obstacles(obstacles)                           
        # Field vectors
        self.draw_field_vectors(agents)                                
        # Connections
        self.draw_connections(agents, state)          
        # Drones
        self.draw_drones(agents)  
        # Running Time
        self.sim_time = self.font24.render(f"Time: {time_executing:.2f} s", True, BLACK)
        self.screen.blit(self.sim_time, (1700, 20))   
        # Title
        self.screen.blit(self.title, (20, 20))
        # Print time of each iteration
        for idx, t in enumerate(out_time):
            try:
                img = self.font20.render(f'{idx+1} - Scan Time: {t:.2f}', True, BLACK)
            except:
                img = self.font20.render(f'{idx+1} - Scan Time: {t}', True, BLACK)
            self.screen.blit(img, (20, 20*(idx+2)))
        # Flip screen
        pygame.display.flip()
        # Record option
        if record:
            pygame.image.save(self.screen, f"replay/screenshot_{self.timesteps}.jpeg")
        

    def draw_obstacles(self, obstacles):
        for coordinate in obstacles: 
            pygame.draw.circle(self.screen, RED, RATIO * coordinate, radius=SIZE_OBSTACLES, width=20)
            pygame.draw.circle(self.screen, BLACK, RATIO * coordinate, radius=RATIO * AVOID_DISTANCE, width=1)
            #pygame.draw.circle(self.screen, BLACK, coordinate, radius=RADIUS_OBSTACLES*1.6 + AVOID_DISTANCE, width=1)


    def draw_connections(self, agents, state):
        drones = list(agents.values())
        num_agents = len(drones)
        for i in range(num_agents):
            for j in range(i+1, num_agents):
                idx_i = drones[i].id
                idx_j = drones[j].id
                if state.adjacencyMatrix[idx_i][idx_j]:
                    pos_i = RATIO * drones[i].location
                    pos_j = RATIO * drones[j].location
                    pygame.draw.line(self.screen, BLACK, pos_i, pos_j, 1)


    def draw_drones(self, agents):
        for drone in agents.values():
            # Draw drone's position            
            drone.draw(self.screen) 
            # writes drone id
            #img = self.font20.render(f'Drone {drone.id}', True, BLACK)
            #self.screen.blit(img, RATIO * drone.location + (20,0))
            # writes drone current position in column and row
            #p = drone.location
            #col = p.x // RESOLUTION + 1
            #row = p.y // RESOLUTION + 1
            #img = self.font20.render(f'Pos:{col},{row}', True, BLUE)
            #self.screen.blit(img, RATIO * drone.location + (0,35))


    def draw_observable_area(self, agents):
        for drone in agents.values():
            #pos = agents[drone].location
            pos = RATIO * drone.location
            pygame.draw.circle(self.screen, LIGHT_YELLOW, pos, radius=RATIO*OBSERVABLE_RADIUS)


    def draw_field_vectors(self, agents):
        for drone in agents.values():
            pos_i = RATIO * drone.location
            # Obstacles vector
            #pos_j = RATIO * drone.location + RATIO * drone.obstacles 
            #pygame.draw.line(self.screen, RED, pos_i, pos_j, 1)
            # Neighbors vector
            #pos_j = RATIO * drone.location + RATIO * drone.neighbors 
            #pygame.draw.line(self.screen, LIGHT_GRAY, pos_i, pos_j, 1)
