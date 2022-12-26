import time

from constants     import *
from environment   import CoverageMissionEnv

class Simulation(object):
    def __init__(self, num_agents=NUM_DRONES, num_obstacles=NUM_OBSTACLES, episodes=10):
        # Simulation Parameters 
        self.num_obstacles = num_obstacles
        self.num_agents = num_agents
        self.episodes = episodes
        self.episode = 0
        self.start_watch = 0
        self.stop_watch = 0
        self.time_executing = 0 
        self.out_time = []

        # Environment variables
        self.target_simulation = SCREEN_WIDTH
        self.environment = CoverageMissionEnv(num_obstacles, num_agents)
        
    def run(self):
        if self.start_watch == 0:
            self.start_watch = time.time()
        self.time_executing += SAMPLE_TIME

        self.environment.scan()
        self.environment.update(self.swarm)
        return self.continue_simulation()

    def continue_simulation(self):
        if self.rate_of_completion() >= 0.8 and self.stop_watch == 0 or self.time_executing > TIME_MAX_SIMULATION:
            self.stop_watch = time.time()
            if self.next_simulation():
                print("Time spent:" + str(self.stop_watch - self.start_watch))
                self.reset_simulation()
            else:
                return False
        return True

    def rate_of_completion(self):
        count_completed = 0
        for drone in self.swarm:
            if drone.reached_goal(self.target_simulation):
                count_completed = count_completed + 1 
        return count_completed / self.num_agents

    def reset_swarm(self):
        for drone in self.swarm:
            del drone
        self.create_swarm()

    def reset_simulation(self):
        time = self.stop_watch - self.start_watch
        if self.time_executing > TIME_MAX_SIMULATION:
            time = "Goal not reached"
        self.set_out_time(time)
            
        self.generate_obstacles()
        self.reset_swarm()
        
        self.time_executing = 0 
        self.start_watch = 0
        self.stop_watch = 0
    
    def get_status(self):
        return self.swarm, self.obstacles, self.environment, self.num_agents, self.out_time

    def set_out_time(self, out_time):
        self.out_time.append(out_time)

    def next_simulation(self):
        if self.episodes == self.episode:
            return False
        else:
            self.episode += 1
            self.print_simulation()
            return True

    def print_simulation(self):
        print(f'{self.episode} - num_agents: {self.num_agents}')