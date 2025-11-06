# fuzzy_logic.py
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from config import FUZZY_OBSTACLE_RANGE, FUZZY_SPEED_RANGE, FUZZY_TURN_RANGE

class FuzzyController:
    def __init__(self):
        # Define fuzzy variables
        self.obstacle_dist = ctrl.Antecedent(np.arange(*FUZZY_OBSTACLE_RANGE, 0.1), 'obstacle_dist')
        self.speed = ctrl.Consequent(np.arange(*FUZZY_SPEED_RANGE, 0.01), 'speed')
        self.turn = ctrl.Consequent(np.arange(*FUZZY_TURN_RANGE, 0.1), 'turn')

        # Membership functions (tuned for less aggressive avoidance)
        self.obstacle_dist['near'] = fuzz.trimf(self.obstacle_dist.universe, [0, 0, 0.7])
        self.obstacle_dist['medium'] = fuzz.trimf(self.obstacle_dist.universe, [0.5, 1.0, 1.5])
        self.obstacle_dist['far'] = fuzz.trimf(self.obstacle_dist.universe, [1.2, 2.0, 2.0])

        self.speed['slow'] = fuzz.trimf(self.speed.universe, [0, 0, 0.2])
        self.speed['medium'] = fuzz.trimf(self.speed.universe, [0.1, 0.3, 0.5])
        self.speed['fast'] = fuzz.trimf(self.speed.universe, [0.3, 0.5, 0.5])

        self.turn['left'] = fuzz.trimf(self.turn.universe, [-2.0, -1.0, 0])
        self.turn['straight'] = fuzz.trimf(self.turn.universe, [-0.5, 0, 0.5])
        self.turn['right'] = fuzz.trimf(self.turn.universe, [0, 1.0, 2.0])

        # Rules (tuned to allow closer approaches)
        rule1 = ctrl.Rule(self.obstacle_dist['near'], (self.speed['slow'], self.turn['left']))
        rule2 = ctrl.Rule(self.obstacle_dist['medium'], (self.speed['medium'], self.turn['straight']))
        rule3 = ctrl.Rule(self.obstacle_dist['far'], (self.speed['fast'], self.turn['straight']))

        # Control system
        self.system = ctrl.ControlSystem([rule1, rule2, rule3])
        self.sim = ctrl.ControlSystemSimulation(self.system)

    def compute(self, obstacle_dist):
        """Compute fuzzy logic outputs for speed and turn rate."""
        self.sim.input['obstacle_dist'] = obstacle_dist
        self.sim.compute()
        return self.sim.output['speed'], self.sim.output['turn']
