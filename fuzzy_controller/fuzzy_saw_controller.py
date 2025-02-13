import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

wood_hardness = ctrl.Antecedent(np.arange(0, 101, 1), 'wood_hardness')  # Skala 0-100
cutting_time = ctrl.Antecedent(np.arange(0, 61, 1), 'cutting_time')      # Czas w minutach 0-60
blade_speed = ctrl.Consequent(np.arange(0, 5001, 1), 'blade_speed')      # Obroty 0-5000 RPM

wood_hardness['soft'] = fuzz.trimf(wood_hardness.universe, [0, 0, 40])
wood_hardness['medium'] = fuzz.trimf(wood_hardness.universe, [30, 50, 70])
wood_hardness['hard'] = fuzz.trimf(wood_hardness.universe, [60, 100, 100])

cutting_time['short'] = fuzz.trimf(cutting_time.universe, [0, 0, 20])
cutting_time['medium'] = fuzz.trimf(cutting_time.universe, [15, 30, 45])
cutting_time['long'] = fuzz.trimf(cutting_time.universe, [40, 60, 60])

blade_speed['very_slow'] = fuzz.trimf(blade_speed.universe, [0, 0, 1500])
blade_speed['slow'] = fuzz.trimf(blade_speed.universe, [1000, 2000, 3000])
blade_speed['medium'] = fuzz.trimf(blade_speed.universe, [2500, 3500, 4000])
blade_speed['fast'] = fuzz.trimf(blade_speed.universe, [3500, 5000, 5000])

rule1 = ctrl.Rule(wood_hardness['soft'] & cutting_time['long'], blade_speed['slow'])
rule2 = ctrl.Rule(wood_hardness['soft'] & cutting_time['medium'], blade_speed['medium'])
rule3 = ctrl.Rule(wood_hardness['soft'] & cutting_time['short'], blade_speed['fast'])
rule4 = ctrl.Rule(wood_hardness['medium'] & cutting_time['long'], blade_speed['medium'])
rule5 = ctrl.Rule(wood_hardness['medium'] & cutting_time['medium'], blade_speed['medium'])
rule6 = ctrl.Rule(wood_hardness['medium'] & cutting_time['short'], blade_speed['fast'])
rule7 = ctrl.Rule(wood_hardness['hard'] & cutting_time['long'], blade_speed['slow'])
rule8 = ctrl.Rule(wood_hardness['hard'] & cutting_time['medium'], blade_speed['medium'])
rule9 = ctrl.Rule(wood_hardness['hard'] & cutting_time['short'], blade_speed['fast'])

blade_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
blade_simulator = ctrl.ControlSystemSimulation(blade_ctrl)

blade_simulator.input['wood_hardness'] = 65  # Twarde drewno
blade_simulator.input['cutting_time'] = 25   # Średni czas

blade_simulator.compute()

print(f"Zalecana prędkość obrotowa tarczy: {blade_simulator.output['blade_speed']:.0f} RPM")

wood_hardness.view()
cutting_time.view()
blade_speed.view()
plt.show() 