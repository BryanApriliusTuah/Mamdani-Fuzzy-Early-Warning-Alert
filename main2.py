import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from collections import deque

class DynamicFuzzyFloodWarningSystem:
	def __init__(self, reading_interval_seconds=1):
		self.calibration_height = None
		self.siaga_level = None
		self.banjir_level = None
		self.fuzzy_system = None
		self.previous_warning_level = None
		self.reading_interval_seconds = reading_interval_seconds
		self.distance_history = deque(maxlen=60)
		self.reading_count = 0
	
	def calibrate(self, ground_distance, siaga_level_override=None, banjir_level_override=None):
		self.calibration_height = ground_distance
		self.banjir_level = banjir_level_override if banjir_level_override else ground_distance
		self.siaga_level = siaga_level_override if siaga_level_override else ground_distance + 30
		
		if self.siaga_level <= self.banjir_level:
			raise ValueError("siaga_level must be greater than banjir_level")
		
		self.fuzzy_system = self._create_fuzzy_system()
		print(f"Calibrated: Ground={ground_distance}cm, Siaga={self.siaga_level}cm, Banjir={self.banjir_level}cm")
	
	def _create_fuzzy_system(self):
		water_level_norm = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'water_level_norm')
		avg_rate_change = ctrl.Antecedent(np.arange(-1, 1.01, 0.01), 'avg_rate_change')
		rainfall_norm = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'rainfall_norm')
		flood_risk = ctrl.Consequent(np.arange(0, 101, 1), 'flood_risk', defuzzify_method='centroid')
		
		water_level_norm['normal'] = fuzz.trapmf(water_level_norm.universe, [0, 0, 0.1, 0.25])
		water_level_norm['siaga'] = fuzz.trimf(water_level_norm.universe, [0.15, 0.4, 0.65])
		water_level_norm['siaga II'] = fuzz.trimf(water_level_norm.universe, [0.5, 0.7, 0.85])
		water_level_norm['banjir'] = fuzz.trapmf(water_level_norm.universe, [0.75, 0.9, 1.0, 1.0])
		
		avg_rate_change['turun sangat cepat'] = fuzz.trapmf(avg_rate_change.universe, [-1, -1, -0.6, -0.4])
		avg_rate_change['turun cepat'] = fuzz.trimf(avg_rate_change.universe, [-0.5, -0.3, -0.15])
		avg_rate_change['turun lambat'] = fuzz.trimf(avg_rate_change.universe, [-0.2, -0.1, -0.03])
		avg_rate_change['stabil'] = fuzz.trimf(avg_rate_change.universe, [-0.05, 0, 0.05])
		avg_rate_change['naik lambat'] = fuzz.trimf(avg_rate_change.universe, [0.03, 0.1, 0.2])
		avg_rate_change['naik cepat'] = fuzz.trimf(avg_rate_change.universe, [0.15, 0.3, 0.5])
		avg_rate_change['naik sangat cepat'] = fuzz.trimf(avg_rate_change.universe, [0.4, 0.65, 0.85])
		avg_rate_change['naik ekstrem'] = fuzz.trapmf(avg_rate_change.universe, [0.75, 0.9, 1, 1])
		
		rainfall_norm['tidak_hujan'] = fuzz.trapmf(rainfall_norm.universe, [0, 0, 0.02, 0.04])
		rainfall_norm['ringan'] = fuzz.trimf(rainfall_norm.universe, [0.02, 0.1, 0.2])
		rainfall_norm['sedang'] = fuzz.trimf(rainfall_norm.universe, [0.15, 0.3, 0.45])
		rainfall_norm['lebat'] = fuzz.trimf(rainfall_norm.universe, [0.35, 0.65, 0.8])
		rainfall_norm['sangat_lebat'] = fuzz.trapmf(rainfall_norm.universe, [0.75, 0.9, 1, 1])
		
		flood_risk['low'] = fuzz.trapmf(flood_risk.universe, [0, 0, 15, 30])
		flood_risk['medium'] = fuzz.trimf(flood_risk.universe, [25, 45, 65])
		flood_risk['high'] = fuzz.trimf(flood_risk.universe, [60, 75, 88])
		flood_risk['critical'] = fuzz.trapmf(flood_risk.universe, [85, 92, 100, 100])
		
		rules = self._define_fuzzy_rules(water_level_norm, avg_rate_change, rainfall_norm, flood_risk)
		return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
	
	def _define_fuzzy_rules(self, water_level, rate_change, rainfall_norm, flood_risk):
		rules = []
		rules.append(ctrl.Rule(rate_change['naik ekstrem'], flood_risk['critical']))
		rules.append(ctrl.Rule(rate_change['naik sangat cepat'] & rainfall_norm['sangat_lebat'], flood_risk['critical']))
		rules.append(ctrl.Rule(rate_change['naik sangat cepat'] & rainfall_norm['lebat'], flood_risk['critical']))
		
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['naik ekstrem'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['naik sangat cepat'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['naik cepat'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['naik lambat'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['stabil'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['turun lambat'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['turun cepat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['turun sangat cepat'], flood_risk['high']))
		
		rules.append(ctrl.Rule(water_level['siaga II'] & rate_change['naik ekstrem'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['siaga II'] & rate_change['naik sangat cepat'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['siaga II'] & rate_change['naik cepat'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['siaga II'] & rate_change['naik lambat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga II'] & rate_change['stabil'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga II'] & rate_change['turun lambat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga II'] & rate_change['turun cepat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga II'] & rate_change['turun sangat cepat'], flood_risk['medium']))
		
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik ekstrem'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik sangat cepat'] & rainfall_norm['sangat_lebat'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik sangat cepat'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik cepat'] & rainfall_norm['lebat'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik cepat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik lambat'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['stabil'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['turun lambat'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['turun cepat'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['turun sangat cepat'], flood_risk['low']))
		
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik ekstrem'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik sangat cepat'] & rainfall_norm['sangat_lebat'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik sangat cepat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik cepat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik lambat'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['stabil'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['turun lambat'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['turun cepat'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['turun sangat cepat'], flood_risk['low']))
		
		return rules
	
	def normalize_water_level(self, current_distance):
		if current_distance >= self.siaga_level:
			return 0.0
		elif current_distance <= self.banjir_level:
			return 1.0
		else:
			total_range = self.siaga_level - self.banjir_level
			return 1.0 - ((current_distance - self.banjir_level) / total_range)
	
	def add_distance_reading(self, distance):
		self.distance_history.append(distance)
		self.reading_count += 1
	
	def calculate_average_rate_change(self):
		if len(self.distance_history) < 2:
			return 0.0
		
		num_intervals = len(self.distance_history) - 1
		time_span_seconds = num_intervals * self.reading_interval_seconds
		
		if time_span_seconds < 1:
			return 0.0
		
		distance_change = self.distance_history[-1] - self.distance_history[0]
		time_span_minutes = time_span_seconds / 60.0
		return distance_change / time_span_minutes
	
	def normalize_avg_rate_change(self, avg_rate_cm_per_min, max_rate=23.0):
		inverted_rate = -avg_rate_cm_per_min
		clipped_rate = np.clip(inverted_rate, -max_rate, max_rate)
		return clipped_rate / max_rate
	
	def normalize_rainfall(self, rainfall_mm_per_hour, max_rainfall=25):
		return np.clip(rainfall_mm_per_hour / max_rainfall, 0, 1)
	
	def calculate_risk(self, current_distance, current_rainfall_mm_per_hour=0):
		if self.calibration_height is None:
			raise ValueError("System not calibrated")
		
		self.add_distance_reading(current_distance)
		avg_rate_cm_per_min = self.calculate_average_rate_change()
		
		water_level_normalized = self.normalize_water_level(current_distance)
		avg_rate_normalized = self.normalize_avg_rate_change(avg_rate_cm_per_min)
		rainfall_normalized = self.normalize_rainfall(current_rainfall_mm_per_hour)
		
		rate_override_bonus = 0
		if avg_rate_cm_per_min < -20:
			rate_override_bonus = 35
		elif avg_rate_cm_per_min < -13:
			rate_override_bonus = 22
		elif avg_rate_cm_per_min < -8:
			rate_override_bonus = 12
		elif avg_rate_cm_per_min < -5:
			rate_override_bonus = 5
		
		self.fuzzy_system.input['water_level_norm'] = water_level_normalized
		self.fuzzy_system.input['avg_rate_change'] = avg_rate_normalized
		self.fuzzy_system.input['rainfall_norm'] = rainfall_normalized
		
		try:
			self.fuzzy_system.compute()
			risk_score = min(100, self.fuzzy_system.output['flood_risk'] + rate_override_bonus)
		except:
			rate_risk = max(0, -avg_rate_normalized) * 35
			risk_score = min(100, water_level_normalized * 50 + rate_risk + rainfall_normalized * 15 + rate_override_bonus)
		
		warning_level = self._determine_warning_level(risk_score, current_distance)
		old_warning_level = self.previous_warning_level
		self.previous_warning_level = warning_level
		
		return {
			'reading_number': self.reading_count,
			'current_distance': current_distance,
			'avg_rate_change_cm_per_min': avg_rate_cm_per_min,
			'water_level_normalized': water_level_normalized,
			'avg_rate_normalized': avg_rate_normalized,
			'rainfall_normalized': rainfall_normalized,
			'risk_score': risk_score,
			'warning_level': warning_level,
			'previous_warning_level': old_warning_level
		}
	
	def _determine_warning_level(self, risk_score, current_distance):
		avg_rate = self.calculate_average_rate_change()
		
		if current_distance <= self.banjir_level + 5:
			base_level = "BANJIR"
		elif current_distance <= self.siaga_level:
			base_level = "SIAGA"
		else:
			base_level = "NORMAL"
		
		if avg_rate < -20:
			return "BANJIR"
		
		if base_level == "BANJIR":
			if current_distance <= self.banjir_level + 5:
				return "BANJIR"
			if avg_rate > 12 and current_distance > self.banjir_level + 12:
				return "SIAGA"
			return "BANJIR"
		
		if base_level == "SIAGA":
			if avg_rate < -15:
				return "BANJIR"
			if avg_rate < -10 and risk_score >= 90:
				return "BANJIR"
			if current_distance <= self.banjir_level + 3 and avg_rate < -5:
				return "BANJIR"
			if avg_rate > 10 and risk_score < 35:
				return "NORMAL"
			return "SIAGA"
		
		if base_level == "NORMAL":
			if avg_rate < -18:
				return "BANJIR"
			if avg_rate < -8:
				return "SIAGA"
			if risk_score >= 65:
				return "SIAGA"
			return "NORMAL"
		
		return base_level
	
	def reset_history(self):
		self.distance_history.clear()
		self.reading_count = 0
		self.previous_warning_level = None


if __name__ == "__main__":
	system = DynamicFuzzyFloodWarningSystem(reading_interval_seconds=1)
	system.calibrate(ground_distance=100, siaga_level_override=130, banjir_level_override=100)
	
	distances = [150, 149.92, 149.75, 149.5, 149.2, 148.8, 148.3, 147.7, 147.0, 146.2]
	rainfall = [0, 0, 0, 5, 5, 10, 15, 20, 20, 15]
	
	for distance, rain in zip(distances, rainfall):
		result = system.calculate_risk(distance, rain)
		print(f"Warning: {result['warning_level']} | Risk: {result['risk_score']:.1f}%")