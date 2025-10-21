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
		# Water level uses actual distance in cm
		# Range should cover from well above siaga to well below banjir
		max_distance = self.siaga_level + 50  # Extra range above siaga
		min_distance = max(0, self.banjir_level - 50)  # Extra range below banjir
		
		water_level = ctrl.Antecedent(np.arange(min_distance, max_distance + 1, 1), 'water_level')
		
		# Avg rate change in cm/min (negative = water rising, positive = water falling)
		avg_rate_change = ctrl.Antecedent(np.arange(-200, 201, 1), 'avg_rate_change')
		
		# Rainfall in mm/hour (actual values)
		rainfall = ctrl.Antecedent(np.arange(0, 101, 1), 'rainfall')
		
		flood_risk = ctrl.Consequent(np.arange(0, 101, 1), 'flood_risk', defuzzify_method='centroid')
		
		# Define water level membership functions based on actual distances
		range_span = self.siaga_level - self.banjir_level
		mid_point = (self.siaga_level + self.banjir_level) / 2
		quarter_span = range_span / 4
		
		# Normal: above siaga_level
		water_level['normal'] = fuzz.trapmf(water_level.universe, 
			[self.siaga_level, self.siaga_level + 10, max_distance, max_distance])
		
		# Siaga I: upper part of the range between banjir and siaga (overlapping)
		water_level['siaga I'] = fuzz.trimf(water_level.universe, 
			[self.banjir_level + quarter_span, mid_point + quarter_span/2, self.siaga_level + 5])
		
		# Siaga II: middle-lower part of the range (overlapping with Siaga I)
		water_level['siaga II'] = fuzz.trimf(water_level.universe, 
			[self.banjir_level, mid_point, mid_point + quarter_span])
		
		# Banjir: below banjir_level
		water_level['banjir'] = fuzz.trapmf(water_level.universe, 
			[min_distance, min_distance, self.banjir_level, self.banjir_level + 5])
		
		# Rate of change in cm/min (negative = rising water)
		avg_rate_change['turun sangat cepat'] = fuzz.trapmf(avg_rate_change.universe, [100, 120, 200, 200])
		avg_rate_change['turun cepat'] = fuzz.trimf(avg_rate_change.universe, [50, 80, 110])
		avg_rate_change['turun lambat'] = fuzz.trimf(avg_rate_change.universe, [10, 30, 60])
		avg_rate_change['stabil'] = fuzz.trimf(avg_rate_change.universe, [-15, 0, 15])
		avg_rate_change['naik lambat'] = fuzz.trimf(avg_rate_change.universe, [-60, -30, -10])
		avg_rate_change['naik cepat'] = fuzz.trimf(avg_rate_change.universe, [-110, -80, -50])
		avg_rate_change['naik sangat cepat'] = fuzz.trimf(avg_rate_change.universe, [-150, -120, -90])
		avg_rate_change['naik ekstrem'] = fuzz.trapmf(avg_rate_change.universe, [-200, -200, -140, -110])
		
		# Rainfall in mm/hour (actual values)
		rainfall['tidak_hujan'] = fuzz.trapmf(rainfall.universe, [0, 0, 0.5, 1])
		rainfall['ringan'] = fuzz.trimf(rainfall.universe, [0.5, 5, 10])
		rainfall['sedang'] = fuzz.trimf(rainfall.universe, [8, 15, 25])
		rainfall['lebat'] = fuzz.trimf(rainfall.universe, [20, 40, 60])
		rainfall['sangat_lebat'] = fuzz.trapmf(rainfall.universe, [55, 75, 100, 100])
		
		# Flood risk output
		flood_risk['low'] = fuzz.trapmf(flood_risk.universe, [0, 0, 15, 30])
		flood_risk['medium'] = fuzz.trimf(flood_risk.universe, [25, 45, 65])
		flood_risk['high'] = fuzz.trimf(flood_risk.universe, [60, 75, 88])
		flood_risk['critical'] = fuzz.trapmf(flood_risk.universe, [85, 92, 100, 100])
		
		rules = self._define_fuzzy_rules(water_level, avg_rate_change, rainfall, flood_risk)
		return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
	
	def _define_fuzzy_rules(self, water_level, rate_change, rainfall, flood_risk):
		rules = []
		
		# Extreme rate rules
		rules.append(ctrl.Rule(rate_change['naik ekstrem'], flood_risk['critical']))
		rules.append(ctrl.Rule(rate_change['naik sangat cepat'] & rainfall['sangat_lebat'], flood_risk['critical']))
		rules.append(ctrl.Rule(rate_change['naik sangat cepat'] & rainfall['lebat'], flood_risk['critical']))
		
		# Banjir level rules (below banjir_level)
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['naik ekstrem'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['naik sangat cepat'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['naik cepat'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['naik lambat'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['stabil'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['turun lambat'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['turun cepat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['turun sangat cepat'], flood_risk['high']))
		
		# Siaga II rules (lower part of range)
		rules.append(ctrl.Rule(water_level['siaga II'] & rate_change['naik ekstrem'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['siaga II'] & rate_change['naik sangat cepat'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['siaga II'] & rate_change['naik cepat'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['siaga II'] & rate_change['naik lambat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga II'] & rate_change['stabil'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga II'] & rate_change['turun lambat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga II'] & rate_change['turun cepat'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['siaga II'] & rate_change['turun sangat cepat'], flood_risk['medium']))
		
		# Siaga I rules (upper part of range)
		rules.append(ctrl.Rule(water_level['siaga I'] & rate_change['naik ekstrem'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['siaga I'] & rate_change['naik sangat cepat'] & rainfall['sangat_lebat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga I'] & rate_change['naik sangat cepat'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['siaga I'] & rate_change['naik cepat'] & rainfall['lebat'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['siaga I'] & rate_change['naik cepat'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['siaga I'] & rate_change['naik lambat'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['siaga I'] & rate_change['stabil'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['siaga I'] & rate_change['turun lambat'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['siaga I'] & rate_change['turun cepat'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['siaga I'] & rate_change['turun sangat cepat'], flood_risk['low']))
		
		# Normal rules (above siaga_level)
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik ekstrem'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik sangat cepat'] & rainfall['sangat_lebat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik sangat cepat'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik cepat'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik lambat'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['stabil'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['turun lambat'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['turun cepat'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['turun sangat cepat'], flood_risk['low']))
		
		return rules
	
	def add_distance_reading(self, distance):
		self.distance_history.append(distance)
		self.reading_count += 1
	
	def calculate_average_rate_change(self):
		"""Calculate rate of change in cm/min (negative = water rising)"""
		if len(self.distance_history) < 2:
			return 0.0
		
		num_intervals = len(self.distance_history) - 1
		time_span_seconds = num_intervals * self.reading_interval_seconds
		
		if time_span_seconds < 1:
			return 0.0
		
		# Distance change (negative = water level rising)
		distance_change = self.distance_history[-1] - self.distance_history[0]
		time_span_minutes = time_span_seconds / 60.0
		return distance_change / time_span_minutes
	
	def get_status_message(self, warning_level, risk_score, avg_rate_cm_per_min):
		"""Generate status message based on warning level and conditions"""
		
		if warning_level == "BANJIR":
			if risk_score >= 95:
				return "🚨 CRITICAL FLOOD EMERGENCY! Immediate evacuation required!"
			elif avg_rate_cm_per_min < -150:
				return "🚨 EXTREME FLOOD - Water rising extremely fast!"
			elif risk_score >= 85:
				return "⚠️ SEVERE FLOOD WARNING - High water levels detected!"
			else:
				return "⚠️ FLOOD WARNING - Water has reached critical level!"
		
		elif warning_level == "SIAGA":
			if avg_rate_cm_per_min < -120:
				return "⚠️ ALERT - Water rising rapidly!"
			elif risk_score >= 75:
				return "⚠️ HIGH ALERT - Elevated water levels with high risk!"
			elif risk_score >= 50:
				return "⚡ STANDBY ALERT - Water level elevated!"
			else:
				return "⚡ CAUTION - Water level approaching alert threshold!"
		
		else:  # NORMAL
			if risk_score >= 85:
				return "⚡ WARNING - Risk elevated despite normal water level!"
			elif risk_score >= 50:
				return "ℹ️ MONITORING - Elevated risk factors detected!"
			elif avg_rate_cm_per_min < -50:
				return "ℹ️ WATCH - Water level rising steadily!"
			else:
				return "✅ NORMAL - Water level safe!"
	
	def calculate_risk(self, current_distance, current_rainfall_mm_per_hour=0):
		if self.calibration_height is None:
			raise ValueError("System not calibrated")
		
		self.add_distance_reading(current_distance)
		avg_rate_cm_per_min = self.calculate_average_rate_change()
		
		# Rate override bonus for extreme situations
		rate_override_bonus = 0
		if avg_rate_cm_per_min < -150:  # Extreme rise
			rate_override_bonus = 25
		elif avg_rate_cm_per_min < -100:  # Very fast rise
			rate_override_bonus = 12
		
		# Use crisp values directly
		self.fuzzy_system.input['water_level'] = current_distance
		self.fuzzy_system.input['avg_rate_change'] = avg_rate_cm_per_min
		self.fuzzy_system.input['rainfall'] = current_rainfall_mm_per_hour
		
		try:
			self.fuzzy_system.compute()
			risk_score = min(100, self.fuzzy_system.output['flood_risk'] + rate_override_bonus)
		except:
			# Fallback calculation
			distance_risk = 0
			if current_distance <= self.banjir_level:
				distance_risk = 50
			elif current_distance <= self.siaga_level:
				normalized_pos = (self.siaga_level - current_distance) / (self.siaga_level - self.banjir_level)
				distance_risk = 25 + (normalized_pos * 25)
			
			rate_risk = max(0, min(35, -avg_rate_cm_per_min / 5))
			rainfall_risk = min(15, current_rainfall_mm_per_hour * 0.6)
			risk_score = min(100, distance_risk + rate_risk + rainfall_risk + rate_override_bonus)
		
		warning_level = self._determine_warning_level(risk_score, current_distance)
		old_warning_level = self.previous_warning_level
		self.previous_warning_level = warning_level
		
		status_message = self.get_status_message(warning_level, risk_score, avg_rate_cm_per_min)
		
		return {
			'reading_number': self.reading_count,
			'current_distance': current_distance,
			'avg_rate_change_cm_per_min': avg_rate_cm_per_min,
			'current_rainfall_mm_per_hour': current_rainfall_mm_per_hour,
			'risk_score': risk_score,
			'warning_level': warning_level,
			'previous_warning_level': old_warning_level,
			'status_message': status_message
		}
	
	def _determine_warning_level(self, risk_score, current_distance):
		avg_rate = self.calculate_average_rate_change()
		
		# Base level from distance
		if current_distance <= self.banjir_level:
			base_level = "BANJIR"
		elif current_distance <= self.siaga_level:
			base_level = "SIAGA"
		else:
			base_level = "NORMAL"
		
		# Rate-based escalation
		if avg_rate < -150:  # Extreme rate anywhere
			return "BANJIR"
		
		if base_level == "BANJIR":
			if avg_rate > 50 and current_distance > self.banjir_level + 10:
				return "SIAGA"
			return "BANJIR"
		
		if base_level == "SIAGA":
			if avg_rate < -120:  # Very fast at SIAGA level
				return "BANJIR"
			if avg_rate < -100 and risk_score >= 95:
				return "BANJIR"
			if current_distance <= self.banjir_level + 2 and avg_rate < -80:
				return "BANJIR"
			if avg_rate > 50 and risk_score < 35:
				return "NORMAL"
			return "SIAGA"
		
		if base_level == "NORMAL":
			if avg_rate < -140:  # Must be very extreme
				return "BANJIR"
			if avg_rate < -80:  # Fast consistent rise
				return "SIAGA"
			if risk_score >= 85:
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
	
	print("=" * 80)
	for distance, rain in zip(distances, rainfall):
		result = system.calculate_risk(distance, rain)
		print(f"Warning: {result['warning_level']} | Risk: {result['risk_score']:.1f}% | Status: {result['status_message']}")