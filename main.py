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
		max_dist = self.siaga_level + 50
		min_dist = max(0, self.banjir_level - 50)
		
		water_level = ctrl.Antecedent(np.arange(min_dist, max_dist + 1, 1), 'water_level')
		avg_rate_change = ctrl.Antecedent(np.arange(-10, 10.01, 0.01), 'avg_rate_change')
		rainfall = ctrl.Antecedent(np.arange(0, 101, 0.1), 'rainfall')
		flood_risk = ctrl.Consequent(np.arange(0, 101, 1), 'flood_risk', defuzzify_method='centroid')
		
		range_span = self.siaga_level - self.banjir_level
		mid = (self.siaga_level + self.banjir_level) / 2
		quarter_span = range_span / 4
		
		water_level['normal'] = fuzz.trapmf(water_level.universe, [self.siaga_level, self.siaga_level + 10, max_dist, max_dist])
		water_level['siaga I'] = fuzz.trimf(water_level.universe, [self.banjir_level + quarter_span, mid + quarter_span/2, self.siaga_level + 5])
		water_level['siaga II'] = fuzz.trimf(water_level.universe, [self.banjir_level, mid, mid + quarter_span])
		water_level['banjir'] = fuzz.trapmf(water_level.universe, [min_dist, min_dist, self.banjir_level, self.banjir_level + 5])
		
		g = 0.67  # guideline
		avg_rate_change['turun sangat cepat'] = fuzz.trapmf(avg_rate_change.universe, [-20, -20, -10*g, -7*g])
		avg_rate_change['turun cepat'] = fuzz.trimf(avg_rate_change.universe, [-8*g, -5*g, -3*g])
		avg_rate_change['turun lambat'] = fuzz.trimf(avg_rate_change.universe, [-3.5*g, -2*g, -g])
		avg_rate_change['stabil'] = fuzz.trimf(avg_rate_change.universe, [-1.5*g, 0, 1.5*g])
		avg_rate_change['naik lambat'] = fuzz.trimf(avg_rate_change.universe, [g, 2*g, 3.5*g])
		avg_rate_change['naik cepat'] = fuzz.trimf(avg_rate_change.universe, [3*g, 5*g, 8*g])
		avg_rate_change['naik sangat cepat'] = fuzz.trimf(avg_rate_change.universe, [6*g, 9*g, 11*g])
		avg_rate_change['naik ekstrem'] = fuzz.trapmf(avg_rate_change.universe, [8*g, 12*g, 20, 20])
		
		rainfall['tidak_hujan'] = fuzz.trapmf(rainfall.universe, [0, 0, 0.5, 1])
		rainfall['ringan'] = fuzz.trimf(rainfall.universe, [0.5, 3, 5])
		rainfall['sedang'] = fuzz.trimf(rainfall.universe, [5, 7.5, 10])
		rainfall['lebat'] = fuzz.trimf(rainfall.universe, [10, 15, 20])
		rainfall['sangat_lebat'] = fuzz.trapmf(rainfall.universe, [20, 30, 100, 100])
		
		flood_risk['low'] = fuzz.trapmf(flood_risk.universe, [0, 0, 15, 30])
		flood_risk['medium'] = fuzz.trimf(flood_risk.universe, [25, 45, 65])
		flood_risk['high'] = fuzz.trimf(flood_risk.universe, [60, 75, 88])
		flood_risk['critical'] = fuzz.trapmf(flood_risk.universe, [85, 92, 100, 100])
		
		rules = self._define_fuzzy_rules(water_level, avg_rate_change, rainfall, flood_risk)
		return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
	
	def _define_fuzzy_rules(self, wl, rc, rf, fr):
		r = []
		r.append(ctrl.Rule(rc['naik ekstrem'], fr['critical']))
		r.append(ctrl.Rule(rc['naik sangat cepat'] & rf['sangat_lebat'], fr['critical']))
		r.append(ctrl.Rule(rc['naik sangat cepat'] & rf['lebat'], fr['critical']))
		
		r.append(ctrl.Rule(wl['banjir'] & rc['naik ekstrem'], fr['critical']))
		r.append(ctrl.Rule(wl['banjir'] & rc['naik sangat cepat'], fr['critical']))
		r.append(ctrl.Rule(wl['banjir'] & rc['naik cepat'], fr['critical']))
		r.append(ctrl.Rule(wl['banjir'] & rc['naik lambat'], fr['critical']))
		r.append(ctrl.Rule(wl['banjir'] & rc['stabil'], fr['critical']))
		r.append(ctrl.Rule(wl['banjir'] & rc['turun lambat'], fr['critical']))
		r.append(ctrl.Rule(wl['banjir'] & rc['turun cepat'], fr['high']))
		r.append(ctrl.Rule(wl['banjir'] & rc['turun sangat cepat'], fr['high']))
		
		r.append(ctrl.Rule(wl['siaga II'] & rc['naik ekstrem'], fr['critical']))
		r.append(ctrl.Rule(wl['siaga II'] & rc['naik sangat cepat'], fr['critical']))
		r.append(ctrl.Rule(wl['siaga II'] & rc['naik cepat'], fr['critical']))
		r.append(ctrl.Rule(wl['siaga II'] & rc['naik lambat'], fr['high']))
		r.append(ctrl.Rule(wl['siaga II'] & rc['stabil'], fr['high']))
		r.append(ctrl.Rule(wl['siaga II'] & rc['turun lambat'], fr['high']))
		r.append(ctrl.Rule(wl['siaga II'] & rc['turun cepat'], fr['medium']))
		r.append(ctrl.Rule(wl['siaga II'] & rc['turun sangat cepat'], fr['medium']))
		
		r.append(ctrl.Rule(wl['siaga I'] & rc['naik ekstrem'], fr['critical']))
		r.append(ctrl.Rule(wl['siaga I'] & rc['naik sangat cepat'] & rf['sangat_lebat'], fr['high']))
		r.append(ctrl.Rule(wl['siaga I'] & rc['naik sangat cepat'], fr['medium']))
		r.append(ctrl.Rule(wl['siaga I'] & rc['naik cepat'] & rf['lebat'], fr['medium']))
		r.append(ctrl.Rule(wl['siaga I'] & rc['naik cepat'], fr['low']))
		r.append(ctrl.Rule(wl['siaga I'] & rc['naik lambat'], fr['low']))
		r.append(ctrl.Rule(wl['siaga I'] & rc['stabil'], fr['low']))
		r.append(ctrl.Rule(wl['siaga I'] & rc['turun lambat'], fr['low']))
		r.append(ctrl.Rule(wl['siaga I'] & rc['turun cepat'], fr['low']))
		r.append(ctrl.Rule(wl['siaga I'] & rc['turun sangat cepat'], fr['low']))
		
		r.append(ctrl.Rule(wl['normal'] & rc['naik ekstrem'], fr['high']))
		r.append(ctrl.Rule(wl['normal'] & rc['naik sangat cepat'] & rf['sangat_lebat'], fr['high']))
		r.append(ctrl.Rule(wl['normal'] & rc['naik sangat cepat'] & rf['lebat'], fr['medium']))
		r.append(ctrl.Rule(wl['normal'] & rc['naik sangat cepat'], fr['low']))
		r.append(ctrl.Rule(wl['normal'] & rc['naik cepat'], fr['low']))
		r.append(ctrl.Rule(wl['normal'] & rc['naik lambat'], fr['low']))
		r.append(ctrl.Rule(wl['normal'] & rc['stabil'], fr['low']))
		r.append(ctrl.Rule(wl['normal'] & rc['turun lambat'], fr['low']))
		r.append(ctrl.Rule(wl['normal'] & rc['turun cepat'], fr['low']))
		r.append(ctrl.Rule(wl['normal'] & rc['turun sangat cepat'], fr['low']))
		return r
	
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
		distance_change = self.distance_history[0] - self.distance_history[-1]
		return distance_change / (time_span_seconds / 60.0)
	
	def get_status_message(self, warning_level, risk_score, avg_rate):
		g = 0.67
		if warning_level == "BANJIR":
			if risk_score >= 95: return "🚨 CRITICAL FLOOD EMERGENCY! Immediate evacuation required!"
			elif avg_rate < -8*g: return "🚨 EXTREME FLOOD - Water rising extremely fast!"
			elif risk_score >= 85: return "⚠️ SEVERE FLOOD WARNING - High water levels detected!"
			else: return "⚠️ FLOOD WARNING - Water has reached critical level!"
		elif warning_level == "SIAGA II":
			if avg_rate < -7*g: return "🔴 SIAGA II - Final preparations! Evacuate immediately if heavy rain!"
			elif risk_score >= 75: return "🔴 SIAGA II - HIGH RISK! Be ready to evacuate!"
			elif risk_score >= 50: return "🔴 SIAGA II - Water level critical, prepare to evacuate!"
			else: return "🔴 SIAGA II - Water approaching flood level!"
		elif warning_level == "SIAGA I":
			if avg_rate < -5*g: return "🟠 SIAGA I - Water rising rapidly, monitor closely!"
			elif risk_score >= 60: return "🟠 SIAGA I - Elevated risk, prepare evacuation supplies!"
			else: return "🟠 SIAGA I - Water level elevated, stay alert!"
		else:
			if risk_score >= 85: return "⚡ WARNING - Risk elevated despite normal water level!"
			elif risk_score >= 50: return "ℹ️ MONITORING - Elevated risk factors detected!"
			elif avg_rate < -2*g: return "ℹ️ WATCH - Water level rising steadily!"
			else: return "✅ NORMAL - Water level safe!"
	
	def calculate_risk(self, current_distance, current_rainfall_mm_per_hour=0):
		if self.calibration_height is None:
			raise ValueError("System not calibrated")
		
		self.add_distance_reading(current_distance)
		avg_rate = self.calculate_average_rate_change()
		g = 0.67
		
		rate_override = 0
		if avg_rate < -12*g: rate_override = 25
		elif avg_rate < -7*g: rate_override = 12
		
		self.fuzzy_system.input['water_level'] = current_distance
		self.fuzzy_system.input['avg_rate_change'] = avg_rate
		self.fuzzy_system.input['rainfall'] = current_rainfall_mm_per_hour
		
		try:
			self.fuzzy_system.compute()
			risk_score = min(100, self.fuzzy_system.output['flood_risk'] + rate_override)
		except:
			dist_risk = 0
			if current_distance <= self.banjir_level:
				dist_risk = 50
			elif current_distance <= self.siaga_level:
				norm_pos = (self.siaga_level - current_distance) / (self.siaga_level - self.banjir_level)
				dist_risk = 25 + (norm_pos * 25)
			rate_risk = max(0, min(35, (-avg_rate / g) * 2))
			rain_risk = min(15, current_rainfall_mm_per_hour * 0.75)
			risk_score = min(100, dist_risk + rate_risk + rain_risk + rate_override)
		
		warning_level = self._determine_warning_level(risk_score, current_distance)
		old_warning = self.previous_warning_level
		self.previous_warning_level = warning_level
		
		return {
			'reading_number': self.reading_count,
			'current_distance': current_distance,
			"rate_change_cm_per_sec": avg_rate / 60.0,
			'avg_rate_change_cm_per_min': avg_rate,
			'current_rainfall_mm_per_hour': current_rainfall_mm_per_hour,
			'risk_score': risk_score,
			'warning_level': warning_level,
			'previous_warning_level': old_warning,
			'status_message': self.get_status_message(warning_level, risk_score, avg_rate)
		}
	
	def _determine_warning_level(self, risk_score, current_distance):
		avg_rate = self.calculate_average_rate_change()
		g = 0.67
		
		if current_distance <= self.banjir_level:
			base = "BANJIR"
		elif current_distance <= self.siaga_level:
			mid = (self.siaga_level + self.banjir_level) / 2
			base = "SIAGA II" if current_distance <= mid else "SIAGA I"
		else:
			base = "NORMAL"
		
		if avg_rate < -12*g: return "BANJIR"
		
		if base == "BANJIR":
			if avg_rate > 3*g and current_distance > self.banjir_level + 10:
				return "SIAGA II"
			return "BANJIR"
		
		if base == "SIAGA II":
			if avg_rate < -9*g: return "BANJIR"
			if avg_rate < -6*g and risk_score >= 95: return "BANJIR"
			if current_distance <= self.banjir_level + 2 and avg_rate < -4*g: return "BANJIR"
			if avg_rate > 3*g and risk_score < 35: return "SIAGA I"
			return "SIAGA II"
		
		if base == "SIAGA I":
			if avg_rate < -9*g: return "BANJIR"
			if avg_rate < -6*g and risk_score >= 90: return "SIAGA II"
			if avg_rate > 3*g and risk_score < 30: return "NORMAL"
			return "SIAGA I"
		
		if base == "NORMAL":
			if avg_rate < -10*g: return "BANJIR"
			if avg_rate < -7*g: return "SIAGA II"
			if avg_rate < -5*g: return "SIAGA I"
			if risk_score >= 85: return "SIAGA I"
			return "NORMAL"
		
		return base
	
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