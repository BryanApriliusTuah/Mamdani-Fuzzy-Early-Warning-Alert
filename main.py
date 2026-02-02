import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from collections import deque
from typing import Dict, Optional


class DynamicFuzzyFloodWarningSystem:

	RATE_FACTOR = 0.67

	def __init__(self, reading_interval_seconds: int = 1):
		self.calibration_height = None
		self.siaga_level = None
		self.banjir_level = None
		self.fuzzy_system = None
		self.previous_warning_level = None
		self.reading_interval_seconds = reading_interval_seconds
		self.distance_history = deque(maxlen=60)
		self.reading_count = 0
		self.rainfall_history_24h: list = []  # Store 24 hours of rainfall data from database

	def get_effective_rainfall(self, current_rainfall: float, rainfall_hourly_data: list = None) -> Dict[str, any]:
		"""
		Determine effective rainfall for risk calculation.

		Logic:
		- If rainfall_hourly_data has >= 1 record (>= 1 hour of data), use moving average
		- Otherwise, use current real-time rainfall

		Args:
			current_rainfall: Real-time rainfall reading (mm/hour)
			rainfall_hourly_data: List of hourly rainfall records from last 24 hours
			                     Each item should have 'rainfall_mm' value

		Returns:
			Dict with effective_rainfall, method_used, and moving_average_hours
		"""
		if rainfall_hourly_data is not None:
			self.rainfall_history_24h = rainfall_hourly_data

		# Check if we have at least 1 hour of historical data
		if len(self.rainfall_history_24h) >= 1:
			# Calculate moving average from historical data
			total_rainfall = sum(r if isinstance(r, (int, float)) else r.get('rainfall_mm', 0)
			                     for r in self.rainfall_history_24h)
			hours_count = len(self.rainfall_history_24h)
			moving_avg = total_rainfall / hours_count

			return {
				'effective_rainfall': moving_avg,
				'method': 'moving_average_24h',
				'hours_in_average': hours_count,
				'current_rainfall': current_rainfall,
				'moving_average': moving_avg
			}
		else:
			# Use real-time rainfall (less than 1 hour of data)
			return {
				'effective_rainfall': current_rainfall,
				'method': 'realtime',
				'hours_in_average': 0,
				'current_rainfall': current_rainfall,
				'moving_average': None
			}
	
	def calibrate(self, ground_distance: float, 
				  siaga_level_override: Optional[float] = None, 
				  banjir_level_override: Optional[float] = None) -> None:
		if ground_distance <= 0:
			raise ValueError("Ground distance must be positive")
		
		self.calibration_height = ground_distance
		self.banjir_level = banjir_level_override if banjir_level_override is not None else ground_distance
		self.siaga_level = siaga_level_override if siaga_level_override is not None else ground_distance + 30
		
		if self.siaga_level <= self.banjir_level:
			raise ValueError("siaga_level must be greater than banjir_level")
		
		self.fuzzy_system = self._create_fuzzy_system()
		self._reset_system()
	
	def _create_fuzzy_system(self) -> ctrl.ControlSystemSimulation:
		jarak_maksimum = self.siaga_level
		jarak_minimum = self.banjir_level
		
		water_level = ctrl.Antecedent(np.arange(jarak_minimum, jarak_maksimum + 1, 1), 'water_level')
		avg_rate_change = ctrl.Antecedent(np.arange(-2.36, 2.36, 0.01), 'avg_rate_change')
		rainfall = ctrl.Antecedent(np.arange(0, 21.1, 0.1), 'rainfall')
		flood_risk = ctrl.Consequent(np.arange(0, 101, 1), 'flood_risk', defuzzify_method='centroid')
		
		safety_margin = self.siaga_level - self.banjir_level
		setengah_safety_margin = safety_margin / 2
		
		water_level['normal'] = fuzz.trapmf(water_level.universe, [self.siaga_level, self.siaga_level, jarak_maksimum, jarak_maksimum])
		water_level['siaga I'] = fuzz.trimf(water_level.universe, [self.banjir_level, self.banjir_level + setengah_safety_margin, self.siaga_level])
		water_level['siaga II'] = fuzz.trimf(water_level.universe, [self.banjir_level, self.banjir_level + setengah_safety_margin, self.siaga_level])
		water_level['banjir'] = fuzz.trapmf(water_level.universe, [jarak_minimum, jarak_minimum, self.banjir_level, self.banjir_level])
	
		avg_rate_change['turun'] = fuzz.trapmf(avg_rate_change.universe, [-2.36, -2.36, -1.75, -0.3])
		avg_rate_change['stabil'] = fuzz.trimf(avg_rate_change.universe, [-0.67, 0, 0.67])
		avg_rate_change['naik'] = fuzz.trapmf(avg_rate_change.universe, [0.3, 1.75, 2.36, 2.36])		
		
		rainfall['tidak_hujan'] = fuzz.trapmf(rainfall.universe, [0, 0, 0.5, 1])
		rainfall['ringan'] = fuzz.trimf(rainfall.universe, [1, 3, 5])
		rainfall['sedang'] = fuzz.trimf(rainfall.universe, [5, 7.5, 10])
		rainfall['lebat'] = fuzz.trimf(rainfall.universe, [10, 15, 20])
		rainfall['sangat_lebat'] = fuzz.trapmf(rainfall.universe, [20, 20.5, 21, 21])
		
		flood_risk['very low'] = fuzz.trapmf(flood_risk.universe, [0, 0, 20, 25])
		flood_risk['low'] = fuzz.trimf(flood_risk.universe, [20, 35, 45])
		flood_risk['moderate'] = fuzz.trimf(flood_risk.universe, [40, 55, 65])
		flood_risk['high'] = fuzz.trimf(flood_risk.universe, [60, 75, 85])
		flood_risk['very high'] = fuzz.trapmf(flood_risk.universe, [80, 90, 100, 100])
		
		rules = self._define_fuzzy_rules(water_level, avg_rate_change, rainfall, flood_risk)
		return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
	
	def _define_fuzzy_rules(self, wl: ctrl.Antecedent, rc: ctrl.Antecedent, 
						   rf: ctrl.Antecedent, fr: ctrl.Consequent) -> list:
		rules = []
		
		rules.extend([
			ctrl.Rule(wl['banjir'] & rc['naik'] & rf['sangat_lebat'], fr['very high']),
			ctrl.Rule(wl['banjir'] & rc['naik'] & rf['lebat'], fr['very high']),
			ctrl.Rule(wl['banjir'] & rc['naik'], fr['very high']),
			ctrl.Rule(wl['banjir'] & rc['stabil'] & rf['sangat_lebat'], fr['very high']),
			ctrl.Rule(wl['banjir'] & rc['stabil'], fr['very high']),
			ctrl.Rule(wl['banjir'] & rc['turun'] & rf['lebat'], fr['very high']),
			ctrl.Rule(wl['banjir'] & rc['turun'], fr['high']),
		])
		
		rules.extend([
			ctrl.Rule(wl['siaga II'] & rc['naik'] & rf['sangat_lebat'], fr['very high']),
			ctrl.Rule(wl['siaga II'] & rc['naik'] & rf['lebat'], fr['very high']),
			ctrl.Rule(wl['siaga II'] & rc['naik'], fr['high']),
			ctrl.Rule(wl['siaga II'] & rc['stabil'] & rf['lebat'], fr['high']),
			ctrl.Rule(wl['siaga II'] & rc['stabil'], fr['high']),
			ctrl.Rule(wl['siaga II'] & rc['turun'] & rf['lebat'], fr['high']),
			ctrl.Rule(wl['siaga II'] & rc['turun'], fr['moderate']),
		])
		
		rules.extend([
			ctrl.Rule(wl['siaga I'] & rc['naik'] & rf['sangat_lebat'], fr['very high']),
			ctrl.Rule(wl['siaga I'] & rc['naik'] & rf['lebat'], fr['high']),
			ctrl.Rule(wl['siaga I'] & rc['naik'], fr['moderate']),
			ctrl.Rule(wl['siaga I'] & rc['stabil'] & rf['lebat'], fr['moderate']),
			ctrl.Rule(wl['siaga I'] & rc['stabil'], fr['low']),
			ctrl.Rule(wl['siaga I'] & rc['turun'] & rf['sangat_lebat'], fr['moderate']),
			ctrl.Rule(wl['siaga I'] & rc['turun'] & rf['lebat'], fr['low']),
			ctrl.Rule(wl['siaga I'] & rc['turun'], fr['low']),
		])
		
		rules.extend([
			ctrl.Rule(wl['normal'] & rc['naik'] & rf['sangat_lebat'], fr['high']),
			ctrl.Rule(wl['normal'] & rc['naik'] & rf['lebat'], fr['moderate']),
			ctrl.Rule(wl['normal'] & rc['naik'], fr['low']),
			ctrl.Rule(wl['normal'] & rc['stabil'] & rf['sangat_lebat'], fr['low']),
			ctrl.Rule(wl['normal'] & rc['stabil'] & rf['lebat'], fr['low']),
			ctrl.Rule(wl['normal'] & rc['stabil'], fr['very low']),
			ctrl.Rule(wl['normal'] & rc['turun'], fr['very low']),
		])
		
		return rules
	
	def calculate_average_rate_change(self) -> float:
		if len(self.distance_history) < 2:
			return 0.0
		
		num_intervals = len(self.distance_history) - 1
		time_span_seconds = num_intervals * self.reading_interval_seconds
		
		if time_span_seconds < 1:
			return 0.0
		
		distance_change = self.distance_history[0] - self.distance_history[-1]
		return distance_change / (time_span_seconds / 60.0)
	
	def get_status_message(self, flood_risk_category: str, risk_score: float, avg_rate: float, current_distance: float) -> str:		
		if flood_risk_category == "very high":
			return "🚨 CRITICAL FLOOD EMERGENCY! Immediate evacuation required!"
		
		elif flood_risk_category == "high":
			return "🔴 HIGH RISK - Significant flood danger, stay alert!"
		
		elif flood_risk_category == "moderate":
			return "🟠 MODERATE RISK - Stay vigilant and prepared!"
		
		elif flood_risk_category == "low":
				return "🟡 LOW RISK - Situation stable, routine monitoring!"
		
		else:
			return "✅ VERY LOW RISK - All clear, water level safe!"
	
	def calculate_risk(self, current_distance: float,
					   current_rainfall_mm_per_hour: float = 0,
					   rainfall_hourly_data: list = None) -> Dict[str, any]:
		"""
		Calculate flood risk using fuzzy logic.

		Args:
			current_distance: Current sensor distance reading (cm)
			current_rainfall_mm_per_hour: Real-time rainfall intensity (mm/hour)
			rainfall_hourly_data: Optional list of hourly rainfall from last 24 hours.
			                     If provided and has >= 1 hour of data, moving average is used.
			                     Otherwise, uses current_rainfall_mm_per_hour.

		Returns:
			Dict with risk assessment results including rainfall_info
		"""
		if self.calibration_height is None:
			raise ValueError("System not calibrated")
		if current_distance < 0:
			raise ValueError("Distance cannot be negative")
		if current_rainfall_mm_per_hour < 0:
			raise ValueError("Rainfall cannot be negative")

		self.add_distance_reading(current_distance)
		avg_rate = self.calculate_average_rate_change()

		# Get effective rainfall (moving average or real-time)
		rainfall_info = self.get_effective_rainfall(current_rainfall_mm_per_hour, rainfall_hourly_data)
		effective_rainfall = rainfall_info['effective_rainfall']

		self.fuzzy_system.input['water_level'] = current_distance
		self.fuzzy_system.input['avg_rate_change'] = avg_rate
		self.fuzzy_system.input['rainfall'] = effective_rainfall

		try:
			self.fuzzy_system.compute()
			risk_score = self.fuzzy_system.output['flood_risk']
		except:
			risk_score = self._calculate_fallback_risk(
				current_distance, avg_rate, effective_rainfall
			)

		warning_level = self._determine_warning_level(risk_score, current_distance, effective_rainfall)
		old_warning = self.previous_warning_level
		self.previous_warning_level = warning_level
		flood_risk_category = self.get_flood_risk_categories(risk_score)['dominant_category']

		return {
			'reading_number': self.reading_count,
			'current_distance': current_distance,
			'rate_change_cm_per_sec': avg_rate / 60.0,
			'avg_rate_change_cm_per_min': avg_rate,
			'current_rainfall_mm_per_hour': current_rainfall_mm_per_hour,
			'effective_rainfall_mm_per_hour': effective_rainfall,
			'rainfall_info': rainfall_info,
			'risk_score': round(risk_score, 2),
			'risk_category': flood_risk_category,
			'warning_level': warning_level,
			'previous_warning_level': old_warning,
			'status_message': self.get_status_message(flood_risk_category, risk_score, avg_rate, current_distance)
		}
		
	def _reset_system(self) -> None:
		self.distance_history.clear()
		self.reading_count = 0
		self.previous_warning_level = None
	
	def reset_history(self) -> None:
		self._reset_system()
	
	def get_flood_risk_categories(self, risk_score: float) -> Dict[str, any]:
		if self.fuzzy_system is None:
			raise ValueError("System not calibrated")
		
		if not 0 <= risk_score <= 100:
			raise ValueError("Risk score must be between 0 and 100")
		
		flood_risk_consequent = None
		for consequent in self.fuzzy_system.ctrl.consequents:
			if consequent.label == 'flood_risk':
				flood_risk_consequent = consequent
				break
		
		if flood_risk_consequent is None:
			raise ValueError("Flood risk consequent not found in fuzzy system")
		
		categories = {}
		for term_name in flood_risk_consequent.terms:
			membership_degree = fuzz.interp_membership(
				flood_risk_consequent.universe,
				flood_risk_consequent[term_name].mf,
				risk_score
			)
			categories[term_name] = round(membership_degree, 4)
		
		dominant_category = max(categories.items(), key=lambda x: x[1])
		
		return {
			'risk_score': risk_score,
			'categories': categories,
			'dominant_category': dominant_category[0],
			'dominant_membership': dominant_category[1]
		}
	
	def get_all_fuzzy_categories(self) -> Dict[str, list]:
		if self.fuzzy_system is None:
			raise ValueError("System not calibrated")
		
		result = {
			'water_level': [],
			'avg_rate_change': [],
			'rainfall': [],
			'flood_risk': []
		}
		
		for antecedent in self.fuzzy_system.ctrl.antecedents:
			if antecedent.label in result:
				result[antecedent.label] = list(antecedent.terms.keys())
		
		for consequent in self.fuzzy_system.ctrl.consequents:
			if consequent.label in result:
				result[consequent.label] = list(consequent.terms.keys())
		
		return result

	def get_system_info(self) -> Dict[str, any]:
		return {
			'calibrated': self.calibration_height is not None,
			'calibration_height': self.calibration_height,
			'siaga_level': self.siaga_level,
			'banjir_level': self.banjir_level,
			'reading_interval_seconds': self.reading_interval_seconds,
			'total_readings': self.reading_count,
			'history_size': len(self.distance_history),
			'current_warning_level': self.previous_warning_level
		}


if __name__ == "__main__":
	system = DynamicFuzzyFloodWarningSystem(reading_interval_seconds=1)
	system.calibrate(ground_distance=100, siaga_level_override=130, banjir_level_override=100)
	
	distances = [150, 149.92, 149.75, 149.5, 149.2, 148.8, 148.3, 147.7, 147.0, 146.2]
	rainfall = [0, 0, 0, 5, 5, 10, 15, 20, 20, 15]
	
	print("=" * 80)
	for distance, rain in zip(distances, rainfall):
		result = system.calculate_risk(distance, rain)
		print(f"Warning: {result['warning_level']} | Risk: {result['risk_score']:.1f}% | Status: {result['status_message']}")