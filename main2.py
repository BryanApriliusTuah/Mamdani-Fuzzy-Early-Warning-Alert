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
		max_dist = self.siaga_level + 50
		min_dist = max(0, self.banjir_level - 50)
		
		water_level = ctrl.Antecedent(np.arange(min_dist, max_dist + 1, 1), 'water_level')
		avg_rate_change = ctrl.Antecedent(np.arange(-2.36, 2.36, 0.01), 'avg_rate_change')
		rainfall = ctrl.Antecedent(np.arange(0, 21.1, 0.1), 'rainfall')
		flood_risk = ctrl.Consequent(np.arange(0, 101, 1), 'flood_risk', defuzzify_method='centroid')
		
		range_span = self.siaga_level - self.banjir_level
		mid = (self.siaga_level + self.banjir_level) / 2
		quarter_span = range_span / 4
		
		water_level['normal'] = fuzz.trapmf(water_level.universe, [self.siaga_level, self.siaga_level + 10, max_dist, max_dist])
		water_level['siaga I'] = fuzz.trimf(water_level.universe, [self.banjir_level + quarter_span, mid + quarter_span/2, self.siaga_level + 5])
		water_level['siaga II'] = fuzz.trimf(water_level.universe, [self.banjir_level, mid, mid + quarter_span])
		water_level['banjir'] = fuzz.trapmf(water_level.universe, [min_dist, min_dist, self.banjir_level, self.banjir_level + 5])
	
		avg_rate_change['turun'] = fuzz.trapmf(avg_rate_change.universe, [-2.36, -2.36, -1.75, -0.3])
		avg_rate_change['stabil'] = fuzz.trimf(avg_rate_change.universe, [-0.67, 0, 0.67])
		avg_rate_change['naik'] = fuzz.trapmf(avg_rate_change.universe, [0.3, 1.75, 2.36, 2.36])		
		
		rainfall['tidak_hujan'] = fuzz.trapmf(rainfall.universe, [0, 0, 0.5, 1])
		rainfall['ringan'] = fuzz.trimf(rainfall.universe, [0.5, 3, 5])
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
	
	def add_distance_reading(self, distance: float) -> None:
		if distance < 0:
			raise ValueError("Distance cannot be negative")
		self.distance_history.append(distance)
		self.reading_count += 1
	
	def calculate_average_rate_change(self) -> float:
		if len(self.distance_history) < 2:
			return 0.0
		
		num_intervals = len(self.distance_history) - 1
		time_span_seconds = num_intervals * self.reading_interval_seconds
		
		if time_span_seconds < 1:
			return 0.0
		
		distance_change = self.distance_history[0] - self.distance_history[-1]
		return distance_change / (time_span_seconds / 60.0)
	
	def get_status_message(self, warning_level: str, risk_score: float, avg_rate: float) -> str:
		g = self.RATE_FACTOR
		
		if warning_level == "BANJIR":
			if risk_score >= 95:
				return "🚨 CRITICAL FLOOD EMERGENCY! Immediate evacuation required!"
			elif avg_rate < -8 * g:
				return "🚨 EXTREME FLOOD - Water rising extremely fast!"
			elif risk_score >= 85:
				return "⚠️ SEVERE FLOOD WARNING - High water levels detected!"
			else:
				return "⚠️ FLOOD WARNING - Water has reached critical level!"
		
		elif warning_level == "SIAGA II":
			if avg_rate < -7 * g:
				return "🔴 SIAGA II - Final preparations! Evacuate immediately if heavy rain!"
			elif risk_score >= 75:
				return "🔴 SIAGA II - HIGH RISK! Be ready to evacuate!"
			elif risk_score >= 50:
				return "🔴 SIAGA II - Water level critical, prepare to evacuate!"
			else:
				return "🔴 SIAGA II - Water approaching flood level!"
		
		elif warning_level == "SIAGA I":
			if avg_rate < -5 * g:
				return "🟠 SIAGA I - Water rising rapidly, monitor closely!"
			elif risk_score >= 60:
				return "🟠 SIAGA I - Elevated risk, prepare evacuation supplies!"
			else:
				return "🟠 SIAGA I - Water level elevated, stay alert!"
		
		else:
			if risk_score >= 85:
				return "⚡ WARNING - Risk elevated despite normal water level!"
			elif risk_score >= 50:
				return "ℹ️ MONITORING - Elevated risk factors detected!"
			elif avg_rate < -2 * g:
				return "ℹ️ WATCH - Water level rising steadily!"
			else:
				return "✅ NORMAL - Water level safe!"
	
	def _calculate_fallback_risk(self, current_distance: float, 
								 avg_rate: float, 
								 current_rainfall: float,
								 rate_override: float) -> float:
		g = self.RATE_FACTOR
		
		dist_risk = 0
		if current_distance <= self.banjir_level:
			dist_risk = 50
		elif current_distance <= self.siaga_level:
			norm_pos = (self.siaga_level - current_distance) / (self.siaga_level - self.banjir_level)
			dist_risk = 25 + (norm_pos * 25)
		
		rate_risk = max(0, min(35, (-avg_rate / g) * 2))
		rain_risk = min(15, current_rainfall * 0.75)
		total_risk = dist_risk + rate_risk + rain_risk + rate_override
		
		return min(100, total_risk)
	
	def calculate_risk(self, current_distance: float, 
					   current_rainfall_mm_per_hour: float = 0) -> Dict[str, any]:
		if self.calibration_height is None:
			raise ValueError("System not calibrated")
		if current_distance < 0:
			raise ValueError("Distance cannot be negative")
		if current_rainfall_mm_per_hour < 0:
			raise ValueError("Rainfall cannot be negative")
		
		self.add_distance_reading(current_distance)
		avg_rate = self.calculate_average_rate_change()
		g = self.RATE_FACTOR
		
		rate_override = 0
		if avg_rate > 2 * g:
			rate_override = 25
		elif avg_rate > 1.5 * g:
			rate_override = 12
		
		self.fuzzy_system.input['water_level'] = current_distance
		self.fuzzy_system.input['avg_rate_change'] = avg_rate
		self.fuzzy_system.input['rainfall'] = current_rainfall_mm_per_hour
		
		try:
			self.fuzzy_system.compute()
			risk_score = min(100, self.fuzzy_system.output['flood_risk'] + rate_override)
		except:
			risk_score = self._calculate_fallback_risk(
				current_distance, avg_rate, current_rainfall_mm_per_hour, rate_override
			)
		
		warning_level = self._determine_warning_level(risk_score, current_distance, current_rainfall_mm_per_hour)
		old_warning = self.previous_warning_level
		self.previous_warning_level = warning_level
		
		return {
			'reading_number': self.reading_count,
			'current_distance': current_distance,
			'rate_change_cm_per_sec': avg_rate / 60.0,
			'avg_rate_change_cm_per_min': avg_rate,
			'current_rainfall_mm_per_hour': current_rainfall_mm_per_hour,
			'risk_score': round(risk_score, 2),
			'warning_level': warning_level,
			'previous_warning_level': old_warning,
			'status_message': self.get_status_message(warning_level, risk_score, avg_rate)
		}
	
	def _determine_warning_level(self, risk_score: float, current_distance: float, rainfall: float) -> str:
		avg_rate = self.calculate_average_rate_change()
		g = self.RATE_FACTOR
		
		if current_distance <= self.banjir_level:
			base = "BANJIR"
		elif current_distance <= self.siaga_level:
			mid = (self.siaga_level + self.banjir_level) / 2
			base = "SIAGA II" if current_distance <= mid else "SIAGA I"
		else:
			base = "NORMAL"
		
		if avg_rate > 12 * g:
			return "BANJIR"
		
		if base == "BANJIR":
			if avg_rate < -5 * g and current_distance > self.banjir_level + 15:
				return "SIAGA II"
			return "BANJIR"
		
		if base == "SIAGA II":
			if current_distance <= self.banjir_level + 8:
				return "BANJIR"
			if avg_rate > 1 * g:
				return "BANJIR"
			if risk_score >= 70:
				return "BANJIR"
			if current_distance <= self.banjir_level + 3 and avg_rate <= 0:
				return "BANJIR"
			if avg_rate < -5 * g and risk_score < 30:
				return "SIAGA I"
			return "SIAGA II"
		
		if base == "SIAGA I":
			if avg_rate > 8 * g:
				return "BANJIR"
			if avg_rate > 4 * g or (current_distance <= self.banjir_level + 12 and risk_score >= 60):
				return "SIAGA II"
			if avg_rate < -6 * g and risk_score < 25:
				return "NORMAL"
			return "SIAGA I"
		
		if base == "NORMAL":
			if avg_rate > 10 * g:
				return "BANJIR"
			if avg_rate > 7 * g:
				return "SIAGA II"
			if avg_rate > 5 * g:
				return "SIAGA I"
			if risk_score >= 75:
				return "SIAGA I"
			return "NORMAL"
		
		return base
	
	def _reset_system(self) -> None:
		self.distance_history.clear()
		self.reading_count = 0
		self.previous_warning_level = None
	
	def reset_history(self) -> None:
		self._reset_system()
	
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