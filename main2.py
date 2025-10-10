import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from collections import deque

class DynamicFuzzyFloodWarningSystem:
	def __init__(self, reading_interval_seconds=1):
		"""
		Initialize the fuzzy logic system with dynamic calibration capability
		Three warning levels: NORMAL, SIAGA, BANJIR
		Uses real-time measurements: water level + average rate of change (cm/min) + current rainfall
		
		Parameters:
		reading_interval_seconds: Time interval between readings (default: 1 second)
		"""
		self.calibration_height = None
		self.siaga_level = None
		self.banjir_level = None
		self.fuzzy_system = None
		self.previous_warning_level = None
		self.reading_interval_seconds = reading_interval_seconds
		
		# Store readings for 60-reading average calculation (FIFO queue)
		self.distance_history = deque(maxlen=60)
		self.reading_count = 0
	
	def calibrate(self, ground_distance, siaga_level_override=None, banjir_level_override=None):
		"""
		Calibrate the system with current ground distance
		
		Parameters:
		ground_distance: Current distance from sensor to ground (cm)
		siaga_level_override: Optional override for siaga level (cm)
		banjir_level_override: Optional override for banjir level (cm)
		
		If overrides are not provided:
		- banjir_level = ground_distance
		- siaga_level = ground_distance + 30
		"""
		self.calibration_height = ground_distance
		
		# Use override values if provided, otherwise use defaults
		if banjir_level_override is not None:
			self.banjir_level = banjir_level_override
		else:
			self.banjir_level = ground_distance
		
		if siaga_level_override is not None:
			self.siaga_level = siaga_level_override
		else:
			self.siaga_level = ground_distance + 30
		
		# Validate that siaga > banjir
		if self.siaga_level <= self.banjir_level:
			raise ValueError("siaga_level must be greater than banjir_level")
		
		self.fuzzy_system = self._create_fuzzy_system()
		
		override_msg = ""
		if siaga_level_override is not None or banjir_level_override is not None:
			override_msg = " (with overrides)"
		
		print(f"System calibrated{override_msg}: Ground={ground_distance}cm, Siaga={self.siaga_level}cm, Banjir={self.banjir_level}cm")
		print(f"Reading interval: {self.reading_interval_seconds} second(s)")
		print(f"History size: 60 readings (FIFO queue)")
	
	def _create_fuzzy_system(self):
		"""Create the fuzzy logic control system with IMPROVED 4-level water categories"""
	
		water_level_norm = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'water_level_norm')
		avg_rate_change = ctrl.Antecedent(np.arange(-1, 1.01, 0.01), 'avg_rate_change')
		rainfall_norm = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'rainfall_norm')
		flood_risk = ctrl.Consequent(np.arange(0, 101, 1), 'flood_risk', defuzzify_method='centroid')
		
		# IMPROVED: 4-level system with progressive BANJIR zones
		water_level_norm['normal'] = fuzz.trapmf(water_level_norm.universe, [0, 0, 0.1, 0.25])
		water_level_norm['siaga'] = fuzz.trimf(water_level_norm.universe, [0.15, 0.4, 0.65])
		water_level_norm['banjir_ringan'] = fuzz.trimf(water_level_norm.universe, [0.5, 0.7, 0.85])
		water_level_norm['banjir_parah'] = fuzz.trapmf(water_level_norm.universe, [0.75, 0.9, 1.0, 1.0])
		
		# IMPROVED: 8-level rate of change with better granularity at extremes
		avg_rate_change['turun sangat cepat'] = fuzz.trapmf(avg_rate_change.universe, [-1, -1, -0.6, -0.4])
		avg_rate_change['turun cepat'] = fuzz.trimf(avg_rate_change.universe, [-0.5, -0.3, -0.15])
		avg_rate_change['turun lambat'] = fuzz.trimf(avg_rate_change.universe, [-0.2, -0.1, -0.03])
		avg_rate_change['stabil'] = fuzz.trimf(avg_rate_change.universe, [-0.05, 0, 0.05])
		avg_rate_change['naik lambat'] = fuzz.trimf(avg_rate_change.universe, [0.03, 0.1, 0.2])
		avg_rate_change['naik cepat'] = fuzz.trimf(avg_rate_change.universe, [0.15, 0.3, 0.5])
		avg_rate_change['naik sangat cepat'] = fuzz.trimf(avg_rate_change.universe, [0.4, 0.65, 0.85])
		avg_rate_change['naik ekstrem'] = fuzz.trapmf(avg_rate_change.universe, [0.75, 0.9, 1, 1])
		
		# IMPROVED: 6-level rainfall with better extreme weather differentiation
		rainfall_norm['tidak_hujan'] = fuzz.trapmf(rainfall_norm.universe, [0, 0, 0.02, 0.04])
		rainfall_norm['ringan'] = fuzz.trimf(rainfall_norm.universe, [0.02, 0.1, 0.2])
		rainfall_norm['sedang'] = fuzz.trimf(rainfall_norm.universe, [0.15, 0.3, 0.45])
		rainfall_norm['lebat'] = fuzz.trimf(rainfall_norm.universe, [0.35, 0.55, 0.7])
		rainfall_norm['sangat_lebat'] = fuzz.trimf(rainfall_norm.universe, [0.6, 0.75, 0.88])
		rainfall_norm['ekstrem'] = fuzz.trapmf(rainfall_norm.universe, [0.8, 0.92, 1.0, 1.0])
		
		# IMPROVED: More granular risk output with 4 levels
		flood_risk['low'] = fuzz.trapmf(flood_risk.universe, [0, 0, 15, 30])
		flood_risk['medium'] = fuzz.trimf(flood_risk.universe, [25, 45, 65])
		flood_risk['high'] = fuzz.trimf(flood_risk.universe, [60, 75, 88])
		flood_risk['critical'] = fuzz.trapmf(flood_risk.universe, [85, 92, 100, 100])
		
		rules = self._define_fuzzy_rules(water_level_norm, avg_rate_change, rainfall_norm, flood_risk)
		
		flood_ctrl = ctrl.ControlSystem(rules)
		return ctrl.ControlSystemSimulation(flood_ctrl)
	
	def _define_fuzzy_rules(self, water_level, rate_change, rainfall_norm, flood_risk):
		"""Improved fuzzy rules with progressive risk increase across all input ranges"""
		rules = []
		
		# ========== BANJIR PARAH LEVEL (0.75-1.0 normalized) ==========
		rules.append(ctrl.Rule(water_level['banjir_parah'] & rate_change['naik ekstrem'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['banjir_parah'] & rate_change['naik sangat cepat'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['banjir_parah'] & rate_change['naik cepat'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['banjir_parah'] & rate_change['naik lambat'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['banjir_parah'] & rate_change['stabil'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['banjir_parah'] & rate_change['turun lambat'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['banjir_parah'] & rate_change['turun cepat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['banjir_parah'] & rate_change['turun sangat cepat'], flood_risk['high']))
		
		# ========== BANJIR RINGAN LEVEL (0.5-0.85 normalized) ==========
		rules.append(ctrl.Rule(water_level['banjir_ringan'] & rate_change['naik ekstrem'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['banjir_ringan'] & rate_change['naik sangat cepat'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['banjir_ringan'] & rate_change['naik cepat'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['banjir_ringan'] & rate_change['naik lambat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['banjir_ringan'] & rate_change['stabil'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['banjir_ringan'] & rate_change['turun lambat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['banjir_ringan'] & rate_change['turun cepat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['banjir_ringan'] & rate_change['turun sangat cepat'], flood_risk['medium']))
		
		# ========== SIAGA LEVEL (0.15-0.65 normalized) ==========
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik ekstrem'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik sangat cepat'] & rainfall_norm['ekstrem'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik sangat cepat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik cepat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik lambat'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['stabil'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['turun lambat'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['turun cepat'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['turun sangat cepat'], flood_risk['low']))
		
		# ========== NORMAL LEVEL (0-0.25 normalized) ==========
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik ekstrem'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik sangat cepat'] & rainfall_norm['ekstrem'], flood_risk['critical']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik sangat cepat'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik cepat'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik lambat'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['stabil'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['turun lambat'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['turun cepat'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['turun sangat cepat'], flood_risk['low']))
		
		return rules
	
	def normalize_water_level(self, current_distance):
		"""Normalize water level (0=safe, 1=flood)"""
		if self.calibration_height is None:
			raise ValueError("System not calibrated. Call calibrate() first.")
		
		if current_distance >= self.siaga_level:
			return 0.0
		elif current_distance <= self.banjir_level:
			return 1.0
		else:
			total_range = self.siaga_level - self.banjir_level
			distance_from_flood = current_distance - self.banjir_level 
			return 1.0 - (distance_from_flood / total_range)
	
	def add_distance_reading(self, distance):
		"""
		Add a distance reading to the FIFO history queue
		Automatically removes oldest reading when queue is full (maxlen=60)
		"""
		self.distance_history.append(distance)
		self.reading_count += 1
	
	def calculate_average_rate_change(self):
		"""
		Calculate the average rate of change over stored readings
		Assumes fixed time interval between readings
		"""
		if len(self.distance_history) < 2:
			return 0.0
		
		# Calculate total time span based on number of readings and interval
		num_intervals = len(self.distance_history) - 1
		time_span_seconds = num_intervals * self.reading_interval_seconds
		
		if time_span_seconds < 1:
			return 0.0
		
		# Calculate distance change from first to last reading
		distance_change = self.distance_history[-1] - self.distance_history[0]
		
		# Convert to cm per minute
		time_span_minutes = time_span_seconds / 60.0
		avg_rate = distance_change / time_span_minutes
		
		return avg_rate
	
	def normalize_avg_rate_change(self, avg_rate_cm_per_min, max_rate=30.0):
		"""
		Normalize average rate of change to -1 to +1 scale
		
		Important: 
		- Positive avg_rate_cm_per_min means distance increasing (water dropping)
		- Negative avg_rate_cm_per_min means distance decreasing (water rising)
		
		For fuzzy system:
		- We want negative normalized values when water is RISING (distance decreasing)
		- We want positive normalized values when water is DROPPING (distance increasing)
		
		Therefore, we INVERT the sign before normalizing
		"""
		# INVERT THE SIGN: positive distance change → negative normalized (water dropping)
		inverted_rate = -avg_rate_cm_per_min
		
		# Now clip and normalize
		clipped_rate = np.clip(inverted_rate, -max_rate, max_rate)
		return clipped_rate / max_rate
	
	def normalize_rainfall(self, rainfall_mm_per_hour, max_rainfall=25):
		"""Normalize rainfall to 0-1 scale"""
		return np.clip(rainfall_mm_per_hour / max_rainfall, 0, 1)
	
	def calculate_time_to_flood(self, current_distance, avg_rate_cm_per_min):
		"""Calculate estimated time until water reaches flood level"""
		if current_distance <= self.banjir_level:
			return None, "Sudah banjir"
		
		if avg_rate_cm_per_min >= 0:
			return None, "Air turun/stabil"
		
		distance_to_flood = current_distance - self.banjir_level
		time_minutes = distance_to_flood / abs(avg_rate_cm_per_min)
		
		if time_minutes < 5:
			status = f"KRITIS! ~{int(time_minutes)} menit lagi"
		elif time_minutes < 15:
			status = f"SEGERA! ~{int(time_minutes)} menit lagi"
		elif time_minutes < 60:
			status = f"~{int(time_minutes)} menit lagi"
		else:
			hours = time_minutes / 60
			status = f"~{hours:.1f} jam lagi"
		
		return round(time_minutes, 1), status
	
	def calculate_risk(self, current_distance, current_rainfall_mm_per_hour=0):
		"""
		Calculate flood risk using FIFO queue for rate of change calculation
		
		Parameters:
		current_distance: Current distance reading (cm)
		current_rainfall_mm_per_hour: Current rainfall rate (mm/hour)
		
		Returns:
		Dictionary with risk assessment and system status
		"""
		if self.calibration_height is None:
			raise ValueError("System not calibrated. Call calibrate() first.")
		
		# Add reading to FIFO queue
		self.add_distance_reading(current_distance)
		
		# Calculate average rate of change
		avg_rate_cm_per_min = self.calculate_average_rate_change()
		
		# Normalize inputs
		water_level_normalized = self.normalize_water_level(current_distance)
		avg_rate_normalized = self.normalize_avg_rate_change(avg_rate_cm_per_min)
		rainfall_normalized = self.normalize_rainfall(current_rainfall_mm_per_hour)
		
		# DEBUG: Print what's going into the fuzzy system
		print(f"\n=== DEBUG INFO ===")
		print(f"Reading #{self.reading_count}")
		print(f"Current distance: {current_distance} cm")
		print(f"Readings in history: {len(self.distance_history)}/60")
		if len(self.distance_history) >= 2:
			print(f"Distance history (last 5): {list(self.distance_history)[-5:]}")
			print(f"First reading: {self.distance_history[0]}, Last: {self.distance_history[-1]}")
		print(f"Avg rate (raw): {avg_rate_cm_per_min:.6f} cm/min")
		print(f"Water level normalized: {water_level_normalized:.4f} ({self._determine_water_level_category(water_level_normalized)})")
		print(f"Avg rate normalized: {avg_rate_normalized:.4f} ({self._determine_avg_rate_category(avg_rate_normalized)})")
		print(f"Rainfall normalized: {rainfall_normalized:.4f} ({self._categorize_rainfall_hourly(rainfall_normalized)})")
		
		# Fuzzy logic computation
		self.fuzzy_system.input['water_level_norm'] = water_level_normalized
		self.fuzzy_system.input['avg_rate_change'] = avg_rate_normalized
		self.fuzzy_system.input['rainfall_norm'] = rainfall_normalized
		
		try:
			self.fuzzy_system.compute()
			risk_score = self.fuzzy_system.output['flood_risk']
			print(f"Fuzzy computation: SUCCESS")
			print(f"Risk score: {risk_score:.2f}%")
		except Exception as e:
			print(f"Fuzzy computation: FAILED - {e}")
			# Fallback calculation
			rate_risk = max(0, -avg_rate_normalized) * 35
			risk_score = water_level_normalized * 50 + rate_risk + rainfall_normalized * 15
			risk_score = min(100, risk_score)
			print(f"Using fallback calculation")
			print(f"  Water component: {water_level_normalized * 50:.2f}")
			print(f"  Rate component: {rate_risk:.2f}")
			print(f"  Rainfall component: {rainfall_normalized * 15:.2f}")
			print(f"  Total risk: {risk_score:.2f}%")
		
		print(f"===================\n")
		
		# Determine warning level and other metrics
		warning_level = self._determine_warning_level(risk_score, current_distance)
		notification_interval = self._get_notification_interval(warning_level, risk_score)
		rainfall_category = self._categorize_rainfall_hourly(rainfall_normalized)
		is_recovery = self._detect_recovery(warning_level)
		time_to_flood_min, time_status = self.calculate_time_to_flood(current_distance, avg_rate_cm_per_min)
		
		old_warning_level = self.previous_warning_level
		self.previous_warning_level = warning_level
		
		return {
			'reading_number': self.reading_count,
			'current_distance': current_distance,
			'water_depth_from_ground': self.calibration_height - current_distance,
			'avg_rate_change_cm_per_min': avg_rate_cm_per_min,
			'readings_count': len(self.distance_history),
			'water_level_normalized': water_level_normalized,
			'water_level_category': self._determine_water_level_category(water_level_normalized),
			'avg_rate_normalized': avg_rate_normalized,
			'avg_rate_category': self._determine_avg_rate_category(avg_rate_normalized),
			'current_rainfall': current_rainfall_mm_per_hour,
			'rainfall_normalized': rainfall_normalized,
			'current_rainfall_category': rainfall_category,
			'risk_score': risk_score,
			'risk_category': self._determine_flood_risk_category(risk_score),
			'warning_level': warning_level,
			'previous_warning_level': old_warning_level,
			'is_recovery': is_recovery,
			'notification_interval': notification_interval,
			'should_send_warning': self._should_send_warning(risk_score),
			'should_send_recovery_notification': is_recovery,
			'time_to_flood_minutes': time_to_flood_min,
			'time_to_flood_status': time_status,
			'status_message': self._get_status_message(warning_level, avg_rate_cm_per_min, 
													rainfall_category, time_to_flood_min, is_recovery),
			'thresholds': {
				'banjir': self.banjir_level,
				'siaga': self.siaga_level,
				'calibration': self.calibration_height
			}
		}
	
	def _detect_recovery(self, current_warning_level):
		"""Detect if status has recovered from SIAGA/BANJIR to NORMAL"""
		if self.previous_warning_level is None:
			return False
		
		if self.previous_warning_level in ["SIAGA", "BANJIR"] and current_warning_level == "NORMAL":
			return True
		
		return False
	
	def _get_fuzzy_category(self, variable_name, normalized_value):
		"""
		Dynamically determine category by finding the fuzzy set with highest membership
		This automatically adapts to any changes in membership function definitions
		
		Parameters:
		variable_name: Name of the fuzzy variable ('water_level_norm', 'avg_rate_change', 'rainfall_norm')
		normalized_value: The normalized input value
		
		Returns:
		String name of the category with highest membership degree
		"""
		if self.fuzzy_system is None:
			return "Unknown"
		
		# Get the fuzzy variable from the control system
		try:
			# Access the antecedent from the fuzzy system
			fuzzy_var = None
			for rule in self.fuzzy_system.ctrl.rules:
				for clause in rule.antecedent_terms:
					if clause.parent.label == variable_name:
						fuzzy_var = clause.parent
						break
				if fuzzy_var:
					break
			
			if fuzzy_var is None:
				return "Unknown"
			
			# Calculate membership degree for each fuzzy set
			max_membership = -1
			best_category = "Unknown"
			
			for term_name, term_mf in fuzzy_var.terms.items():
				membership = fuzz.interp_membership(fuzzy_var.universe, term_mf.mf, normalized_value)
				if membership > max_membership:
					max_membership = membership
					best_category = term_name
			
			return best_category
			
		except Exception as e:
			print(f"Error in fuzzy category determination: {e}")
			return "Unknown"
	
	def _categorize_rainfall_hourly(self, rainfall_normalized):
		"""
		Dynamically categorize rainfall based on fuzzy membership functions
		Automatically adapts to membership function changes
		"""
		category = self._get_fuzzy_category('rainfall_norm', rainfall_normalized)
		
		# Map fuzzy set names to readable Indonesian names
		category_map = {
			'tidak_hujan': 'Tidak Hujan',
			'ringan': 'Hujan Ringan',
			'sedang': 'Hujan Sedang',
			'lebat': 'Hujan Lebat',
			'sangat_lebat': 'Hujan Sangat Lebat',
			'ekstrem': 'Hujan Ekstrem'
		}
		
		return category_map.get(category, category)
	
	def _determine_water_level_category(self, water_level_normalized):
		"""
		Dynamically categorize water level based on fuzzy membership functions
		Automatically adapts to membership function changes
		"""
		category = self._get_fuzzy_category('water_level_norm', water_level_normalized)
		
		# Map fuzzy set names to readable Indonesian names
		category_map = {
			'normal': 'NORMAL',
			'siaga': 'SIAGA',
			'banjir_ringan': 'BANJIR RINGAN',
			'banjir_parah': 'BANJIR PARAH'
		}
		
		return category_map.get(category, category.upper())

	def _determine_warning_level(self, risk_score, current_distance):
		"""Determine warning level"""
		if current_distance <= self.banjir_level:
			return "BANJIR"
		elif current_distance <= self.siaga_level:
			if risk_score >= 75:
				return "BANJIR"
			else:
				return "SIAGA"
		else:
			if risk_score >= 50:
				return "SIAGA"
			else:
				return "NORMAL"
	
	def _determine_avg_rate_category(self, avg_rate_normalized):
		"""
		Dynamically categorize rate of change based on fuzzy membership functions
		Automatically adapts to membership function changes
		
		Remember in normalized form:
		- Negative normalized = water RISING
		- Positive normalized = water DROPPING
		"""
		category = self._get_fuzzy_category('avg_rate_change', avg_rate_normalized)
		
		# Map fuzzy set names to readable Indonesian names
		category_map = {
			'turun sangat cepat': 'Turun Sangat Cepat',
			'turun cepat': 'Turun Cepat',
			'turun lambat': 'Turun Lambat',
			'stabil': 'Stabil',
			'naik lambat': 'Naik Lambat',
			'naik cepat': 'Naik Cepat',
			'naik sangat cepat': 'Naik Sangat Cepat',
			'naik ekstrem': 'Naik Ekstrem'
		}
		
		return category_map.get(category, category.title())

	def _determine_flood_risk_category(self, risk_score):
		"""Categorize flood risk"""
		if risk_score >= 70:
			return "Tinggi"
		elif risk_score >= 40:
			return "Sedang"
		else:
			return "Rendah"

	def _should_send_warning(self, risk_score):
		"""Determine if warning should be sent - only when risk ≥75%"""
		return risk_score >= 75
	
	def _get_notification_interval(self, warning_level, risk_score):
		"""Get notification interval - only for high risk (≥75%)"""
		if risk_score >= 75:
			if warning_level == "BANJIR":
				return 5
			elif warning_level == "SIAGA":
				return 10
		
		return None
	
	def _get_status_message(self, warning_level, avg_rate, current_rain_category, time_to_flood, is_recovery=False):
		"""Generate detailed status messages for different scenarios"""
		
		# Recovery messages
		if is_recovery:
			if avg_rate > 0.2:
				msg = "✅ PEMULIHAN CEPAT - Air surut dengan sangat cepat! Tingkat air kembali normal."
			elif avg_rate > 0.1:
				msg = "✅ KONDISI MEMBAIK - Air telah surut. Tingkat air kembali normal."
			elif avg_rate > 0.05:
				msg = "✅ SITUASI AMAN - Air surut perlahan. Tingkat air kembali normal."
			else:
				msg = "✅ STATUS NORMAL KEMBALI - Tingkat air telah stabil di zona aman."
			
			if self.previous_warning_level == "BANJIR":
				msg += " Status banjir telah berakhir. Tetap waspada terhadap kondisi cuaca."
			elif self.previous_warning_level == "SIAGA":
				msg += " Status siaga telah dicabut. Pantau terus perkembangan situasi."
			
			return msg
		
		# BANJIR status messages
		if warning_level == "BANJIR":
			if avg_rate < -0.28:
				if current_rain_category == "Hujan Sangat Lebat":
					msg = "🚨 DARURAT BANJIR! Air naik sangat cepat dengan hujan sangat lebat! EVAKUASI SEKARANG!"
				else:
					msg = "🚨 BANJIR KRITIS! Air naik sangat cepat! SEGERA EVAKUASI!"
			elif avg_rate < -0.14:
				msg = "🚨 STATUS BANJIR! Air naik cepat! EVAKUASI SEGERA!"
			elif avg_rate > 0.05:
				msg = "⚠️ BANJIR AKTIF - Air mulai surut. Tetap di lokasi aman sampai air normal kembali."
			else:
				msg = "⚠️ BANJIR STABIL - Air di level kritis. Jika belum evakuasi, lakukan sekarang!"
		
		# SIAGA status messages
		elif warning_level == "SIAGA":
			if time_to_flood and time_to_flood < 5:
				msg = f"🚨 SIAGA DARURAT! Banjir dalam ~{int(time_to_flood)} menit! SEGERA EVAKUASI SEKARANG!"
			elif time_to_flood and time_to_flood < 15:
				msg = f"⚠️ SIAGA TINGGI! Banjir dalam ~{int(time_to_flood)} menit! Persiapkan evakuasi!"
			elif avg_rate < -0.28:
				msg = "🚨 SIAGA DARURAT! Air naik sangat cepat! Bersiap evakuasi segera!"
			elif avg_rate < -0.14:
				msg = "⚠️ SIAGA TINGGI! Air naik cepat! Persiapkan evakuasi!"
			elif avg_rate > 0.05:
				msg = "⚠️ SIAGA - Air mulai surut. Situasi membaik, tetap pantau!"
			else:
				msg = "⚠️ SIAGA AKTIF - Air di level siaga. Pantau terus perkembangan!"
		
		# NORMAL status messages
		else:
			if avg_rate < -0.28:
				msg = "⚡ NORMAL - PERHATIAN! Air naik sangat cepat! Pantau ketat situasi!"
			elif avg_rate < -0.14:
				msg = "⚡ NORMAL - WASPADA! Air naik cepat! Pantau ketat!"
			elif avg_rate > 0.14:
				msg = "✅ NORMAL AMAN - Air surut cepat. Kondisi sangat baik!"
			elif avg_rate > 0.05:
				msg = "✅ NORMAL AMAN - Air surut perlahan. Kondisi baik!"
			else:
				msg = "✅ NORMAL AMAN - Kondisi air stabil. Tidak ada ancaman."
		
		return msg

	def reset_history(self):
		"""Reset the reading history (useful for testing or recalibration)"""
		self.distance_history.clear()
		self.reading_count = 0
		self.previous_warning_level = None
		print("History reset. Ready for new readings.")


if __name__ == "__main__":
	print("=== Dynamic Fuzzy Flood Warning System (FIFO Queue) ===\n")
	
	# Initialize system with 1-second intervals between readings
	system = DynamicFuzzyFloodWarningSystem(reading_interval_seconds=1)
	
	# Calibrate with default settings
	print("=== Calibration ===")
	system.calibrate(ground_distance=100)
	
	print("\n=== Simulating water rising scenario ===")
	print("Each reading is 1 second apart")
	print("System stores up to 60 readings (FIFO)\n")
	
	# Simulate readings - water rising scenario
	distances = [150, 149.92, 149.75, 149.5, 149.2, 148.8, 148.3, 147.7, 147.0, 146.2,]
	rainfall = [0, 0, 0, 5, 5, 10, 15, 20, 20, 15, 10, 5, 5, 0, 0]
	
	for i, distance in enumerate(distances):
		result = system.calculate_risk(
			current_distance=distance,
			current_rainfall_mm_per_hour=rainfall[i]
		)