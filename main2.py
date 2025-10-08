import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime, timedelta

class DynamicFuzzyFloodWarningSystem:
	def __init__(self):
		"""
		Initialize the fuzzy logic system with dynamic calibration capability
		Three warning levels: NORMAL, SIAGA, BANJIR
		Uses real-time measurements: water level + average rate of change (cm/min) + current rainfall
		"""
		self.calibration_height = None
		self.siaga_level = None
		self.banjir_level = None
		self.fuzzy_system = None
		self.previous_warning_level = None
		
		# Store readings for 60-second average calculation
		self.distance_history = deque(maxlen=60)
		self.timestamp_history = deque(maxlen=60)
	
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
	
	def _create_fuzzy_system(self):
		"""Create the fuzzy logic control system with tuned membership functions"""
		
		water_level_norm = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'water_level_norm')
		avg_rate_change = ctrl.Antecedent(np.arange(-1, 1.01, 0.01), 'avg_rate_change')
		rainfall_norm = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'rainfall_norm')
		flood_risk = ctrl.Consequent(np.arange(0, 101, 1), 'flood_risk', defuzzify_method='centroid')
		
		# TUNED: Water level membership functions - more sensitive
		water_level_norm['normal'] = fuzz.trapmf(water_level_norm.universe, [0, 0, 0.3, 0.5])
		water_level_norm['siaga'] = fuzz.trimf(water_level_norm.universe, [0.45, 0.7, 0.85])
		water_level_norm['banjir'] = fuzz.trapmf(water_level_norm.universe, [0.8, 0.95, 1.0, 1.0])
		
		# TUNED: Rate of change - more granular and realistic
		# Water DROPPING (negative normalized values)
		avg_rate_change['turun cepat'] = fuzz.trapmf(avg_rate_change.universe, [-1, -1, -0.45, -0.2])
		avg_rate_change['turun lambat'] = fuzz.trimf(avg_rate_change.universe, [-0.3, -0.15, -0.05])

		# Water STABLE
		avg_rate_change['stabil'] = fuzz.trimf(avg_rate_change.universe, [-0.08, 0, 0.08])

		# Water RISING (positive normalized values)  
		avg_rate_change['naik lambat'] = fuzz.trimf(avg_rate_change.universe, [0.05, 0.15, 0.3])
		avg_rate_change['naik cepat'] = fuzz.trimf(avg_rate_change.universe, [0.2, 0.35, 0.6])
		avg_rate_change['naik sangat cepat'] = fuzz.trapmf(avg_rate_change.universe, [0.45, 0.7, 1, 1])
		
		# TUNED: Rainfall - adjusted for better sensitivity
		rainfall_norm['tidak_hujan'] = fuzz.trapmf(rainfall_norm.universe, [0, 0, 0.03, 0.06])
		rainfall_norm['ringan'] = fuzz.trimf(rainfall_norm.universe, [0.03, 0.15, 0.25])
		rainfall_norm['sedang'] = fuzz.trimf(rainfall_norm.universe, [0.18, 0.35, 0.5])
		rainfall_norm['lebat'] = fuzz.trimf(rainfall_norm.universe, [0.4, 0.65, 0.85])
		rainfall_norm['sangat_lebat'] = fuzz.trapmf(rainfall_norm.universe, [0.75, 0.9, 1.0, 1.0])
		
		# TUNED: Risk output - narrower medium, earlier high threshold
		flood_risk['low'] = fuzz.trapmf(flood_risk.universe, [0, 0, 20, 40])
		flood_risk['medium'] = fuzz.trimf(flood_risk.universe, [30, 55, 75])
		flood_risk['high'] = fuzz.trapmf(flood_risk.universe, [65, 80, 100, 100])
		
		rules = self._define_fuzzy_rules(water_level_norm, avg_rate_change, rainfall_norm, flood_risk)
		
		flood_ctrl = ctrl.ControlSystem(rules)
		return ctrl.ControlSystemSimulation(flood_ctrl)
	
	def _define_fuzzy_rules(self, water_level, rate_change, rainfall_norm, flood_risk):
		"""Define fuzzy logic rules - ENHANCED with more critical scenarios"""
		rules = []
		
		# BANJIR level rules - always high risk when at flood level
		rules.append(ctrl.Rule(water_level['banjir'], flood_risk['high']))
		
		# SIAGA level rules - CRITICAL: these need to trigger warnings
		# Very fast rising at SIAGA = HIGH RISK
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik sangat cepat'], flood_risk['high']))
		
		# Fast rising at SIAGA with any significant rain = HIGH RISK
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik cepat'] & rainfall_norm['sangat_lebat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik cepat'] & rainfall_norm['lebat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik cepat'] & rainfall_norm['sedang'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik cepat'], flood_risk['high']))
		
		# Slow rising at SIAGA with heavy rain = HIGH RISK
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik lambat'] & rainfall_norm['sangat_lebat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik lambat'] & rainfall_norm['lebat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik lambat'] & rainfall_norm['sedang'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik lambat'], flood_risk['medium']))
		
		# Stable or falling at SIAGA
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['stabil'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['turun lambat'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['turun cepat'], flood_risk['low']))
		
		# NORMAL level rules
		# Very fast rising even at normal level with heavy rain = HIGH RISK
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik sangat cepat'] & rainfall_norm['sangat_lebat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik sangat cepat'] & rainfall_norm['lebat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik sangat cepat'], flood_risk['medium']))
		
		# Fast rising at normal with heavy rain
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik cepat'] & rainfall_norm['sangat_lebat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik cepat'] & rainfall_norm['lebat'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik cepat'], flood_risk['medium']))
		
		# Slow changes at normal level
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik lambat'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['stabil'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['turun lambat'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['turun cepat'], flood_risk['low']))
		
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
	
	def add_distance_reading(self, distance, timestamp=None):
		"""Add a distance reading to the history for average calculation"""
		if timestamp is None:
			timestamp = datetime.now()
		
		self.distance_history.append(distance)
		self.timestamp_history.append(timestamp)
	
	def calculate_average_rate_change(self):
		"""Calculate the average rate of change over the last 60 seconds"""
		if len(self.distance_history) < 2:
			return 0.0
		
		time_span_seconds = (self.timestamp_history[-1] - self.timestamp_history[0]).total_seconds()
		
		if time_span_seconds < 1:
			return 0.0
		
		distance_change = self.distance_history[-1] - self.distance_history[0]
		time_span_minutes = time_span_seconds / 60.0
		avg_rate = distance_change / time_span_minutes
		
		return avg_rate
	
	def normalize_avg_rate_change(self, avg_rate_cm_per_min, max_rate=0.4):
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
	
	def calculate_risk(self, current_distance, current_rainfall_mm_per_hour=0, timestamp=None):
		"""Calculate flood risk using 60-second average rate of change - DEBUG VERSION"""
		if self.calibration_height is None:
			raise ValueError("System not calibrated. Call calibrate() first.")
		
		self.add_distance_reading(current_distance, timestamp)
		avg_rate_cm_per_min = self.calculate_average_rate_change()
		
		# Normalize inputs
		water_level_normalized = self.normalize_water_level(current_distance)
		avg_rate_normalized = self.normalize_avg_rate_change(avg_rate_cm_per_min)
		rainfall_normalized = self.normalize_rainfall(current_rainfall_mm_per_hour)
		
		# DEBUG: Print what's going into the fuzzy system
		print(f"\n=== DEBUG INFO ===")
		print(f"Current distance: {current_distance} cm")
		print(f"Readings in history: {len(self.distance_history)}")
		if len(self.distance_history) >= 2:
			print(f"Distance history: {list(self.distance_history)[-5:]}")  # Last 5
			print(f"First reading: {self.distance_history[0]}, Last: {self.distance_history[-1]}")
		print(f"Avg rate (raw): {avg_rate_cm_per_min:.6f} cm/min")
		print(f"Water level normalized: {water_level_normalized:.4f}")
		print(f"Avg rate normalized: {avg_rate_normalized:.4f}")
		print(f"Rainfall normalized: {rainfall_normalized:.4f}")
		
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
		
		# ... rest of the method remains the same
		warning_level = self._determine_warning_level(risk_score, current_distance)
		notification_interval = self._get_notification_interval(warning_level, risk_score)
		rainfall_category = self._categorize_rainfall_hourly(current_rainfall_mm_per_hour)
		is_recovery = self._detect_recovery(warning_level)
		time_to_flood_min, time_status = self.calculate_time_to_flood(current_distance, avg_rate_cm_per_min)
		
		old_warning_level = self.previous_warning_level
		self.previous_warning_level = warning_level
		
		return {
			'current_distance': current_distance,
			'water_depth_from_ground': self.calibration_height - current_distance,
			'avg_rate_change_cm_per_min': avg_rate_cm_per_min,
			'readings_count': len(self.distance_history),
			'water_level_normalized': water_level_normalized,
			'avg_rate_normalized': avg_rate_normalized,
			'avg_rate_category': self._determine_avg_rate_category(avg_rate_cm_per_min),
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
	
	def _categorize_rainfall_hourly(self, rainfall_mm_per_hour):
		"""Categorize rainfall (BMKG)"""
		if rainfall_mm_per_hour > 20:
			return "Hujan Sangat Lebat"
		elif rainfall_mm_per_hour >= 10:
			return "Hujan Lebat"
		elif rainfall_mm_per_hour >= 5:
			return "Hujan Sedang"
		elif rainfall_mm_per_hour >= 1:
			return "Hujan Ringan"
		else:
			return "Tidak Hujan"
	
	def _determine_warning_level(self, risk_score, current_distance):
		"""Determine warning level"""
		if current_distance <= self.banjir_level:
			return "BANJIR"
		elif current_distance <= self.siaga_level:
			if risk_score >= 70:
				return "BANJIR"
			else:
				return "SIAGA"
		else:
			if risk_score >= 60:
				return "SIAGA"
			else:
				return "NORMAL"
	
	def _determine_avg_rate_category(self, avg_rate_cm_per_min):
		"""
		Categorize average rate of change
		
		Remember: 
		- Negative avg_rate = distance decreasing = water RISING
		- Positive avg_rate = distance increasing = water DROPPING
		"""
		if avg_rate_cm_per_min < -0.28:  # Very fast RISING
			return "Naik Sangat Cepat"
		elif avg_rate_cm_per_min < -0.14:  # Fast RISING
			return "Naik Cepat"
		elif avg_rate_cm_per_min < -0.05:  # Slow RISING
			return "Naik Lambat"
		elif avg_rate_cm_per_min < 0.05:  # Stable
			return "Stabil"
		elif avg_rate_cm_per_min < 0.14:  # Slow DROPPING
			return "Turun Lambat"
		else:  # Fast DROPPING
			return "Turun Cepat"

	def _determine_flood_risk_category(self, risk_score):
		"""Categorize flood risk"""
		if risk_score >= 70:
			return "Tinggi"
		elif risk_score >= 40:
			return "Sedang"
		else:
			return "Rendah"

	def _should_send_warning(self, risk_score):
		"""Determine if warning should be sent - only when risk ≥70%"""
		return risk_score >= 70
	
	def _get_notification_interval(self, warning_level, risk_score):
		"""Get notification interval - only for high risk (≥70%)"""
		if risk_score >= 70:
			if warning_level == "BANJIR":
				return 5
			elif warning_level == "SIAGA":
				return 10
		
		return None
	
	def _get_status_message(self, warning_level, avg_rate, current_rain_category, time_to_flood, is_recovery=False):
		"""Generate detailed status messages for different scenarios"""
		
		# ========== RECOVERY SCENARIOS ==========
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
		
		# ========== BANJIR (FLOOD) STATUS ==========
		if warning_level == "BANJIR":
			# Critical rising scenarios
			if avg_rate < -0.28:  # Very fast rising
				if current_rain_category == "Hujan Sangat Lebat":
					msg = "🚨 DARURAT BANJIR! Air naik sangat cepat dengan hujan sangat lebat! EVAKUASI SEKARANG! Bahaya ekstrem!"
				elif current_rain_category == "Hujan Lebat":
					msg = "🚨 DARURAT BANJIR! Air naik sangat cepat dengan hujan lebat! SEGERA EVAKUASI! Situasi sangat berbahaya!"
				elif current_rain_category in ["Hujan Sedang", "Hujan Ringan"]:
					msg = f"🚨 BANJIR KRITIS! Air naik sangat cepat ({current_rain_category})! EVAKUASI SEKARANG!"
				else:
					msg = "🚨 BANJIR KRITIS! Air naik sangat cepat meski tanpa hujan! SEGERA EVAKUASI! Kemungkinan banjir dari hulu!"
			
			elif avg_rate < -0.14:  # Fast rising
				if current_rain_category in ["Hujan Sangat Lebat", "Hujan Lebat"]:
					msg = f"🚨 STATUS BANJIR! Air naik cepat dengan {current_rain_category}! EVAKUASI SEGERA!"
				elif current_rain_category == "Hujan Sedang":
					msg = "⚠️ STATUS BANJIR! Air naik cepat dengan hujan sedang! Lakukan evakuasi sekarang!"
				else:
					msg = "⚠️ STATUS BANJIR! Air naik cepat! EVAKUASI diperlukan!"
			
			elif avg_rate < -0.05:  # Slow rising
				if current_rain_category in ["Hujan Sangat Lebat", "Hujan Lebat"]:
					msg = f"⚠️ STATUS BANJIR! Air terus naik dengan {current_rain_category}! Lakukan evakuasi!"
				else:
					msg = "⚠️ STATUS BANJIR! Air masih naik perlahan. Lakukan evakuasi sekarang!"
			
			elif avg_rate > 0.05:  # Water dropping
				if avg_rate > 0.14:
					msg = "⚠️ BANJIR AKTIF - Air mulai surut cepat. Tetap di lokasi aman sampai air normal kembali."
				else:
					msg = "⚠️ BANJIR AKTIF - Air mulai surut perlahan. Jangan kembali, tetap di tempat aman."
			
			else:  # Stable at flood level
				if current_rain_category in ["Hujan Sangat Lebat", "Hujan Lebat"]:
					msg = f"⚠️ BANJIR STABIL - Air di level kritis, {current_rain_category}. Tetap di lokasi evakuasi!"
				else:
					msg = "⚠️ BANJIR STABIL - Air di level kritis. Jika belum evakuasi, lakukan sekarang!"
		
		# ========== SIAGA (ALERT) STATUS ==========
		elif warning_level == "SIAGA":
			# Critical time-based warnings
			if time_to_flood and time_to_flood < 5:
				msg = f"🚨 SIAGA DARURAT! Banjir dalam ~{int(time_to_flood)} menit! SEGERA EVAKUASI SEKARANG!"
			elif time_to_flood and time_to_flood < 10:
				msg = f"🚨 SIAGA KRITIS! Banjir dalam ~{int(time_to_flood)} menit! Bersiap evakuasi SEGERA!"
			elif time_to_flood and time_to_flood < 15:
				msg = f"⚠️ SIAGA TINGGI! Banjir dalam ~{int(time_to_flood)} menit! Persiapkan evakuasi!"
			
			# Very fast rising scenarios
			elif avg_rate < -0.28:
				if current_rain_category == "Hujan Sangat Lebat":
					msg = "🚨 SIAGA DARURAT! Air naik sangat cepat dengan hujan sangat lebat! Evakuasi dalam 15 menit!"
				elif current_rain_category == "Hujan Lebat":
					msg = "🚨 SIAGA KRITIS! Air naik sangat cepat dengan hujan lebat! Bersiap evakuasi segera!"
				elif current_rain_category == "Hujan Sedang":
					msg = "⚠️ SIAGA! Air naik sangat cepat dengan hujan sedang! Persiapkan evakuasi!"
				else:
					msg = "⚠️ SIAGA! Air naik sangat cepat! Bersiap untuk evakuasi segera!"
			
			# Fast rising scenarios
			elif avg_rate < -0.14:
				if current_rain_category in ["Hujan Sangat Lebat", "Hujan Lebat"]:
					msg = f"⚠️ SIAGA TINGGI! Air naik cepat dengan {current_rain_category}! Persiapkan evakuasi!"
				elif current_rain_category == "Hujan Sedang":
					msg = "⚠️ SIAGA! Air naik cepat dengan hujan sedang. Bersiap untuk evakuasi!"
				else:
					msg = "⚠️ SIAGA! Air naik cepat mendekati level banjir. Waspada tinggi!"
			
			# Slow rising scenarios
			elif avg_rate < -0.05:
				if current_rain_category in ["Hujan Sangat Lebat", "Hujan Lebat"]:
					msg = f"⚠️ SIAGA! Air naik perlahan dengan {current_rain_category}. Pantau terus dan bersiap!"
				elif current_rain_category == "Hujan Sedang":
					msg = "⚠️ SIAGA! Air naik perlahan dengan hujan sedang. Tetap waspada!"
				else:
					msg = "⚠️ SIAGA! Air naik perlahan mendekati level banjir. Waspada!"
			
			# Stable or dropping at alert level
			elif avg_rate > 0.05:
				if avg_rate > 0.14:
					msg = "⚠️ SIAGA - Air mulai surut cepat dari level siaga. Situasi membaik, tetap pantau!"
				else:
					msg = "⚠️ SIAGA - Air mulai surut perlahan. Situasi membaik, tetap waspada!"
			
			else:  # Stable at alert level
				if current_rain_category in ["Hujan Sangat Lebat", "Hujan Lebat"]:
					msg = f"⚠️ SIAGA AKTIF - Air stabil di level siaga, {current_rain_category}. Bersiap untuk kemungkinan evakuasi!"
				elif current_rain_category == "Hujan Sedang":
					msg = "⚠️ SIAGA AKTIF - Air stabil di level siaga, hujan sedang. Tetap waspada!"
				else:
					msg = "⚠️ SIAGA AKTIF - Air di level siaga. Pantau terus perkembangan!"
		
		# ========== NORMAL STATUS ==========
		else:
			# Fast rising at normal level (potential threat)
			if avg_rate < -0.28:
				if current_rain_category in ["Hujan Sangat Lebat", "Hujan Lebat"]:
					msg = f"⚡ NORMAL - PERHATIAN! Air naik sangat cepat dengan {current_rain_category}! Potensi SIAGA!"
				else:
					msg = "✅ NORMAL - Namun air naik sangat cepat! Pantau ketat situasi!"
			
			elif avg_rate < -0.14:
				if current_rain_category in ["Hujan Sangat Lebat", "Hujan Lebat"]:
					msg = f"⚡ NORMAL - WASPADA! Air naik cepat dengan {current_rain_category}! Pantau ketat!"
				elif current_rain_category == "Hujan Sedang":
					msg = "✅ NORMAL - Air naik cepat dengan hujan sedang. Tetap pantau!"
				else:
					msg = "✅ NORMAL - Air naik cepat. Pantau terus perkembangan!"
			
			elif avg_rate < -0.05:
				if current_rain_category in ["Hujan Sangat Lebat", "Hujan Lebat"]:
					msg = f"✅ NORMAL - Air naik perlahan, {current_rain_category}. Waspada!"
				else:
					msg = "✅ NORMAL - Air naik perlahan. Situasi terpantau."
			
			# Dropping water
			elif avg_rate > 0.14:
				msg = "✅ NORMAL AMAN - Air surut cepat. Kondisi sangat baik!"
			elif avg_rate > 0.05:
				msg = "✅ NORMAL AMAN - Air surut perlahan. Kondisi baik!"
			
			# Stable conditions
			else:
				if current_rain_category in ["Hujan Sangat Lebat", "Hujan Lebat"]:
					msg = f"✅ NORMAL - Air stabil meski {current_rain_category}. Tetap pantau!"
				elif current_rain_category == "Hujan Sedang":
					msg = "✅ NORMAL - Air stabil dengan hujan sedang. Situasi terkendali."
				elif current_rain_category == "Hujan Ringan":
					msg = "✅ NORMAL AMAN - Air stabil, hujan ringan. Kondisi baik."
				else:
					msg = "✅ NORMAL AMAN - Kondisi air stabil. Tidak ada ancaman."
		
		return msg

if __name__ == "__main__":
	system = DynamicFuzzyFloodWarningSystem()
	
	# Test with default calibration
	print("\n=== Testing with default calibration ===")
	system.calibrate(ground_distance=100)
	
	print("\n=== Simulating 60-second readings ===")
	print("Simulating water rising scenario...\n")
	
	base_time = datetime.now()
	
	distances = [200.0, 199.99, 199.98, 201, 199]
	
	for i, distance in enumerate(distances):
		timestamp = base_time + timedelta(seconds=i*6)
		
		result = system.calculate_risk(
			current_distance=distance,
			current_rainfall_mm_per_hour=0,
			timestamp=timestamp
		)