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
	
	def calibrate(self, ground_distance):
		"""
		Calibrate the system with current ground distance
		
		Parameters:
		ground_distance: Current distance from sensor to ground (h cm)
		"""
		self.calibration_height = ground_distance
		self.banjir_level = ground_distance
		self.siaga_level = ground_distance + 30
		
		self.fuzzy_system = self._create_fuzzy_system()
		
		print(f"System calibrated: Ground={ground_distance}cm, Siaga={self.siaga_level}cm, Banjir={self.banjir_level}cm")
	
	def _create_fuzzy_system(self):
		"""Create the fuzzy logic control system with tuned membership functions"""
		
		water_level_norm = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'water_level_norm')
		avg_rate_change = ctrl.Antecedent(np.arange(-1, 1.01, 0.01), 'avg_rate_change')
		rainfall_norm = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'rainfall_norm')
		flood_risk = ctrl.Consequent(np.arange(0, 101, 1), 'flood_risk')
		
		# TUNED: Water level membership functions - more sensitive
		water_level_norm['normal'] = fuzz.trapmf(water_level_norm.universe, [0, 0, 0.35, 0.55])
		water_level_norm['siaga'] = fuzz.trimf(water_level_norm.universe, [0.45, 0.7, 0.85])
		water_level_norm['banjir'] = fuzz.trapmf(water_level_norm.universe, [0.75, 0.9, 1.0, 1.0])
		
		# TUNED: Rate of change - more granular and realistic
		# Normalized values: -1.0 = -0.4 cm/min (or faster rising)
		# Average guideline: 0.067 cm/min = -0.1675 normalized
		avg_rate_change['naik sangat cepat'] = fuzz.trapmf(avg_rate_change.universe, [-1, -1, -0.7, -0.45])
		avg_rate_change['naik cepat'] = fuzz.trimf(avg_rate_change.universe, [-0.6, -0.35, -0.2])
		avg_rate_change['naik lambat'] = fuzz.trimf(avg_rate_change.universe, [-0.3, -0.15, -0.05])
		avg_rate_change['stabil'] = fuzz.trimf(avg_rate_change.universe, [-0.08, 0, 0.08])
		avg_rate_change['turun lambat'] = fuzz.trimf(avg_rate_change.universe, [0.05, 0.15, 0.3])
		avg_rate_change['turun cepat'] = fuzz.trapmf(avg_rate_change.universe, [0.2, 0.45, 1, 1])
		
		# TUNED: Rainfall - adjusted for better sensitivity
		rainfall_norm['tidak_hujan'] = fuzz.trapmf(rainfall_norm.universe, [0, 0, 0.03, 0.06])
		rainfall_norm['ringan'] = fuzz.trimf(rainfall_norm.universe, [0.03, 0.15, 0.25])
		rainfall_norm['sedang'] = fuzz.trimf(rainfall_norm.universe, [0.18, 0.35, 0.5])
		rainfall_norm['lebat'] = fuzz.trimf(rainfall_norm.universe, [0.4, 0.65, 0.85])
		rainfall_norm['sangat_lebat'] = fuzz.trapmf(rainfall_norm.universe, [0.75, 0.9, 1.0, 1.0])
		
		# TUNED: Risk output - narrower medium, earlier high threshold
		flood_risk['low'] = fuzz.trimf(flood_risk.universe, [0, 0, 40])
		flood_risk['medium'] = fuzz.trimf(flood_risk.universe, [30, 55, 75])
		flood_risk['high'] = fuzz.trapmf(flood_risk.universe, [65, 80, 100, 100])
		
		rules = self._define_fuzzy_rules(water_level_norm, avg_rate_change, rainfall_norm, flood_risk)
		
		flood_ctrl = ctrl.ControlSystem(rules)
		return ctrl.ControlSystemSimulation(flood_ctrl)
	
	def _define_fuzzy_rules(self, water_level, rate_change, rainfall_norm, flood_risk):
		"""Define fuzzy logic rules - COMPREHENSIVE with 70+ rules for maximum accuracy"""
		rules = []
		
		print("🔧 Building comprehensive fuzzy rule system...")
		
		# ==================== BANJIR LEVEL RULES (15 rules) ====================
		# At flood level, risk varies based on trend and rainfall
		
		# Rising water at BANJIR = VERY HIGH RISK
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['naik sangat cepat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['naik cepat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['naik lambat'], flood_risk['high']))
		
		# Stable at BANJIR with heavy rain = HIGH RISK (could worsen)
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['stabil'] & rainfall_norm['sangat_lebat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['stabil'] & rainfall_norm['lebat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['stabil'] & rainfall_norm['sedang'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['stabil'], flood_risk['high']))
		
		# Falling at BANJIR but still at critical level
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['turun lambat'] & rainfall_norm['sangat_lebat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['turun lambat'] & rainfall_norm['lebat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['turun lambat'], flood_risk['medium']))
		
		# Fast falling at BANJIR = improving but still risky
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['turun cepat'] & rainfall_norm['sangat_lebat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['turun cepat'] & rainfall_norm['lebat'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['turun cepat'] & rainfall_norm['sedang'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['turun cepat'], flood_risk['medium']))
	
		# ==================== SIAGA LEVEL RULES (30 rules) ====================
		# Alert level - most critical for early warning
		
		# VERY FAST RISING at SIAGA = HIGH RISK regardless of rain
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik sangat cepat'] & rainfall_norm['sangat_lebat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik sangat cepat'] & rainfall_norm['lebat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik sangat cepat'] & rainfall_norm['sedang'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik sangat cepat'] & rainfall_norm['ringan'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik sangat cepat'] & rainfall_norm['tidak_hujan'], flood_risk['high']))
		
		# FAST RISING at SIAGA with rainfall = HIGH RISK
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik cepat'] & rainfall_norm['sangat_lebat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik cepat'] & rainfall_norm['lebat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik cepat'] & rainfall_norm['sedang'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik cepat'] & rainfall_norm['ringan'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik cepat'] & rainfall_norm['tidak_hujan'], flood_risk['medium']))
		
		# SLOW RISING at SIAGA
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik lambat'] & rainfall_norm['sangat_lebat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik lambat'] & rainfall_norm['lebat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik lambat'] & rainfall_norm['sedang'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik lambat'] & rainfall_norm['ringan'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik lambat'] & rainfall_norm['tidak_hujan'], flood_risk['medium']))
		
		# STABLE at SIAGA
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['stabil'] & rainfall_norm['sangat_lebat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['stabil'] & rainfall_norm['lebat'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['stabil'] & rainfall_norm['sedang'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['stabil'] & rainfall_norm['ringan'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['stabil'] & rainfall_norm['tidak_hujan'], flood_risk['medium']))
		
		# SLOW FALLING at SIAGA
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['turun lambat'] & rainfall_norm['sangat_lebat'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['turun lambat'] & rainfall_norm['lebat'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['turun lambat'] & rainfall_norm['sedang'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['turun lambat'], flood_risk['low']))
		
		# FAST FALLING at SIAGA = improving
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['turun cepat'] & rainfall_norm['sangat_lebat'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['turun cepat'] & rainfall_norm['lebat'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['turun cepat'], flood_risk['low']))
	
		# ==================== NORMAL LEVEL RULES (25 rules) ====================
		# Safe level but need to watch for rapid changes
		
		# VERY FAST RISING at NORMAL = potential flood incoming
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik sangat cepat'] & rainfall_norm['sangat_lebat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik sangat cepat'] & rainfall_norm['lebat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik sangat cepat'] & rainfall_norm['sedang'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik sangat cepat'] & rainfall_norm['ringan'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik sangat cepat'] & rainfall_norm['tidak_hujan'], flood_risk['medium']))
		
		# FAST RISING at NORMAL
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik cepat'] & rainfall_norm['sangat_lebat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik cepat'] & rainfall_norm['lebat'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik cepat'] & rainfall_norm['sedang'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik cepat'] & rainfall_norm['ringan'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik cepat'] & rainfall_norm['tidak_hujan'], flood_risk['low']))
		
		# SLOW RISING at NORMAL
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik lambat'] & rainfall_norm['sangat_lebat'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik lambat'] & rainfall_norm['lebat'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik lambat'] & rainfall_norm['sedang'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik lambat'], flood_risk['low']))
		
		# STABLE at NORMAL
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['stabil'] & rainfall_norm['sangat_lebat'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['stabil'] & rainfall_norm['lebat'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['stabil'], flood_risk['low']))
		
		# FALLING at NORMAL = all safe
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['turun lambat'] & rainfall_norm['sangat_lebat'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['turun lambat'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['turun cepat'] & rainfall_norm['sangat_lebat'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['turun cepat'], flood_risk['low']))
		
		print(f"✅ Fuzzy system initialized with {len(rules)} comprehensive rules\n")
		
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
	
	def normalize_avg_rate_change(self, avg_rate_cm_per_min, max_rate=2.0):
		"""Normalize average rate of change to -1 to +1 scale"""
		clipped_rate = np.clip(avg_rate_cm_per_min, -max_rate, max_rate)
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
		"""Calculate flood risk using 60-second average rate of change"""
		if self.calibration_height is None:
			raise ValueError("System not calibrated. Call calibrate() first.")
		
		self.add_distance_reading(current_distance, timestamp)
		avg_rate_cm_per_min = self.calculate_average_rate_change()
		
		# Normalize inputs
		water_level_normalized = self.normalize_water_level(current_distance)
		avg_rate_normalized = self.normalize_avg_rate_change(avg_rate_cm_per_min)
		rainfall_normalized = self.normalize_rainfall(current_rainfall_mm_per_hour)
		
		# Fuzzy logic computation
		self.fuzzy_system.input['water_level_norm'] = water_level_normalized
		self.fuzzy_system.input['avg_rate_change'] = avg_rate_normalized
		self.fuzzy_system.input['rainfall_norm'] = rainfall_normalized
		
		try:
			self.fuzzy_system.compute()
			risk_score = self.fuzzy_system.output['flood_risk']
		except:
			# Fallback calculation
			risk_score = water_level_normalized * 50 + abs(avg_rate_normalized) * 35 + \
						rainfall_normalized * 15
			risk_score = min(100, risk_score)
		
		# Determine warning level
		warning_level = self._determine_warning_level(risk_score, current_distance)
		notification_interval = self._get_notification_interval(warning_level, risk_score)
		rainfall_category = self._categorize_rainfall_hourly(current_rainfall_mm_per_hour)
		
		# Detect status recovery
		is_recovery = self._detect_recovery(warning_level)
		
		# Calculate time to flood
		time_to_flood_min, time_status = self.calculate_time_to_flood(current_distance, avg_rate_cm_per_min)
		
		# Save old warning level before updating
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
			if risk_score >= 75:
				return "BANJIR"
			else:
				return "SIAGA"
		else:
			if risk_score >= 50:
				return "SIAGA"
			else:
				return "NORMAL"
	
	def _determine_avg_rate_category(self, avg_rate_cm_per_min):
		"""Categorize average rate of change"""
		if avg_rate_cm_per_min < -0.28:  # ~-0.7 normalized
			return "Naik Sangat Cepat"
		elif avg_rate_cm_per_min < -0.14:  # ~-0.35 normalized
			return "Naik Cepat"
		elif avg_rate_cm_per_min < -0.05:
			return "Naik Lambat"
		elif avg_rate_cm_per_min < 0.05:
			return "Stabil"
		elif avg_rate_cm_per_min < 0.14:
			return "Turun Lambat"
		else:
			return "Turun Cepat"

	def _determine_flood_risk_category(self, risk_score):
		"""Categorize flood risk"""
		if risk_score >= 75:
			return "Tinggi"
		elif risk_score >= 50:
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
		"""Generate status message"""
		
		if is_recovery:
			if avg_rate > 0.1:
				msg = "✅ KONDISI SUDAH AMAN - Air telah surut dengan cepat. Tingkat air kembali normal."
			else:
				msg = "✅ KONDISI SUDAH AMAN - Tingkat air telah kembali normal. Situasi terkendali."
			
			if self.previous_warning_level == "BANJIR":
				msg += " Status banjir telah berakhir."
			elif self.previous_warning_level == "SIAGA":
				msg += " Status siaga telah dicabut."
			
			return msg
		
		if warning_level == "BANJIR":
			if avg_rate < -0.28:
				msg = "⚠️ STATUS BANJIR - Air naik sangat cepat! SEGERA EVAKUASI!"
			elif avg_rate < -0.14:
				msg = "⚠️ STATUS BANJIR - Air naik cepat! SEGERA EVAKUASI!"
			elif avg_rate < -0.05:
				msg = "⚠️ STATUS BANJIR - Air terus naik! Lakukan evakuasi sekarang!"
			else:
				msg = "⚠️ STATUS BANJIR - Permukaan air di level kritis!"
			
			if current_rain_category in ["Hujan Lebat", "Hujan Sangat Lebat"]:
				msg += f" {current_rain_category} sedang berlangsung."
		
		elif warning_level == "SIAGA":
			if time_to_flood and time_to_flood < 15:
				msg = f"⚠️ STATUS SIAGA - Air akan mencapai level banjir dalam ~{int(time_to_flood)} menit! Bersiap evakuasi!"
			elif avg_rate < -0.28:
				msg = "⚠️ STATUS SIAGA - Air naik sangat cepat! Bersiap untuk evakuasi!"
			elif avg_rate < -0.14:
				msg = "⚠️ STATUS SIAGA - Air naik cepat! Bersiap untuk evakuasi!"
			elif avg_rate < -0.067:
				msg = "⚠️ STATUS SIAGA - Air mendekati level banjir!"
			else:
				msg = "⚠️ STATUS SIAGA - Waspada terhadap kenaikan air!"
			
			if current_rain_category in ["Hujan Sedang", "Hujan Lebat", "Hujan Sangat Lebat"]:
				msg += f" Saat ini: {current_rain_category}."
		
		else:  # NORMAL
			if avg_rate < -0.14:
				msg = "✅ STATUS NORMAL - Namun air mulai naik cepat, pantau terus!"
			elif avg_rate < -0.05:
				msg = "✅ STATUS NORMAL - Namun air mulai naik, pantau terus!"
			else:
				msg = "✅ STATUS NORMAL - Kondisi aman"
			
			if current_rain_category in ["Hujan Lebat", "Hujan Sangat Lebat"]:
				msg += f" Saat ini: {current_rain_category}."
		
		return msg
	
	def visualize_system(self, style='clean'):
		"""Visualize membership functions"""
		if self.fuzzy_system is None:
			print("System not calibrated yet!")
			return
		
		if style == 'clean':
			self._visualize_clean()
		else:
			self._visualize_detailed()
	
	def _visualize_clean(self):
		"""Clean visualization"""
		fig = plt.figure(figsize=(16, 10))
		
		colors = {
			'normal': '#4CAF50',
			'siaga': '#FFC107',
			'banjir': '#F44336',
			'rising': '#E91E63',
			'stabil': '#9E9E9E',
			'falling': '#2196F3',
			'rain': ['#E0E0E0', '#81D4FA', '#FFE082', '#FF9800', '#D32F2F'],
			'risk': ['#4CAF50', '#FFC107', '#F44336']
		}
		
		# 1. Water Level
		ax1 = plt.subplot(2, 2, 1)
		x = np.arange(0, 1.01, 0.01)
		
		ax1.fill_between(x, 0, fuzz.trapmf(x, [0, 0, 0.35, 0.55]), 
						 color=colors['normal'], alpha=0.3, label='Normal')
		ax1.plot(x, fuzz.trapmf(x, [0, 0, 0.35, 0.55]), 
				color=colors['normal'], linewidth=3)
		
		ax1.fill_between(x, 0, fuzz.trimf(x, [0.45, 0.7, 0.85]), 
						 color=colors['siaga'], alpha=0.3, label='Siaga')
		ax1.plot(x, fuzz.trimf(x, [0.45, 0.7, 0.85]), 
				color=colors['siaga'], linewidth=3)
		
		ax1.fill_between(x, 0, fuzz.trapmf(x, [0.75, 0.9, 1.0, 1.0]), 
						 color=colors['banjir'], alpha=0.3, label='Banjir')
		ax1.plot(x, fuzz.trapmf(x, [0.75, 0.9, 1.0, 1.0]), 
				color=colors['banjir'], linewidth=3)
		
		ax1.set_xlim(-0.05, 1.05)
		ax1.set_ylim(0, 1.1)
		ax1.set_xlabel('Water Level (Normalized)', fontsize=11, weight='bold')
		ax1.set_ylabel('Membership', fontsize=11, weight='bold')
		ax1.set_title('Input: Water Level (TUNED)', fontsize=12, weight='bold')
		ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
		ax1.grid(True, alpha=0.2, linestyle='--')
		ax1.spines['top'].set_visible(False)
		ax1.spines['right'].set_visible(False)
		
		# 2. Average Rate of Change
		ax2 = plt.subplot(2, 2, 2)
		x = np.arange(-1, 1.01, 0.01)
		
		ax2.fill_between(x, 0, fuzz.trapmf(x, [-1, -1, -0.7, -0.45]), 
						 color='#C2185B', alpha=0.4)
		ax2.plot(x, fuzz.trapmf(x, [-1, -1, -0.7, -0.45]), 
				color='#C2185B', linewidth=3, label='Naik Sangat Cepat')
		
		ax2.fill_between(x, 0, fuzz.trimf(x, [-0.6, -0.35, -0.2]), 
						 color=colors['rising'], alpha=0.3)
		ax2.plot(x, fuzz.trimf(x, [-0.6, -0.35, -0.2]), 
				color=colors['rising'], linewidth=3, label='Naik Cepat')
		
		ax2.fill_between(x, 0, fuzz.trimf(x, [-0.3, -0.15, -0.05]), 
						 color=colors['rising'], alpha=0.2)
		ax2.plot(x, fuzz.trimf(x, [-0.3, -0.15, -0.05]), 
				color=colors['rising'], linewidth=2, linestyle='--', label='Naik Lambat')
		
		ax2.fill_between(x, 0, fuzz.trimf(x, [-0.08, 0, 0.08]), 
						 color=colors['stabil'], alpha=0.3)
		ax2.plot(x, fuzz.trimf(x, [-0.08, 0, 0.08]), 
				color=colors['stabil'], linewidth=3, label='Stabil')
		
		ax2.fill_between(x, 0, fuzz.trimf(x, [0.05, 0.15, 0.3]), 
						 color=colors['falling'], alpha=0.2)
		ax2.plot(x, fuzz.trimf(x, [0.05, 0.15, 0.3]), 
				color=colors['falling'], linewidth=2, linestyle='--', label='Turun Lambat')
		
		ax2.fill_between(x, 0, fuzz.trapmf(x, [0.2, 0.45, 1, 1]), 
						 color=colors['falling'], alpha=0.3)
		ax2.plot(x, fuzz.trapmf(x, [0.2, 0.45, 1, 1]), 
				color=colors['falling'], linewidth=3, label='Turun Cepat')
		
		guideline_norm = -0.0673611 / 0.4
		ax2.axvline(guideline_norm, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Guideline')
		
		ax2.set_xlim(-1.05, 1.05)
		ax2.set_ylim(0, 1.1)
		ax2.set_xlabel('Avg Rate of Change - 60s (Normalized)', fontsize=11, weight='bold')
		ax2.set_ylabel('Membership', fontsize=11, weight='bold')
		ax2.set_title('Input: Rate of Change (TUNED - More Granular)', fontsize=12, weight='bold')
		ax2.legend(loc='upper left', fontsize=9, framealpha=0.9)
		ax2.grid(True, alpha=0.2, linestyle='--')
		ax2.spines['top'].set_visible(False)
		ax2.spines['right'].set_visible(False)
		
		# 3. Rainfall
		ax3 = plt.subplot(2, 2, 3)
		x = np.arange(0, 1.01, 0.01)
		
		rain_labels = ['Tidak Hujan', 'Ringan', 'Sedang', 'Lebat', 'Sangat Lebat']
		rain_mfs = [
			fuzz.trapmf(x, [0, 0, 0.03, 0.06]),
			fuzz.trimf(x, [0.03, 0.15, 0.25]),
			fuzz.trimf(x, [0.18, 0.35, 0.5]),
			fuzz.trimf(x, [0.4, 0.65, 0.85]),
			fuzz.trapmf(x, [0.75, 0.9, 1.0, 1.0])
		]
		
		for i, (label, mf, color) in enumerate(zip(rain_labels, rain_mfs, colors['rain'])):
			ax3.fill_between(x, 0, mf, color=color, alpha=0.4)
			ax3.plot(x, mf, color=color, linewidth=3, label=label)
		
		ax3.set_xlim(-0.05, 1.05)
		ax3.set_ylim(0, 1.1)
		ax3.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
		ax3.set_xticklabels(['0', '5', '10', '15', '20', '25'])
		ax3.set_xlabel('Rainfall (mm/hour)', fontsize=11, weight='bold')
		ax3.set_ylabel('Membership', fontsize=11, weight='bold')
		ax3.set_title('Input: Rainfall Intensity (TUNED)', fontsize=12, weight='bold')
		ax3.legend(loc='upper right', fontsize=9, framealpha=0.9)
		ax3.grid(True, alpha=0.2, linestyle='--')
		ax3.spines['top'].set_visible(False)
		ax3.spines['right'].set_visible(False)
		
		# 4. Flood Risk Output
		ax4 = plt.subplot(2, 2, 4)
		x = np.arange(0, 101, 1)
		
		risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']
		risk_mfs = [
			fuzz.trimf(x, [0, 0, 40]),
			fuzz.trimf(x, [30, 55, 75]),
			fuzz.trapmf(x, [65, 80, 100, 100])
		]
		
		for label, mf, color in zip(risk_labels, risk_mfs, colors['risk']):
			ax4.fill_between(x, 0, mf, color=color, alpha=0.4)
			ax4.plot(x, mf, color=color, linewidth=3, label=label)
		
		ax4.axvline(70, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Warning Threshold (70%)')
		
		ax4.set_xlim(-5, 105)
		ax4.set_ylim(0, 1.1)
		ax4.set_xlabel('Flood Risk (%)', fontsize=11, weight='bold')
		ax4.set_ylabel('Membership', fontsize=11, weight='bold')
		ax4.set_title('Output: Flood Risk (TUNED - Earlier High Risk)', fontsize=12, weight='bold')
		ax4.legend(loc='upper left', fontsize=10, framealpha=0.9)
		ax4.grid(True, alpha=0.2, linestyle='--')
		ax4.spines['top'].set_visible(False)
		ax4.spines['right'].set_visible(False)
		
		plt.suptitle('TUNED Flood Warning System - Enhanced Sensitivity & Accuracy', 
					fontsize=14, weight='bold', y=0.98)
		plt.tight_layout()
		plt.show()
	
	def _visualize_detailed(self):
		"""Detailed technical visualization"""
		fig, axes = plt.subplots(nrows=4, figsize=(12, 14))
		
		# Water level
		ax = axes[0]
		x = np.arange(0, 1.01, 0.01)
		ax.plot(x, fuzz.trapmf(x, [0, 0, 0.35, 0.55]), 'b', linewidth=1.5, label='NORMAL')
		ax.plot(x, fuzz.trimf(x, [0.45, 0.7, 0.85]), 'y', linewidth=1.5, label='SIAGA')
		ax.plot(x, fuzz.trapmf(x, [0.75, 0.9, 1.0, 1.0]), 'r', linewidth=1.5, label='BANJIR')
		ax.set_title('Water Level (TUNED)')
		ax.set_ylabel('Membership')
		ax.legend()
		ax.grid(True, alpha=0.3)
		
		# Rate of change
		ax = axes[1]
		x = np.arange(-1, 1.01, 0.01)
		ax.plot(x, fuzz.trapmf(x, [-1, -1, -0.7, -0.45]), 'darkred', linewidth=1.5, label='Naik Sangat Cepat')
		ax.plot(x, fuzz.trimf(x, [-0.6, -0.35, -0.2]), 'r', linewidth=1.5, label='Naik Cepat')
		ax.plot(x, fuzz.trimf(x, [-0.3, -0.15, -0.05]), 'orange', linewidth=1.5, label='Naik Lambat')
		ax.plot(x, fuzz.trimf(x, [-0.08, 0, 0.08]), 'gray', linewidth=1.5, label='Stabil')
		ax.plot(x, fuzz.trimf(x, [0.05, 0.15, 0.3]), 'lightblue', linewidth=1.5, label='Turun Lambat')
		ax.plot(x, fuzz.trapmf(x, [0.2, 0.45, 1, 1]), 'blue', linewidth=1.5, label='Turun Cepat')
		
		guideline_norm = -0.0673611 / 0.4
		ax.axvline(guideline_norm, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Guideline')
		
		ax.set_title('Rate of Change (TUNED)')
		ax.set_ylabel('Membership')
		ax.legend()
		ax.grid(True, alpha=0.3)
		
		# Rainfall
		ax = axes[2]
		x = np.arange(0, 1.01, 0.01)
		ax.plot(x, fuzz.trapmf(x, [0, 0, 0.03, 0.06]), 'lightgray', linewidth=1.5, label='Tidak Hujan')
		ax.plot(x, fuzz.trimf(x, [0.03, 0.15, 0.25]), 'lightblue', linewidth=1.5, label='Ringan')
		ax.plot(x, fuzz.trimf(x, [0.18, 0.35, 0.5]), 'yellow', linewidth=1.5, label='Sedang')
		ax.plot(x, fuzz.trimf(x, [0.4, 0.65, 0.85]), 'orange', linewidth=1.5, label='Lebat')
		ax.plot(x, fuzz.trapmf(x, [0.75, 0.9, 1.0, 1.0]), 'red', linewidth=1.5, label='Sangat Lebat')
		ax.set_title('Rainfall (TUNED)')
		ax.set_ylabel('Membership')
		ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
		ax.set_xticklabels(['0', '5', '10', '15', '20', '25'])
		ax.set_xlabel('mm/hour')
		ax.legend()
		ax.grid(True, alpha=0.3)
		
		# Risk
		ax = axes[3]
		x = np.arange(0, 101, 1)
		ax.plot(x, fuzz.trimf(x, [0, 0, 40]), 'g', linewidth=1.5, label='Low Risk')
		ax.plot(x, fuzz.trimf(x, [30, 55, 75]), 'y', linewidth=1.5, label='Medium Risk')
		ax.plot(x, fuzz.trapmf(x, [65, 80, 100, 100]), 'r', linewidth=1.5, label='High Risk')
		ax.axvline(70, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Warning Threshold')
		ax.set_title('Flood Risk (TUNED - Earlier High Risk)')
		ax.set_ylabel('Membership')
		ax.legend()
		ax.grid(True, alpha=0.3)
		
		plt.tight_layout()
		plt.show()

if __name__ == "__main__":
	system = DynamicFuzzyFloodWarningSystem()
	system.calibrate(156.59)
	
	# Visualize the system
	# system.visualize_system(style='clean')
	
	print("\n=== Simulating 60-second readings ===")
	print("Simulating water rising scenario...\n")
	
	base_time = datetime.now()
	
	distances = [200.0, 199.9, 200.1, 199.8, 199.85, 199.79, 199.81, 199.78, 198.6, 198.2]
	
	for i, distance in enumerate(distances):
		timestamp = base_time + timedelta(seconds=i*6)
		
		result = system.calculate_risk(
			current_distance=distance,
			# current_rainfall_mm_per_hour=5,
			timestamp=timestamp
		)
		
		print(f"Reading {i+1} (t={i*6}s):")
		print(f"  Distance: {result['current_distance']} cm ({result['warning_level']})")
		print(f"  Avg Rate (60s): {result['avg_rate_change_cm_per_min']:.4f} cm/min ({result['avg_rate_category']})")
		print(f"  Rainfall: {result['current_rainfall']} mm/hour ({result['current_rainfall_category']})")
		print(f"  Risk: {result['risk_score']:.1f}% ({result['risk_category']})")
		print(f"  Should Send Warning: {result['should_send_warning']}")
		if result['time_to_flood_minutes']:
			print(f"  Time to Flood: {result['time_to_flood_minutes']} min ({result['time_to_flood_status']})")
		print(f"  Message: {result['status_message']}")
		print()
	
	print("\n=== Test: Water returning to normal ===")
	for i in range(10):
		timestamp = base_time + timedelta(seconds=(22+i)*6)
		distance = 160.5 + (i * 3)
		
		result = system.calculate_risk(
			current_distance=distance,
			current_rainfall_mm_per_hour=0.5,
			timestamp=timestamp
		)
		
		if i == 9:
			print(f"Final Reading:")
			print(f"  Distance: {result['current_distance']} cm")
			print(f"  Avg Rate (60s): {result['avg_rate_change_cm_per_min']:.4f} cm/min")
			print(f"  Previous: {result['previous_warning_level']} -> Current: {result['warning_level']}")
			print(f"  Risk: {result['risk_score']:.1f}%")
			print(f"  Should Send Warning: {result['should_send_warning']}")
			print(f"  Is Recovery: {result['is_recovery']}")
			print(f"  Message: {result['status_message']}")
			print(f"  Should send recovery notification: {result['should_send_recovery_notification']}")