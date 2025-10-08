import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patches as mpatches

class DynamicFuzzyFloodWarningSystem:
	def __init__(self):
		"""
		Initialize the fuzzy logic system with dynamic calibration capability
		Three warning levels: NORMAL, SIAGA, BANJIR
		Uses real-time measurements: water level + rate of change (cm/sec) + current rainfall
		"""
		self.calibration_height = None
		self.siaga_level = None
		self.banjir_level = None
		self.fuzzy_system = None
		self.previous_elevation = None
		self.previous_warning_level = None  # Track previous status for recovery detection
	
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
		"""Create the fuzzy logic control system"""
		
		water_level_norm = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'water_level_norm')
		rate_change_norm = ctrl.Antecedent(np.arange(-1, 1.01, 0.01), 'rate_change_norm')
		rainfall_norm = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'rainfall_norm')
		flood_risk = ctrl.Consequent(np.arange(0, 101, 1), 'flood_risk')
		
		# Water level membership functions
		water_level_norm['normal'] = fuzz.trapmf(water_level_norm.universe, [0, 0, 0.4, 0.6])
		water_level_norm['siaga'] = fuzz.trimf(water_level_norm.universe, [0.5, 0.75, 0.9])
		water_level_norm['banjir'] = fuzz.trapmf(water_level_norm.universe, [0.85, 0.95, 1.0, 1.0])
		
		# Rate of change membership functions (normalized from cm/sec)
		rate_change_norm['naik cepat'] = fuzz.trapmf(rate_change_norm.universe, [-1, -1, -0.4, -0.2])
		rate_change_norm['naik lambat'] = fuzz.trimf(rate_change_norm.universe, [-0.3, -0.15, 0])
		rate_change_norm['stabil'] = fuzz.trimf(rate_change_norm.universe, [-0.1, 0, 0.1])
		rate_change_norm['turun lambat'] = fuzz.trimf(rate_change_norm.universe, [0, 0.15, 0.3])
		rate_change_norm['turun cepat'] = fuzz.trapmf(rate_change_norm.universe, [0.2, 0.4, 1, 1])
		
		# Current rainfall (BMKG hourly categories - NORMALIZED)
		# 0-1 mm/h → 0-0.04, 0.5-5 mm/h → 0.02-0.2, 4-10 mm/h → 0.16-0.4
		# 9-20 mm/h → 0.36-0.8, 18-25 mm/h → 0.72-1.0
		rainfall_norm['tidak_hujan'] = fuzz.trapmf(rainfall_norm.universe, [0, 0, 0.02, 0.04])
		rainfall_norm['ringan'] = fuzz.trimf(rainfall_norm.universe, [0.02, 0.12, 0.2])
		rainfall_norm['sedang'] = fuzz.trimf(rainfall_norm.universe, [0.16, 0.3, 0.4])
		rainfall_norm['lebat'] = fuzz.trimf(rainfall_norm.universe, [0.36, 0.6, 0.8])
		rainfall_norm['sangat_lebat'] = fuzz.trapmf(rainfall_norm.universe, [0.72, 0.88, 1.0, 1.0])
		
		# Risk output
		flood_risk['low'] = fuzz.trimf(flood_risk.universe, [0, 0, 30])
		flood_risk['medium'] = fuzz.trimf(flood_risk.universe, [20, 50, 80])
		flood_risk['high'] = fuzz.trimf(flood_risk.universe, [70, 100, 100])
		
		rules = self._define_fuzzy_rules(water_level_norm, rate_change_norm, rainfall_norm, flood_risk)
		
		flood_ctrl = ctrl.ControlSystem(rules)
		return ctrl.ControlSystemSimulation(flood_ctrl)
	
	def _define_fuzzy_rules(self, water_level, rate_change, rainfall_norm, flood_risk):
		"""Define fuzzy logic rules"""
		rules = []
		
		# BANJIR level rules
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['naik cepat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['naik lambat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['stabil'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['turun lambat'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['banjir'] & rate_change['turun cepat'], flood_risk['medium']))
		
		# SIAGA level rules
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik cepat'] & rainfall_norm['sangat_lebat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik cepat'] & rainfall_norm['lebat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik cepat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik lambat'] & rainfall_norm['sangat_lebat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik lambat'] & rainfall_norm['lebat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['naik lambat'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['stabil'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['turun cepat'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['siaga'] & rate_change['turun lambat'], flood_risk['low']))
		
		# NORMAL level rules
		rules.append(ctrl.Rule(water_level['normal'] & rainfall_norm['sangat_lebat'] & rate_change['naik cepat'], flood_risk['high']))
		rules.append(ctrl.Rule(water_level['normal'] & rainfall_norm['sangat_lebat'] & rate_change['naik lambat'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['normal'] & rainfall_norm['lebat'] & rate_change['naik cepat'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik cepat'], flood_risk['medium']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['naik lambat'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['stabil'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['turun lambat'], flood_risk['low']))
		rules.append(ctrl.Rule(water_level['normal'] & rate_change['turun cepat'], flood_risk['low']))
		
		return rules
	
	def normalize_water_level(self, current_distance):
		"""Normalize water level (0=safe, 1=flood)"""
		if self.calibration_height is None:
			raise ValueError("System not calibrated. Call calibrate() first.")
		
		if current_distance >= self.siaga_level: # 30cm siaga
			return 0.0
		elif current_distance <= self.banjir_level:
			return 1.0
		else:
			total_range = self.siaga_level - self.banjir_level
			distance_from_flood = current_distance - self.banjir_level 
			return 1.0 - (distance_from_flood / total_range)
	
	def normalize_rate_change(self, rate_cm_per_sec, max_rate=0.17):
		"""
		Normalize rate of change to -1 to +1 scale
		
		Parameters:
		rate_cm_per_sec: Rate in cm/sec (negative=rising, positive=falling)
		max_rate: Max expected rate (default 0.17 cm/sec ≈ 10 cm/min)
		"""
		clipped_rate = np.clip(rate_cm_per_sec, -max_rate, max_rate)
		return clipped_rate / max_rate
	
	def normalize_rainfall(self, rainfall_mm_per_hour, max_rainfall=25):
		"""
		Normalize rainfall to 0-1 scale
		
		Parameters:
		rainfall_mm_per_hour: Rainfall intensity in mm/hour
		max_rainfall: Maximum expected rainfall (default 25 mm/hour)
		"""
		return np.clip(rainfall_mm_per_hour / max_rainfall, 0, 1)
	
	def calculate_time_to_flood(self, current_distance, rate_cm_per_sec):
		"""
		Calculate estimated time until water reaches flood level
		
		Returns:
		- time_minutes: Minutes until flood (None if not applicable)
		- status: Description of time estimate
		"""
		if current_distance <= self.banjir_level:
			return None, "Sudah banjir"
		
		if rate_cm_per_sec >= 0:
			return None, "Air turun/stabil"
		
		# Calculate time: distance / rate
		distance_to_flood = current_distance - self.banjir_level
		time_seconds = distance_to_flood / abs(rate_cm_per_sec)
		time_minutes = time_seconds / 60
		
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
	
	def calculate_risk(self, current_distance, current_rainfall_mm_per_hour=0, 
					  water_elevation_change_cm_per_sec=None):
		"""
		Calculate flood risk
		
		Parameters:
		current_distance: Distance from sensor to water (cm)
		current_rainfall_mm_per_hour: Rainfall intensity (mm/hour)
		water_elevation_change_cm_per_sec: Rate of change (cm/sec)
		time_interval_minutes: Time since last reading (if rate not provided)
		"""
		if self.calibration_height is None:
			raise ValueError("System not calibrated. Call calibrate() first.")
		
		# Determine rate of change
		if water_elevation_change_cm_per_sec is not None:
			rate_of_change = water_elevation_change_cm_per_sec
		else:
			if self.previous_elevation is not None:
				rate_of_change = (current_distance - self.previous_elevation)
			else:
				rate_of_change = 0
		
		self.previous_elevation = current_distance
		
		# Normalize inputs
		water_level_normalized = self.normalize_water_level(current_distance)
		rate_change_normalized = self.normalize_rate_change(rate_of_change)
		rainfall_normalized = self.normalize_rainfall(current_rainfall_mm_per_hour)
		
		# Fuzzy logic computation
		self.fuzzy_system.input['water_level_norm'] = water_level_normalized
		self.fuzzy_system.input['rate_change_norm'] = rate_change_normalized
		self.fuzzy_system.input['rainfall_norm'] = rainfall_normalized
		
		try:
			self.fuzzy_system.compute()
			risk_score = self.fuzzy_system.output['flood_risk']
		except:
			risk_score = water_level_normalized * 60 + abs(rate_change_normalized) * 25 + \
						rainfall_normalized * 15
			risk_score = min(100, risk_score)
		
		# Determine warning level
		warning_level = self._determine_warning_level(risk_score, current_distance)
		notification_interval = self._get_notification_interval(warning_level)
		rainfall_category = self._categorize_rainfall_hourly(current_rainfall_mm_per_hour)
		
		# Detect status recovery (SIAGA/BANJIR -> NORMAL)
		is_recovery = self._detect_recovery(warning_level)
		
		# Calculate time to flood
		time_to_flood_min, time_status = self.calculate_time_to_flood(current_distance, rate_of_change)
		
		# Save old warning level before updating
		old_warning_level = self.previous_warning_level
		
		# Update previous warning level for next iteration
		self.previous_warning_level = warning_level
		
		return {
			'current_distance': current_distance,
			'water_depth_from_ground': self.calibration_height - current_distance,
			'rate_of_change_cm_per_sec': rate_of_change,
			'rate_of_change_cm_per_min': rate_of_change * 60,
			'water_level_normalized': water_level_normalized,
			'rate_change_normalized': rate_change_normalized,
			'current_rainfall': current_rainfall_mm_per_hour,
			'rainfall_normalized': rainfall_normalized,
			'current_rainfall_category': rainfall_category,
			'risk_score': risk_score,
			'warning_level': warning_level,
			'previous_warning_level': old_warning_level,
			'is_recovery': is_recovery,
			'notification_interval': notification_interval,
			'should_send_warning': self._should_send_warning(warning_level, current_distance),
			'should_send_recovery_notification': is_recovery,
			'time_to_flood_minutes': time_to_flood_min,
			'time_to_flood_status': time_status,
			'status_message': self._get_status_message(warning_level, rate_of_change, 
													   rainfall_category, time_to_flood_min, is_recovery),
			'thresholds': {
				'banjir': self.banjir_level,
				'siaga': self.siaga_level,
				'calibration': self.calibration_height
			}
		}
	
	def _detect_recovery(self, current_warning_level):
		"""
		Detect if status has recovered from SIAGA/BANJIR to NORMAL
		
		Returns:
		- True if recovery detected (previous was SIAGA/BANJIR and current is NORMAL)
		- False otherwise
		"""
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
	
	def _should_send_warning(self, warning_level, current_distance):
		"""Determine if warning should be sent"""
		if warning_level == "NORMAL":
			return False
		elif warning_level == "BANJIR":
			return current_distance > self.banjir_level - 5
		else:
			return True
	
	def _get_notification_interval(self, warning_level):
		"""Get notification interval"""
		intervals = {
			"BANJIR": 5,
			"SIAGA": 10,
			"NORMAL": None
		}
		return intervals.get(warning_level)
	
	def _get_status_message(self, warning_level, rate_change, current_rain_category, time_to_flood, is_recovery=False):
		"""Generate status message"""
		
		# Recovery message (SIAGA/BANJIR -> NORMAL)
		if is_recovery:
			if rate_change > 0.05:  # Turun cepat
				msg = "✅ KONDISI SUDAH AMAN - Air telah surut dengan cepat. Tingkat air kembali normal."
			else:
				msg = "✅ KONDISI SUDAH AMAN - Tingkat air telah kembali normal. Situasi terkendali."
			
			if self.previous_warning_level == "BANJIR":
				msg += " Status banjir telah berakhir."
			elif self.previous_warning_level == "SIAGA":
				msg += " Status siaga telah dicabut."
			
			return msg
		
		# Regular status messages
		if warning_level == "BANJIR":
			if rate_change < -0.083:  # < -5 cm/min
				msg = "STATUS BANJIR - Air naik sangat cepat! SEGERA EVAKUASI!"
			elif rate_change < -0.033:  # < -2 cm/min
				msg = "STATUS BANJIR - Air terus naik! Lakukan evakuasi sekarang!"
			else:
				msg = "STATUS BANJIR - Permukaan air di level kritis!"
			
			if current_rain_category in ["Hujan Lebat", "Hujan Sangat Lebat"]:
				msg += f" {current_rain_category} sedang berlangsung."
		
		elif warning_level == "SIAGA":
			if time_to_flood and time_to_flood < 15:
				msg = f"STATUS SIAGA - Air akan mencapai level banjir dalam ~{int(time_to_flood)} menit! Bersiap evakuasi!"
			elif rate_change < -0.05:  # < -3 cm/min
				msg = "STATUS SIAGA - Air naik cepat! Bersiap untuk evakuasi!"
			elif rate_change < -0.017:  # < -1 cm/min
				msg = "STATUS SIAGA - Air mendekati level banjir!"
			else:
				msg = "STATUS SIAGA - Waspada terhadap kenaikan air!"
			
			if current_rain_category in ["Hujan Sedang", "Hujan Lebat", "Hujan Sangat Lebat"]:
				msg += f" Saat ini: {current_rain_category}."
		
		else:  # NORMAL
			if rate_change < -0.033:  # < -2 cm/min
				msg = "STATUS NORMAL - Namun air mulai naik, pantau terus!"
			else:
				msg = "STATUS NORMAL - Kondisi aman"
			
			if current_rain_category in ["Hujan Lebat", "Hujan Sangat Lebat"]:
				msg += f" Saat ini: {current_rain_category}."
		
		return msg
	
	def visualize_system(self, style='tree'):
		"""
		Visualize membership functions
		
		Parameters:
		style: 'tree' for tree diagram, 'clean' for clean plots, 'detailed' for technical plots
		"""
		if self.fuzzy_system is None:
			print("System not calibrated yet!")
			return
		
		if style == 'tree':
			self._visualize_tree()
		elif style == 'clean':
			self._visualize_clean()
		else:
			self._visualize_detailed()
	
	def _visualize_tree(self):
		"""Create a tree diagram visualization like the example"""
		fig, ax = plt.subplots(1, 1, figsize=(12, 8))
		ax.set_xlim(0, 10)
		ax.set_ylim(0, 10)
		ax.axis('off')
		
		# Main question box
		question_box = Rectangle((0.5, 4.5), 2.5, 1, 
								linewidth=2, edgecolor='green', 
								facecolor='lightgreen')
		ax.add_patch(question_box)
		ax.text(1.75, 5, 'Water Level?', fontsize=14, 
			   ha='center', va='center', color='green', weight='bold')
		
		# Branch lines
		# Top branch
		ax.plot([3, 4, 4, 5.5], [5, 5, 7.5, 7.5], 'green', linewidth=2)
		# Middle branch  
		ax.plot([3, 4, 4, 5.5], [5, 5, 5, 5], 'green', linewidth=2)
		# Bottom branch
		ax.plot([3, 4, 4, 5.5], [5, 5, 2.5, 2.5], 'green', linewidth=2)
		
		# Answer boxes and text
		# BANJIR (top)
		ax.text(5.5, 7.5, 'BANJIR (0.9)', fontsize=13, 
			   va='center', color='red')
		
		# SIAGA (middle)
		ax.text(5.5, 5, 'SIAGA (0.6)', fontsize=13, 
			   va='center', color='orange')
		
		# NORMAL (bottom)
		ax.text(5.5, 2.5, 'NORMAL (0.3)', fontsize=13, 
			   va='center', color='green')
		
		# Fuzzy Logic label
		ax.text(8.5, 5, 'Fuzzy\nLogic', fontsize=14, 
			   ha='center', va='center', color='black', weight='bold')
		
		# Add title
		ax.text(5, 9, 'Flood Warning System - Fuzzy Logic Tree', 
			   fontsize=16, ha='center', weight='bold')
		
		# Add subtitle with more detail
		ax.text(5, 0.5, 'Membership values shown in parentheses', 
			   fontsize=10, ha='center', style='italic', color='gray')
		
		plt.tight_layout()
		plt.show()
		
		# Create a second tree for Rate of Change
		fig2, ax2 = plt.subplots(1, 1, figsize=(14, 10))
		ax2.set_xlim(0, 12)
		ax2.set_ylim(0, 12)
		ax2.axis('off')
		
		# Main question box
		question_box = Rectangle((0.5, 5.5), 3, 1, 
								linewidth=2, edgecolor='blue', 
								facecolor='lightblue')
		ax2.add_patch(question_box)
		ax2.text(2, 6, 'Rate of Change?', fontsize=14, 
				ha='center', va='center', color='blue', weight='bold')
		
		# Branch lines - 5 branches for rate of change
		positions = [9.5, 8, 6, 4, 2.5]
		labels = [
			('Naik Cepat', 'red', '(0.8)'),
			('Naik Lambat', 'orange', '(0.5)'),
			('Stabil', 'gray', '(0.2)'),
			('Turun Lambat', 'lightblue', '(0.4)'),
			('Turun Cepat', 'blue', '(0.7)')
		]
		
		for pos, (label, color, value) in zip(positions, labels):
			ax2.plot([3.5, 4.5, 4.5, 6], [6, 6, pos, pos], 'blue', linewidth=2)
			ax2.text(6, pos, f'{label} {value}', fontsize=12, 
					va='center', color=color)
		
		# Fuzzy Logic label
		ax2.text(10, 6, 'Fuzzy\nLogic', fontsize=14, 
				ha='center', va='center', color='black', weight='bold')
		
		# Add title
		ax2.text(6, 11, 'Rate of Change - Fuzzy Logic Tree', 
				fontsize=16, ha='center', weight='bold')
		
		plt.tight_layout()
		plt.show()
		
		# Create a third tree for Rainfall
		fig3, ax3 = plt.subplots(1, 1, figsize=(14, 10))
		ax3.set_xlim(0, 12)
		ax3.set_ylim(0, 12)
		ax3.axis('off')
		
		# Main question box
		question_box = Rectangle((0.5, 5.5), 2.5, 1, 
								linewidth=2, edgecolor='purple', 
								facecolor='lavender')
		ax3.add_patch(question_box)
		ax3.text(1.75, 6, 'Rainfall?', fontsize=14, 
				ha='center', va='center', color='purple', weight='bold')
		
		# Branch lines - 5 branches for rainfall
		rain_positions = [9.5, 8, 6, 4, 2.5]
		rain_labels = [
			('Sangat Lebat', 'darkred', '(0.9)'),
			('Lebat', 'red', '(0.7)'),
			('Sedang', 'orange', '(0.5)'),
			('Ringan', 'lightblue', '(0.3)'),
			('Tidak Hujan', 'lightgray', '(0.1)')
		]
		
		for pos, (label, color, value) in zip(rain_positions, rain_labels):
			ax3.plot([3, 4, 4, 5.5], [6, 6, pos, pos], 'purple', linewidth=2)
			ax3.text(5.5, pos, f'{label} {value}', fontsize=12, 
					va='center', color=color)
		
		# Fuzzy Logic label
		ax3.text(10, 6, 'Fuzzy\nLogic', fontsize=14, 
				ha='center', va='center', color='black', weight='bold')
		
		# Add title
		ax3.text(6, 11, 'Rainfall Intensity - Fuzzy Logic Tree', 
				fontsize=16, ha='center', weight='bold')
		
		ax3.text(6, 0.5, 'Membership values shown in parentheses', 
				fontsize=10, ha='center', style='italic', color='gray')
		
		plt.tight_layout()
		plt.show()
	
	def _visualize_clean(self):
		"""Clean, diagram-like visualization"""
		fig = plt.figure(figsize=(16, 10))
		
		# Define colors
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
		
		ax1.fill_between(x, 0, fuzz.trapmf(x, [0, 0, 0.4, 0.6]), 
						 color=colors['normal'], alpha=0.3, label='Normal')
		ax1.plot(x, fuzz.trapmf(x, [0, 0, 0.4, 0.6]), 
				color=colors['normal'], linewidth=3)
		
		ax1.fill_between(x, 0, fuzz.trimf(x, [0.5, 0.75, 0.9]), 
						 color=colors['siaga'], alpha=0.3, label='Siaga')
		ax1.plot(x, fuzz.trimf(x, [0.5, 0.75, 0.9]), 
				color=colors['siaga'], linewidth=3)
		
		ax1.fill_between(x, 0, fuzz.trapmf(x, [0.85, 0.95, 1.0, 1.0]), 
						 color=colors['banjir'], alpha=0.3, label='Banjir')
		ax1.plot(x, fuzz.trapmf(x, [0.85, 0.95, 1.0, 1.0]), 
				color=colors['banjir'], linewidth=3)
		
		# Add input box
		bbox = FancyBboxPatch((-0.35, 0.45), 0.25, 0.1, 
							 boxstyle="round,pad=0.01", 
							 edgecolor='green', facecolor='lightgreen', 
							 linewidth=2, transform=ax1.transAxes)
		ax1.add_patch(bbox)
		
		ax1.set_xlim(-0.05, 1.05)
		ax1.set_ylim(0, 1.1)
		ax1.set_xlabel('Skala Elevasi Air (Normalized)', fontsize=11, weight='bold')
		ax1.set_ylabel('Membership', fontsize=11, weight='bold')
		ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
		ax1.grid(True, alpha=0.2, linestyle='--')
		ax1.spines['top'].set_visible(False)
		ax1.spines['right'].set_visible(False)
		
		# 2. Rate of Change
		ax2 = plt.subplot(2, 2, 2)
		x = np.arange(-1, 1.01, 0.01)
		
		# Rising
		ax2.fill_between(x, 0, fuzz.trapmf(x, [-1, -1, -0.4, -0.2]), 
						 color=colors['rising'], alpha=0.3)
		ax2.plot(x, fuzz.trapmf(x, [-1, -1, -0.4, -0.2]), 
				color=colors['rising'], linewidth=3, label='Naik Cepat')
		
		ax2.fill_between(x, 0, fuzz.trimf(x, [-0.3, -0.15, 0]), 
						 color=colors['rising'], alpha=0.2)
		ax2.plot(x, fuzz.trimf(x, [-0.3, -0.15, 0]), 
				color=colors['rising'], linewidth=2, linestyle='--', label='Naik Lambat')
		
		# stabil
		ax2.fill_between(x, 0, fuzz.trimf(x, [-0.1, 0, 0.1]), 
						 color=colors['stabil'], alpha=0.3)
		ax2.plot(x, fuzz.trimf(x, [-0.1, 0, 0.1]), 
				color=colors['stabil'], linewidth=3, label='Stabil')
		
		# Falling
		ax2.fill_between(x, 0, fuzz.trimf(x, [0, 0.15, 0.3]), 
						 color=colors['falling'], alpha=0.2)
		ax2.plot(x, fuzz.trimf(x, [0, 0.15, 0.3]), 
				color=colors['falling'], linewidth=2, linestyle='--', label='Turun Lambat')
		
		ax2.fill_between(x, 0, fuzz.trapmf(x, [0.2, 0.4, 1, 1]), 
						 color=colors['falling'], alpha=0.3)
		ax2.plot(x, fuzz.trapmf(x, [0.2, 0.4, 1, 1]), 
				color=colors['falling'], linewidth=3, label='Turun Cepat')
		
		# Add input box
		bbox = FancyBboxPatch((-0.35, 0.45), 0.25, 0.1, 
							 boxstyle="round,pad=0.01", 
							 edgecolor='green', facecolor='lightgreen', 
							 linewidth=2, transform=ax2.transAxes)
		ax2.add_patch(bbox)
		
		ax2.set_xlim(-1.05, 1.05)
		ax2.set_ylim(0, 1.1)
		ax2.set_xlabel('Rate of Change (Normalized)', fontsize=11, weight='bold')
		ax2.set_ylabel('Membership', fontsize=11, weight='bold')
		ax2.legend(loc='upper left', fontsize=9, framealpha=0.9)
		ax2.grid(True, alpha=0.2, linestyle='--')
		ax2.spines['top'].set_visible(False)
		ax2.spines['right'].set_visible(False)
		
		# 3. Rainfall (NORMALIZED)
		ax3 = plt.subplot(2, 2, 3)
		x = np.arange(0, 1.01, 0.01)
		
		rain_labels = ['Tidak Hujan', 'Ringan', 'Sedang', 'Lebat', 'Sangat Lebat']
		rain_mfs = [
			fuzz.trapmf(x, [0, 0, 0.02, 0.04]),
			fuzz.trimf(x, [0.02, 0.12, 0.2]),
			fuzz.trimf(x, [0.16, 0.3, 0.4]),
			fuzz.trimf(x, [0.36, 0.6, 0.8]),
			fuzz.trapmf(x, [0.72, 0.88, 1.0, 1.0])
		]
		
		for i, (label, mf, color) in enumerate(zip(rain_labels, rain_mfs, colors['rain'])):
			ax3.fill_between(x, 0, mf, color=color, alpha=0.4)
			ax3.plot(x, mf, color=color, linewidth=3, label=label)
		
		# Add input box
		bbox = FancyBboxPatch((-0.35, 0.45), 0.25, 0.1, 
							 boxstyle="round,pad=0.01", 
							 edgecolor='green', facecolor='lightgreen', 
							 linewidth=2, transform=ax3.transAxes)
		ax3.add_patch(bbox)
		
		# Custom x-axis labels to show mm/hour equivalents
		ax3.set_xlim(-0.05, 1.05)
		ax3.set_ylim(0, 1.1)
		ax3.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
		ax3.set_xticklabels(['0', '5', '10', '15', '20', '25'])
		ax3.set_xlabel('Rainfall (mm/hour)', fontsize=11, weight='bold')
		ax3.set_ylabel('Membership', fontsize=11, weight='bold')
		ax3.legend(loc='upper right', fontsize=9, framealpha=0.9)
		ax3.grid(True, alpha=0.2, linestyle='--')
		ax3.spines['top'].set_visible(False)
		ax3.spines['right'].set_visible(False)
		
		# 4. Flood Risk Output
		ax4 = plt.subplot(2, 2, 4)
		x = np.arange(0, 101, 1)
		
		risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']
		risk_mfs = [
			fuzz.trimf(x, [0, 0, 30]),
			fuzz.trimf(x, [20, 50, 80]),
			fuzz.trimf(x, [70, 100, 100])
		]
		
		for label, mf, color in zip(risk_labels, risk_mfs, colors['risk']):
			ax4.fill_between(x, 0, mf, color=color, alpha=0.4)
			ax4.plot(x, mf, color=color, linewidth=3, label=label)
				
		# Add output box
		bbox = FancyBboxPatch((1.05, 0.45), 0.25, 0.1, 
							 boxstyle="round,pad=0.01", 
							 edgecolor='red', facecolor='#ffcccc', 
							 linewidth=2, transform=ax4.transAxes)
		ax4.add_patch(bbox)
		
		ax4.set_xlim(-5, 105)
		ax4.set_ylim(0, 1.1)
		ax4.set_xlabel('Flood Risk (%)', fontsize=11, weight='bold')
		ax4.set_ylabel('Membership', fontsize=11, weight='bold')
		ax4.legend(loc='upper left', fontsize=10, framealpha=0.9)
		ax4.grid(True, alpha=0.2, linestyle='--')
		ax4.spines['top'].set_visible(False)
		ax4.spines['right'].set_visible(False)
		
		plt.suptitle('Flood Warning System - Fuzzy Logic (All Inputs Normalized)', 
					fontsize=14, weight='bold', y=0.98)
		plt.tight_layout()
		plt.show()
	
	def _visualize_detailed(self):
		"""Original detailed technical visualization"""
		fig, axes = plt.subplots(nrows=4, figsize=(12, 14))
		
		# Water level
		ax = axes[0]
		x = np.arange(0, 1.01, 0.01)
		ax.plot(x, fuzz.trapmf(x, [0, 0, 0.4, 0.6]), 'b', linewidth=1.5, label='NORMAL')
		ax.plot(x, fuzz.trimf(x, [0.5, 0.75, 0.9]), 'y', linewidth=1.5, label='SIAGA')
		ax.plot(x, fuzz.trapmf(x, [0.85, 0.95, 1.0, 1.0]), 'r', linewidth=1.5, label='BANJIR')
		ax.set_title('Water Level (Normalized)')
		ax.set_ylabel('Membership')
		ax.legend()
		ax.grid(True, alpha=0.3)
		
		# Rate of change
		ax = axes[1]
		x = np.arange(-1, 1.01, 0.01)
		ax.plot(x, fuzz.trapmf(x, [-1, -1, -0.4, -0.2]), 'r', linewidth=1.5, label='Naik Cepat')
		ax.plot(x, fuzz.trimf(x, [-0.3, -0.15, 0]), 'orange', linewidth=1.5, label='Naik Lambat')
		ax.plot(x, fuzz.trimf(x, [-0.1, 0, 0.1]), 'gray', linewidth=1.5, label='Stabil')
		ax.plot(x, fuzz.trimf(x, [0, 0.15, 0.3]), 'lightblue', linewidth=1.5, label='Turun Lambat')
		ax.plot(x, fuzz.trapmf(x, [0.2, 0.4, 1, 1]), 'blue', linewidth=1.5, label='Turun Cepat')
		ax.set_title('Rate of Change (Normalized from cm/sec)')
		ax.set_ylabel('Membership')
		ax.legend()
		ax.grid(True, alpha=0.3)
		
		# Rainfall (NORMALIZED)
		ax = axes[2]
		x = np.arange(0, 1.01, 0.01)
		ax.plot(x, fuzz.trapmf(x, [0, 0, 0.02, 0.04]), 'lightgray', linewidth=1.5, label='Tidak Hujan')
		ax.plot(x, fuzz.trimf(x, [0.02, 0.12, 0.2]), 'lightblue', linewidth=1.5, label='Ringan')
		ax.plot(x, fuzz.trimf(x, [0.16, 0.3, 0.4]), 'yellow', linewidth=1.5, label='Sedang')
		ax.plot(x, fuzz.trimf(x, [0.36, 0.6, 0.8]), 'orange', linewidth=1.5, label='Lebat')
		ax.plot(x, fuzz.trapmf(x, [0.72, 0.88, 1.0, 1.0]), 'red', linewidth=1.5, label='Sangat Lebat')
		ax.set_title('Rainfall Intensity (Normalized)')
		ax.set_ylabel('Membership')
		ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
		ax.set_xticklabels(['0', '5', '10', '15', '20', '25'])
		ax.set_xlabel('mm/hour')
		ax.legend()
		ax.grid(True, alpha=0.3)
		
		# Risk
		ax = axes[3]
		x = np.arange(0, 101, 1)
		ax.plot(x, fuzz.trimf(x, [0, 0, 30]), 'g', linewidth=1.5, label='Low Risk')
		ax.plot(x, fuzz.trimf(x, [20, 50, 80]), 'y', linewidth=1.5, label='Medium Risk')
		ax.plot(x, fuzz.trimf(x, [70, 100, 100]), 'r', linewidth=1.5, label='High Risk')
		ax.set_title('Flood Risk Output')
		ax.set_ylabel('Membership')
		ax.legend()
		ax.grid(True, alpha=0.3)
		
		plt.tight_layout()
		plt.show()

if __name__ == "__main__":
	system = DynamicFuzzyFloodWarningSystem()
	system.calibrate(156.59)
	
	# Visualize with tree style (new default)
	# system.visualize_system(style='tree')
	
	# You can still use the other visualization styles:
	# system.visualize_system(style='clean')  # For the clean membership function plots
	# system.visualize_system(style='detailed')  # For the technical plots
	
	print("=== Test 1: Rising water (SIAGA) ===")
	result = system.calculate_risk(
		current_distance=160.5,
		current_rainfall_mm_per_hour=12.5,
	)
	print(f"Current Distance: {result['current_distance']} cm")
	print(f"Current Rate of Change: {result['rate_of_change_cm_per_sec']} cm/sec")
	print(f"Current Rainfall: {result['current_rainfall']} mm/hour")
	print(f"Current Water Level (normalized): {result['water_level_normalized']:.3f}")
	print(f"Rate of Change (normalized): {result['rate_of_change_cm_per_sec']} cm/sec")
	print(f"Rainfall (normalized): {result['rainfall_normalized']:.3f}")
	print(f"Warning: {result['warning_level']}")
	print(f"Risk: {result['risk_score']:.1f}%")
	print(f"Message: {result['status_message']}")
	print(f"Should send recovery notification: {result['should_send_recovery_notification']}")
	print()
	
	print("=== Test 2: Water returning to normal ===")
	result = system.calculate_risk(
		current_distance=165.0,
		current_rainfall_mm_per_hour=0.5,
	)

	print(f"Current Distance: {result['current_distance']} cm")
	print(f"Current Rate of Change: {result['rate_of_change_cm_per_sec']} cm/sec")
	print(f"Current Rainfall: {result['current_rainfall']} mm/hour")
	print(f"Current Water Level (normalized): {result['water_level_normalized']:.3f}")
	print(f"Rate of Change (normalized): {result['rate_of_change_cm_per_sec']} cm/sec")
	print(f"Rainfall (normalized): {result['rainfall_normalized']:.3f}")
	print(f"Previous: {result['previous_warning_level']} -> Current: {result['warning_level']}")
	print(f"Risk: {result['risk_score']:.1f}%")
	print(f"Is Recovery: {result['is_recovery']}")
	print(f"Message: {result['status_message']}")
	print(f"Should send recovery notification: {result['should_send_recovery_notification']}")