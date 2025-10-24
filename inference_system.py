import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Import the flood warning system from main2
from main2 import DynamicFuzzyFloodWarningSystem

plt.rcParams['font.family'] = 'DejaVu Sans'

class DynamicFuzzyVisualizer:
	def __init__(self, ground_distance=100, siaga_level=130, banjir_level=100):
		"""
		Initialize fuzzy system visualizer by extracting configuration from main2.py
		
		Args:
			ground_distance: Calibration ground distance in cm
			siaga_level: Siaga alert level in cm
			banjir_level: Flood level in cm
		"""
		# Create an instance of the flood warning system
		self.flood_system = DynamicFuzzyFloodWarningSystem(reading_interval_seconds=1)
		self.flood_system.calibrate(ground_distance, siaga_level, banjir_level)
		
		# Store calibration parameters
		self.ground_distance = ground_distance
		self.siaga_level = siaga_level
		self.banjir_level = banjir_level
		
		# Extract fuzzy system components
		self.extract_fuzzy_system()
		self.create_color_schemes()
	
	def extract_fuzzy_system(self):
		"""Extract membership functions and universes from main2.py fuzzy system"""
		# Get the control system from main2
		ctrl_system = self.flood_system.fuzzy_system.ctrl
		
		# Extract antecedents and consequent
		water_level_ctrl = None
		rate_change_ctrl = None
		rainfall_ctrl = None
		flood_risk_ctrl = None
		
		for var in ctrl_system.antecedents:
			if var.label == 'water_level':
				water_level_ctrl = var
			elif var.label == 'avg_rate_change':
				rate_change_ctrl = var
			elif var.label == 'rainfall':
				rainfall_ctrl = var
		
		for var in ctrl_system.consequents:
			if var.label == 'flood_risk':
				flood_risk_ctrl = var
		
		# Extract universes
		self.water_level_universe = water_level_ctrl.universe
		self.rate_change_universe = rate_change_ctrl.universe
		self.rainfall_universe = rainfall_ctrl.universe
		self.flood_risk_universe = flood_risk_ctrl.universe
		
		# Extract membership functions
		self.water_mf = {}
		for term_name, term in water_level_ctrl.terms.items():
			self.water_mf[term_name] = term.mf
		
		self.rate_mf = {}
		for term_name, term in rate_change_ctrl.terms.items():
			self.rate_mf[term_name] = term.mf
		
		self.rain_mf = {}
		for term_name, term in rainfall_ctrl.terms.items():
			self.rain_mf[term_name] = term.mf
		
		self.risk_mf = {}
		for term_name, term in flood_risk_ctrl.terms.items():
			self.risk_mf[term_name] = term.mf
		
		# Store the rules for reference (convert generator to list)
		self.rules = list(ctrl_system.rules)
		
		print(f"\n{'='*60}")
		print("EXTRACTED FUZZY SYSTEM CONFIGURATION FROM main2.py")
		print(f"{'='*60}")
		print(f"Water Level Terms: {list(self.water_mf.keys())}")
		print(f"Rate Change Terms: {list(self.rate_mf.keys())}")
		print(f"Rainfall Terms: {list(self.rain_mf.keys())}")
		print(f"Flood Risk Terms: {list(self.risk_mf.keys())}")
		print(f"Total Rules: {len(self.rules)}")
		print(f"{'='*60}\n")
	
	def create_color_schemes(self):
		"""Define color schemes for visualization"""
		# Water level colors
		self.color_water = {
			'normal': '#2E86AB',
			'siaga I': '#A23B72',
			'siaga II': '#F18F01',
			'banjir': '#C73E1D'
		}
		
		# Rate of change colors
		self.color_rate = {
			'turun sangat cepat': '#0A2463',
			'turun cepat': '#1B4965',
			'turun lambat': '#5FA8D3',
			'stabil': '#62B6CB',
			'naik lambat': '#CAE9FF',
			'naik cepat': '#FFB627',
			'naik sangat cepat': '#FF6B35',
			'naik ekstrem': '#C1121F'
		}
		
		# Rainfall colors
		self.color_rain = {
			'tidak_hujan': '#B8E0D2',
			'ringan': '#95D5B2',
			'sedang': '#74C69D',
			'lebat': '#52B788',
			'sangat_lebat': '#40916C'
		}
		
		# Risk colors
		self.color_risk = {
			'low': '#06D6A0',
			'medium': '#FFD166',
			'high': '#EF476F',
			'critical': '#9D0208'
		}
	
	def calculate_memberships(self, universe, mf_dict, value):
		"""Calculate membership degrees for all categories"""
		memberships = {}
		for term, mf in mf_dict.items():
			memberships[term] = fuzz.interp_membership(universe, mf, value)
		return memberships
	
	def normalize_to_universe(self, value, universe_min, universe_max):
		"""Helper to normalize values if needed"""
		return np.clip(value, universe_min, universe_max)
	
	def visualize_all_steps(self, water_distance_cm, rate_change_cm_per_min, rainfall_mm_per_hour):
		"""
		Generate all visualization steps
		
		Args:
			water_distance_cm: Water level distance from sensor in cm
			rate_change_cm_per_min: Rate of change in cm per minute
			rainfall_mm_per_hour: Rainfall intensity in mm/hour
		"""
		print("\n" + "=" * 60)
		print("FUZZY INFERENCE VISUALIZATION")
		print("=" * 60)
		print(f"\nInput Values:")
		print(f"  • Water Level Distance: {water_distance_cm} cm")
		print(f"  • Rate of Change: {rate_change_cm_per_min:.2f} cm/min")
		print(f"  • Rainfall: {rainfall_mm_per_hour} mm/hour\n")
		
		# Ensure values are within valid ranges
		water_val = self.normalize_to_universe(
			water_distance_cm, 
			self.water_level_universe[0], 
			self.water_level_universe[-1]
		)
		rate_val = self.normalize_to_universe(
			rate_change_cm_per_min,
			self.rate_change_universe[0],
			self.rate_change_universe[-1]
		)
		rain_val = self.normalize_to_universe(
			rainfall_mm_per_hour,
			self.rainfall_universe[0],
			self.rainfall_universe[-1]
		)
		
		# Step 1: Fuzzification
		print("=" * 60)
		print("STEP 1: FUZZIFICATION")
		print("=" * 60)
		print("Converting crisp inputs to fuzzy membership values...")
		fig = self.plot_fuzzification_clean(water_val, rate_val, rain_val)
		fig.savefig('step1_fuzzification.png', dpi=150, bbox_inches='tight')
		print("✓ Saved: step1_fuzzification.png\n")
		
		# Step 2: Implication (Rule Firing)
		print("=" * 60)
		print("STEP 2: IMPLICATION (Rule Evaluation)")
		print("=" * 60)
		print("Evaluating fuzzy rules and clipping output functions...")
		fig2 = self.plot_implication(water_val, rate_val, rain_val)
		fig2.savefig('step2_implication.png', dpi=150, bbox_inches='tight')
		print("✓ Saved: step2_implication.png\n")

		# Step 3: Aggregation & Defuzzification
		print("=" * 60)
		print("STEP 3: AGGREGATION & DEFUZZIFICATION")
		print("=" * 60)
		print("Combining all rule outputs and computing final crisp value...")
		fig3, defuzz_value = self.plot_aggregation(water_val, rate_val, rain_val)
		fig3.savefig('step3_aggregation_defuzzification.png', dpi=150, bbox_inches='tight')
		print("✓ Saved: step3_aggregation_defuzzification.png\n")
		
		# Step 4: Detailed Centroid Calculation
		print("=" * 60)
		print("STEP 4: CENTROID CALCULATION (Detailed)")
		print("=" * 60)
		print("Visualizing the centroid defuzzification process...")
		fig4, centroid_value = self.plot_centroid_calculation(water_val, rate_val, rain_val)
		fig4.savefig('step4_centroid_calculation.png', dpi=150, bbox_inches='tight')
		print("✓ Saved: step4_centroid_calculation.png\n")
		
		# Also run through the actual system for comparison
		print("=" * 60)
		print("VERIFICATION WITH ACTUAL SYSTEM")
		print("=" * 60)
		self.flood_system.reset_history()
		result = self.flood_system.calculate_risk(water_distance_cm, rainfall_mm_per_hour)
		print(f"System Risk Score: {result['risk_score']:.2f}%")
		print(f"Warning Level: {result['warning_level']}")
		print(f"Status: {result['status_message']}")
		print(f"{'='*60}\n")
		
		return defuzz_value

	def plot_fuzzification_clean(self, water_val, rate_val, rain_val):
		"""Plot improved fuzzification - only active memberships"""
		fig, axes = plt.subplots(3, 1, figsize=(12, 10))
		
		# Calculate memberships
		water_mem = self.calculate_memberships(self.water_level_universe, self.water_mf, water_val)
		rate_mem = self.calculate_memberships(self.rate_change_universe, self.rate_mf, rate_val)
		rain_mem = self.calculate_memberships(self.rainfall_universe, self.rain_mf, rain_val)
		
		# Get active memberships (> 0.01)
		active_water = {k: v for k, v in water_mem.items() if v > 0.01}
		active_rate = {k: v for k, v in rate_mem.items() if v > 0.01}
		active_rain = {k: v for k, v in rain_mem.items() if v > 0.01}
		
		# Plot Water Level
		ax = axes[0]
		for term in active_water.keys():
			color = self.color_water.get(term, '#888888')
			ax.plot(self.water_level_universe, self.water_mf[term], 
				   label=f'{term}',
				   linewidth=3, color=color, alpha=0.8)
			ax.fill_between(self.water_level_universe, 0, self.water_mf[term], 
						   alpha=0.25, color=color)
		
		# Add input line and membership values
		ax.axvline(water_val, color='#FF0000', linestyle='-', linewidth=3, 
				  label=f'Input = {water_val:.2f} cm', zorder=10)
		
		# Add membership value markers
		for term in active_water.keys():
			y_val = water_mem[term]
			color = self.color_water.get(term, '#888888')
			ax.plot(water_val, y_val, 'o', markersize=10, 
				   color=color, markeredgecolor='white', 
				   markeredgewidth=2, zorder=11)
			ax.text(water_val + (self.water_level_universe[-1] - self.water_level_universe[0]) * 0.02, 
				   y_val, f'μ={y_val:.3f}', 
				   fontsize=10, fontweight='bold', color=color,
				   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
						   edgecolor=color, linewidth=1.5))
		
		ax.set_title('Fuzzifikasi Elevasi Air', fontweight='bold', fontsize=13, pad=12)
		ax.set_xlabel('Elevasi Air (cm)', fontsize=11, fontweight='600')
		ax.set_ylabel('Derajat Keanggotaan (μ)', fontsize=11, fontweight='600')
		ax.legend(fontsize=10, loc='best', framealpha=0.95, edgecolor='black')
		ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
		ax.set_ylim([0, 1.08])
		ax.set_facecolor('#FAFAFA')
		
		# Plot Rate of Change
		ax = axes[1]
		for term in active_rate.keys():
			color = self.color_rate.get(term, '#888888')
			ax.plot(self.rate_change_universe, self.rate_mf[term], 
				   label=f'{term}',
				   linewidth=3, color=color, alpha=0.8)
			ax.fill_between(self.rate_change_universe, 0, self.rate_mf[term], 
						   alpha=0.25, color=color)
		
		ax.axvline(rate_val, color='#FF0000', linestyle='-', linewidth=3, 
				  label=f'Input = {rate_val:.2f} cm/min', zorder=10)
		
		# Add membership value markers
		for term in active_rate.keys():
			y_val = rate_mem[term]
			color = self.color_rate.get(term, '#888888')
			ax.plot(rate_val, y_val, 'o', markersize=10, 
				   color=color, markeredgecolor='white', 
				   markeredgewidth=2, zorder=11)
			ax.text(rate_val + (self.rate_change_universe[-1] - self.rate_change_universe[0]) * 0.03, 
				   y_val, f'μ={y_val:.3f}', 
				   fontsize=10, fontweight='bold', color=color,
				   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
						   edgecolor=color, linewidth=1.5))
		
		ax.set_title('Fuzzifikasi Kenaikan Air', fontweight='bold', fontsize=13, pad=12)
		ax.set_xlabel('Kenaikan Air (cm/min)', fontsize=11, fontweight='600')
		ax.set_ylabel('Derajat Keanggotaan (μ)', fontsize=11, fontweight='600')
		ax.legend(fontsize=10, loc='best', framealpha=0.95, edgecolor='black')
		ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
		ax.set_ylim([0, 1.08])
		ax.set_facecolor('#FAFAFA')
		
		# Plot Rainfall
		ax = axes[2]
		for term in active_rain.keys():
			color = self.color_rain.get(term, '#888888')
			ax.plot(self.rainfall_universe, self.rain_mf[term], 
				   label=f'{term}',
				   linewidth=3, color=color, alpha=0.8)
			ax.fill_between(self.rainfall_universe, 0, self.rain_mf[term], 
						   alpha=0.25, color=color)
		
		ax.axvline(rain_val, color='#FF0000', linestyle='-', linewidth=3, 
				  label=f'Input = {rain_val:.1f} mm/h', zorder=10)
		
		# Add membership value markers
		for term in active_rain.keys():
			y_val = rain_mem[term]
			color = self.color_rain.get(term, '#888888')
			ax.plot(rain_val, y_val, 'o', markersize=10, 
				   color=color, markeredgecolor='white', 
				   markeredgewidth=2, zorder=11)
			ax.text(rain_val + (self.rainfall_universe[-1] - self.rainfall_universe[0]) * 0.02, 
				   y_val, f'μ={y_val:.3f}', 
				   fontsize=10, fontweight='bold', color=color,
				   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
						   edgecolor=color, linewidth=1.5))
		
		ax.set_title('Fuzzifikasi Curah Hujan', fontweight='bold', fontsize=13, pad=12)
		ax.set_xlabel('Curah Hujan (mm/jam)', fontsize=11, fontweight='600')
		ax.set_ylabel('Derajat Keanggotaan (μ)', fontsize=11, fontweight='600')
		ax.legend(fontsize=10, loc='best', framealpha=0.95, edgecolor='black')
		ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
		ax.set_ylim([0, 1.08])
		ax.set_facecolor('#FAFAFA')
		
		plt.tight_layout()
		return fig

	def plot_implication(self, water_val, rate_val, rain_val):
		"""Plot rule implication with clipped consequents"""
		fig, ax = plt.subplots(figsize=(14, 10))
		
		# Calculate input memberships
		water_mem = self.calculate_memberships(self.water_level_universe, self.water_mf, water_val)
		rate_mem = self.calculate_memberships(self.rate_change_universe, self.rate_mf, rate_val)
		rain_mem = self.calculate_memberships(self.rainfall_universe, self.rain_mf, rain_val)
		
		print(f"\n{'='*60}")
		print("ACTIVE INPUT MEMBERSHIPS:")
		print(f"{'='*60}")
		print("Water Level:")
		for term, val in water_mem.items():
			if val > 0.01:
				print(f"  • {term}: μ={val:.3f}")
		print("Rate of Change:")
		for term, val in rate_mem.items():
			if val > 0.01:
				print(f"  • {term}: μ={val:.3f}")
		print("Rainfall:")
		for term, val in rain_mem.items():
			if val > 0.01:
				print(f"  • {term}: μ={val:.3f}")
		print(f"{'='*60}\n")
		
		# Evaluate each rule and collect activations
		rule_activations = []
		
		for i, rule in enumerate(self.rules):
			try:
				# Use the rule's own evaluation method
				antecedent_activation = rule.antecedent.view(sim=self.flood_system.fuzzy_system)
				
				# Get consequent term name
				consequent_term = rule.consequent[0].term.label
				
				if antecedent_activation > 0.001:
					rule_activations.append((consequent_term, antecedent_activation, i+1, rule))
					
			except Exception as e:
				# Fallback to string matching if direct evaluation fails
				rule_str = str(rule)
				activation = 1.0
				
				for wl_term, wl_val in water_mem.items():
					if wl_term in rule_str:
						activation = min(activation, wl_val)
				
				for rc_term, rc_val in rate_mem.items():
					if rc_term in rule_str:
						activation = min(activation, rc_val)
				
				for rf_term, rf_val in rain_mem.items():
					if rf_term in rule_str:
						activation = min(activation, rf_val)
				
				for risk_term in self.risk_mf.keys():
					if risk_term in rule_str:
						if activation > 0.001:
							rule_activations.append((risk_term, activation, i+1, rule))
						break
		
		# Sort by activation level
		rule_activations.sort(key=lambda x: -x[1])
		
		# Display rule firing information
		print(f"{'='*60}")
		print(f"FIRED RULES (All Active):")
		print(f"{'='*60}")
		for j, (risk_term, activation, rule_num, rule) in enumerate(rule_activations):
			print(f"Rule {rule_num}: {risk_term.upper()} (α={activation:.3f})")
			print(f"  {rule}")
			if j >= 14:  # Limit console output
				remaining = len(rule_activations) - 15
				if remaining > 0:
					print(f"  ... and {remaining} more rules")
				break
		print(f"{'='*60}\n")
		
		# Plot the original membership functions (dashed, lighter)
		for risk_term, risk_mf in self.risk_mf.items():
			color = self.color_risk.get(risk_term, '#888888')
			ax.plot(self.flood_risk_universe, risk_mf, '--', 
				   color=color, linewidth=1, alpha=0.2)
		
		# Group rules by consequent to avoid overlapping annotations
		consequent_groups = {}
		for risk_term, activation, rule_num, rule in rule_activations[:10]:
			if risk_term not in consequent_groups:
				consequent_groups[risk_term] = []
			consequent_groups[risk_term].append((activation, rule_num))
		
		# Plot each consequent group
		shown_terms = set()
		annotation_positions = []  # Track annotation positions to avoid overlaps
		
		for risk_term in list(self.risk_mf.keys()):  # FIXED: Use actual risk terms
			if risk_term not in consequent_groups:
				continue
			
			color = self.color_risk.get(risk_term, '#888888')
			rules_in_group = sorted(consequent_groups[risk_term], key=lambda x: -x[0])
			
			# Plot all clipped shapes for this consequent with transparency
			for idx, (activation, rule_num) in enumerate(rules_in_group):
				# Clip the membership function at activation level
				clipped_mf = np.minimum(self.risk_mf[risk_term], activation)
				
				# Use alpha to show multiple rules
				alpha_val = 0.6 - (idx * 0.1)
				alpha_val = max(0.3, alpha_val)
				
				# Add label only once per term
				label = None
				if risk_term not in shown_terms:
					label = f'{risk_term}'
					shown_terms.add(risk_term)
				
				# Plot the clipped shape
				ax.fill_between(self.flood_risk_universe, 0, clipped_mf,
							   alpha=alpha_val, color=color, edgecolor=color, linewidth=1.5,
							   label=label)
				
				# Add horizontal line showing clipping level (only for highest activation)
				if idx == 0:
					ax.hlines(activation, 0, 100, colors=color, 
							 linestyles=':', linewidth=1.5, alpha=0.3)
			
			# Add single annotation for this consequent showing all rules
			peak_idx = np.argmax(self.risk_mf[risk_term])
			peak_x = self.flood_risk_universe[peak_idx]
			max_activation = rules_in_group[0][0]
			
			# Create text showing all rules in this group
			if len(rules_in_group) == 1:
				rule_text = f'R{rules_in_group[0][1]}\nα={rules_in_group[0][0]:.3f}'
			elif len(rules_in_group) == 2:
				rule_text = (f'R{rules_in_group[0][1]} (α={rules_in_group[0][0]:.3f})\n'
						   f'R{rules_in_group[1][1]} (α={rules_in_group[1][0]:.3f})')
			else:
				# Show all rules without truncation
				rule_text = f'{len(rules_in_group)} rules\n'
				for act, rn in rules_in_group:
					rule_text += f'R{rn} (α={act:.3f})\n'
				# Remove the trailing newline
				rule_text = rule_text.rstrip('\n')
			
			# Find a non-overlapping position for annotation
			# Try different y-offsets to avoid overlaps
			base_y_offset = max_activation + 0.12
			y_offset = base_y_offset
			
			# Check if this position overlaps with existing annotations
			for prev_x, prev_y in annotation_positions:
				if abs(peak_x - prev_x) < 15 and abs(y_offset - prev_y) < 0.15:
					# Too close, move up
					y_offset = prev_y + 0.18
			
			annotation_positions.append((peak_x, y_offset))
			
			ax.annotate(rule_text,
					   xy=(peak_x, max_activation),
					   xytext=(peak_x, y_offset),
					   ha='center', fontsize=9, fontweight='bold',
					   color=color,
					   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
						   edgecolor=color, linewidth=1.5, alpha=0.95),
					   arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.7))
		
		ax.set_title('Implication', 
				   fontweight='bold', fontsize=13, pad=15)
		ax.set_xlabel('Flood Risk (%)', fontsize=12, fontweight='600')
		ax.set_ylabel('Membership Degree (μ)', fontsize=12, fontweight='600')
		ax.legend(fontsize=10, loc='upper left', framealpha=0.95, edgecolor='black')
		ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
		ax.set_facecolor('#FAFAFA')
		ax.set_xlim([0, 100])
		ax.set_ylim([0, 1.1])
		
		plt.tight_layout()
		return fig

	def plot_aggregation(self, water_val, rate_val, rain_val):
		"""Plot aggregation and defuzzification"""
		fig, axes = plt.subplots(2, 1, figsize=(14, 10))
		
		# Calculate the aggregated output by running the fuzzy system
		self.flood_system.fuzzy_system.input['water_level'] = water_val
		self.flood_system.fuzzy_system.input['avg_rate_change'] = rate_val
		self.flood_system.fuzzy_system.input['rainfall'] = rain_val
		
		try:
			self.flood_system.fuzzy_system.compute()
			
			# Get aggregated output
			aggregated_output = np.zeros_like(self.flood_risk_universe, dtype=float)
			
			# Aggregate all rule outputs (using OR/max aggregation)
			water_mem = self.calculate_memberships(self.water_level_universe, self.water_mf, water_val)
			rate_mem = self.calculate_memberships(self.rate_change_universe, self.rate_mf, rate_val)
			rain_mem = self.calculate_memberships(self.rainfall_universe, self.rain_mf, rain_val)
			
			for rule in self.rules:
				rule_str = str(rule)
				activation = 1.0
				
				# Calculate rule activation
				for wl_term, wl_val in water_mem.items():
					if wl_term in rule_str:
						activation = min(activation, wl_val)
				
				for rc_term, rc_val in rate_mem.items():
					if rc_term in rule_str:
						activation = min(activation, rc_val)
				
				for rf_term, rf_val in rain_mem.items():
					if rf_term in rule_str:
						activation = min(activation, rf_val)
				
				# Apply to consequent
				for risk_term in self.risk_mf.keys():
					if risk_term in rule_str:
						clipped = np.minimum(self.risk_mf[risk_term], activation)
						aggregated_output = np.maximum(aggregated_output, clipped)
						break
			
			# Defuzzify
			if np.sum(aggregated_output) > 0:
				defuzz_value = fuzz.defuzz(self.flood_risk_universe, aggregated_output, 'centroid')
			else:
				defuzz_value = 0
			
			# Plot 1: Individual membership functions
			ax1 = axes[0]
			for risk_term, risk_mf in self.risk_mf.items():
				color = self.color_risk.get(risk_term, '#888888')
				ax1.plot(self.flood_risk_universe, risk_mf, '--', 
					   label=risk_term, color=color, linewidth=2, alpha=0.5)
			
			ax1.set_title('Resiko Banjir (Unclipped)', 
					   fontweight='bold', fontsize=13, pad=12)
			ax1.set_xlabel('Resiko Banjir (%)', fontsize=11, fontweight='600')
			ax1.set_ylabel('Derajat Keanggotaan (μ)', fontsize=11, fontweight='600')
			ax1.legend(fontsize=10, loc='best')
			ax1.grid(True, alpha=0.25)
			ax1.set_facecolor('#FAFAFA')
			ax1.set_ylim([0, 1.05])
			
			# Plot 2: Aggregated output with defuzzification
			ax2 = axes[1]
			
			# Fill with lighter color and less opacity
			ax2.fill_between(self.flood_risk_universe, 0, aggregated_output,
						   alpha=0.4, color='#4CAF50', label='Aggregated Output')
			
			# Plot outline with crisp, thin line and darker color
			ax2.plot(self.flood_risk_universe, aggregated_output, 
				   color='#1B5E20', linewidth=1.5, linestyle='-', zorder=5)
			
			# Mark centroid with vertical line
			# ax2.axvline(defuzz_value, color='#FF0000', linestyle='--', 
			# 		  linewidth=2, label=f'Centroid = {defuzz_value:.2f}%', zorder=10)
			
			# Add centroid marker
			# y_at_centroid = fuzz.interp_membership(self.flood_risk_universe, aggregated_output, defuzz_value)
			# ax2.plot([defuzz_value], [y_at_centroid], 'o', markersize=10, 
			# 	   color='#FF0000', markeredgecolor='white', markeredgewidth=2, zorder=11)
			
			ax2.set_title('Agregasi dan Defuzzifikasi', 
					   fontweight='bold', fontsize=13, pad=12)
			ax2.set_xlabel('Resiko Banjir (%)', fontsize=11, fontweight='600')
			ax2.set_ylabel('Derajat Keanggotaan (μ)', fontsize=11, fontweight='600')
			ax2.legend(fontsize=11, loc='best', framealpha=0.95)
			ax2.grid(True, alpha=0.25)
			ax2.set_facecolor('#FAFAFA')
			ax2.set_ylim([0, max(aggregated_output) * 1.15])
			
		except Exception as e:
			print(f"Error in aggregation: {e}")
			defuzz_value = 0
			ax1 = axes[0]
			ax1.text(0.5, 0.5, 'Error computing fuzzy output', 
				   ha='center', va='center', transform=ax1.transAxes)
		
		plt.tight_layout()
		return fig, defuzz_value

	def plot_centroid_calculation(self, water_val, rate_val, rain_val):
		"""Plot detailed centroid calculation with no overlapping elements"""
		fig = plt.figure(figsize=(20, 18))
		gs = fig.add_gridspec(6, 3, hspace=0.5, wspace=0.4, 
							height_ratios=[2.5, 1.5, 1.2, 1.5, 1.2, 1.0],
							top=0.98, bottom=0.02, left=0.05, right=0.97)
		
		# Main aggregated plot
		ax_main = fig.add_subplot(gs[0, :])
		
		# Step 1: Area calculation (left)
		ax_area = fig.add_subplot(gs[1, 0])
		ax_area_detail = fig.add_subplot(gs[2, 0])
		ax_area_detail.axis('off')
		
		# Step 2: Moment calculation (middle)
		ax_moment = fig.add_subplot(gs[1, 1])
		ax_moment_detail = fig.add_subplot(gs[2, 1])
		ax_moment_detail.axis('off')
		
		# Step 3: Division (right)
		ax_division = fig.add_subplot(gs[1, 2])
		ax_division.axis('off')
		ax_division_detail = fig.add_subplot(gs[2, 2])
		ax_division_detail.axis('off')
		
		# Calculation flow panel
		ax_calc = fig.add_subplot(gs[3, :])
		ax_calc.axis('off')
		
		# Calculation detail panel
		ax_calc_detail = fig.add_subplot(gs[4, :])
		ax_calc_detail.axis('off')
		
		# Formula panel
		ax_formula = fig.add_subplot(gs[5, :])
		ax_formula.axis('off')
		
		# Calculate aggregated output
		self.flood_system.fuzzy_system.input['water_level'] = water_val
		self.flood_system.fuzzy_system.input['avg_rate_change'] = rate_val
		self.flood_system.fuzzy_system.input['rainfall'] = rain_val
		
		try:
			self.flood_system.fuzzy_system.compute()
			
			# Get aggregated output
			aggregated_output = np.zeros_like(self.flood_risk_universe, dtype=float)
			
			water_mem = self.calculate_memberships(self.water_level_universe, self.water_mf, water_val)
			rate_mem = self.calculate_memberships(self.rate_change_universe, self.rate_mf, rate_val)
			rain_mem = self.calculate_memberships(self.rainfall_universe, self.rain_mf, rain_val)
			
			for rule in self.rules:
				rule_str = str(rule)
				activation = 1.0
				
				for wl_term, wl_val in water_mem.items():
					if wl_term in rule_str:
						activation = min(activation, wl_val)
				
				for rc_term, rc_val in rate_mem.items():
					if rc_term in rule_str:
						activation = min(activation, rc_val)
				
				for rf_term, rf_val in rain_mem.items():
					if rf_term in rule_str:
						activation = min(activation, rf_val)
				
				for risk_term in self.risk_mf.keys():
					if risk_term in rule_str:
						clipped = np.minimum(self.risk_mf[risk_term], activation)
						aggregated_output = np.maximum(aggregated_output, clipped)
						break
			
			if np.sum(aggregated_output) > 0:
				# Calculate centroid manually with detailed breakdown
				numerator = np.sum(self.flood_risk_universe * aggregated_output)
				denominator = np.sum(aggregated_output)
				centroid = numerator / denominator
				
				# Get sample points for detailed calculation display
				sample_indices = [0, 25, 50, 75, 99]  # Show 5 sample points
				
				# Verify with fuzz.defuzz
				centroid_verify = fuzz.defuzz(self.flood_risk_universe, aggregated_output, 'centroid')
				
				# === MAIN PLOT ===
				ax_main.fill_between(self.flood_risk_universe, 0, aggregated_output,
								alpha=0.4, color='#4CAF50', label='Aggregated Membership')
				ax_main.plot(self.flood_risk_universe, aggregated_output, 
						linewidth=2.5, color='#2E7D32')
				
				# Mark sample points
				for idx in sample_indices:
					x_val = self.flood_risk_universe[idx]
					y_val = aggregated_output[idx]
					ax_main.plot(x_val, y_val, 'o', markersize=8, 
							color='#2196F3', markeredgecolor='white', markeredgewidth=2, zorder=8)
				
				# Mark centroid
				y_centroid = fuzz.interp_membership(self.flood_risk_universe, aggregated_output, centroid)
				ax_main.axvline(centroid, color='#FF0000', linestyle='--', 
							linewidth=3, alpha=0.8, zorder=5)
				ax_main.plot([centroid], [y_centroid], 's', markersize=18, 
						color='#FF0000', markeredgecolor='white', markeredgewidth=3, zorder=6)
				
				# Annotation - positioned to avoid overlap
				annotation_x = centroid + 8 if centroid < 65 else centroid - 8
				annotation_ha = 'left' if centroid < 65 else 'right'
				ax_main.annotate(f'CoG = {centroid:.2f}%', 
						xy=(centroid, y_centroid),
						xytext=(annotation_x, y_centroid + 0.22),
						fontsize=13, fontweight='bold', color='#FF0000', ha=annotation_ha,
						bbox=dict(boxstyle='round,pad=0.7', facecolor='white', 
							edgecolor='#FF0000', linewidth=3),
						arrowprops=dict(arrowstyle='->', color='#FF0000', lw=2.5))
				
				# Balance visualization
				triangle_x = [centroid - 4, centroid + 4, centroid, centroid - 4]
				triangle_y = [-0.08, -0.08, -0.03, -0.08]
				ax_main.fill(triangle_x, triangle_y, color='#FF0000', alpha=0.7, zorder=7)
				ax_main.plot([centroid, centroid], [-0.03, 0], 'r-', linewidth=3, zorder=7)
				
				ax_main.set_xlabel('Flood Risk (%)', fontsize=12, fontweight='600')
				ax_main.set_ylabel('Membership Degree (μ)', fontsize=12, fontweight='600')
				ax_main.legend(fontsize=11, loc='upper left', framealpha=0.95, edgecolor='black')
				ax_main.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
				ax_main.set_ylim([-0.12, 1.15])
				ax_main.set_xlim([0, 100])
				ax_main.set_facecolor('#FAFAFA')
				
				# === STEP 1: AREA (DENOMINATOR) ===
				ax_area.fill_between(self.flood_risk_universe, 0, aggregated_output,
								alpha=0.6, color='#4CAF50', edgecolor='#2E7D32', linewidth=1.5)
				ax_area.plot(self.flood_risk_universe, aggregated_output, 
						linewidth=2.5, color='#1B5E20')
				
				# Mark sample points
				for idx in sample_indices:
					x_val = self.flood_risk_universe[idx]
					y_val = aggregated_output[idx]
					ax_area.plot(x_val, y_val, 'o', markersize=7, 
							color='#2196F3', markeredgecolor='white', markeredgewidth=1.5)
					ax_area.plot([x_val, x_val], [0, y_val], 'b--', linewidth=1, alpha=0.4)
				
				# Result box - positioned at top right to avoid overlap
				ax_area.text(0.97, 0.96, f'Σμ(x) = {denominator:.2f}', 
						transform=ax_area.transAxes, ha='right', va='top',
						fontsize=12, fontweight='bold', color='#1B5E20',
						bbox=dict(boxstyle='round,pad=0.6', facecolor='#E8F5E9', 
							edgecolor='#2E7D32', linewidth=2.5))
				
				ax_area.set_xlabel('Risk Level x (%)', fontsize=10, fontweight='600')
				ax_area.set_ylabel('μ(x)', fontsize=10, fontweight='600')
				ax_area.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
				ax_area.set_facecolor('#FAFAFA')
				ax_area.set_xlim([0, 100])
				ax_area.set_ylim([0, max(aggregated_output) * 1.12])
				
				# Detailed calculation for area
				detail_text = "Sum all membership values:\n"
				detail_text += "─────────────────────\n"
				for i, idx in enumerate(sample_indices):
					x_val = self.flood_risk_universe[idx]
					y_val = aggregated_output[idx]
					detail_text += f"μ[{idx:2d}] = {y_val:.4f}\n"
				detail_text += f"... (100 total)\n"
				detail_text += "─────────────────────\n"
				detail_text += f"Σμ(x) = {denominator:.4f}"
				
				ax_area_detail.text(0.5, 0.5, detail_text, 
							ha='center', va='center',
							fontsize=9.5, family='monospace',
							bbox=dict(boxstyle='round,pad=0.9', facecolor='#E8F5E9', 
								edgecolor='#2E7D32', linewidth=2.5))
				
				# === STEP 2: MOMENT (NUMERATOR) ===
				weighted_contribution = self.flood_risk_universe * aggregated_output
				ax_moment.fill_between(self.flood_risk_universe, 0, weighted_contribution,
									alpha=0.6, color='#FF9800', edgecolor='#F57C00', linewidth=1.5)
				ax_moment.plot(self.flood_risk_universe, weighted_contribution, 
						linewidth=2.5, color='#E65100')
				
				# Mark sample points
				for idx in sample_indices:
					x_val = self.flood_risk_universe[idx]
					y_val = weighted_contribution[idx]
					ax_moment.plot(x_val, y_val, 'o', markersize=7, 
							color='#2196F3', markeredgecolor='white', markeredgewidth=1.5)
					ax_moment.plot([x_val, x_val], [0, y_val], 'b--', linewidth=1, alpha=0.4)
				
				ax_moment.text(0.97, 0.96, f'Σ(x·μ(x)) = {numerator:.2f}', 
						transform=ax_moment.transAxes, ha='right', va='top',
						fontsize=12, fontweight='bold', color='#E65100',
						bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFF3E0', 
							edgecolor='#F57C00', linewidth=2.5))
				
				ax_moment.set_xlabel('Risk Level x (%)', fontsize=10, fontweight='600')
				ax_moment.set_ylabel('x × μ(x)', fontsize=10, fontweight='600')
				ax_moment.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
				ax_moment.set_facecolor('#FAFAFA')
				ax_moment.set_xlim([0, 100])
				ax_moment.set_ylim([0, max(weighted_contribution) * 1.12])
				
				# Detailed calculation for moment
				detail_text = "Multiply x by μ(x), then sum:\n"
				detail_text += "──────────────────────────\n"
				for i, idx in enumerate(sample_indices):
					x_val = self.flood_risk_universe[idx]
					mu_val = aggregated_output[idx]
					product = x_val * mu_val
					detail_text += f"{x_val:5.1f}×{mu_val:.4f}={product:6.2f}\n"
				detail_text += f"... (100 total)\n"
				detail_text += "──────────────────────────\n"
				detail_text += f"Σ(x·μ(x)) = {numerator:.4f}"
				
				ax_moment_detail.text(0.5, 0.5, detail_text, 
							ha='center', va='center',
							fontsize=9.5, family='monospace',
							bbox=dict(boxstyle='round,pad=0.9', facecolor='#FFF3E0', 
								edgecolor='#F57C00', linewidth=2.5))
				
				# === STEP 3: DIVISION ===
				# Visual representation of division
				# ax_division.text(0.5, 0.88, 'STEP 3: Divide', ha='center', va='top',
				# 			fontsize=13, fontweight='bold', color='#2196F3')
				
				# Large division symbol
				ax_division.plot([0.15, 0.85], [0.5, 0.5], 'k-', linewidth=3)
				
				# Numerator box
				ax_division.text(0.5, 0.7, f'{numerator:.2f}', ha='center', va='center',
							fontsize=18, fontweight='bold', color='#E65100',
							bbox=dict(boxstyle='round,pad=0.7', facecolor='#FFF3E0', 
								edgecolor='#F57C00', linewidth=2.5))
				
				# Denominator box
				ax_division.text(0.5, 0.3, f'{denominator:.2f}', ha='center', va='center',
							fontsize=18, fontweight='bold', color='#1B5E20',
							bbox=dict(boxstyle='round,pad=0.7', facecolor='#E8F5E9', 
								edgecolor='#2E7D32', linewidth=2.5))
				
				# Equals arrow
				ax_division.annotate('', xy=(0.5, 0.08), xytext=(0.5, 0.15),
							arrowprops=dict(arrowstyle='->', lw=3, color='#2196F3'))
				
				ax_division.set_xlim([0, 1])
				ax_division.set_ylim([0, 1])
				
				# Division detail
				detail_text = "Division:\n"
				detail_text += "─────────────────\n"
				detail_text += f"  {numerator:.4f}\n"
				detail_text += f"  ─────────\n"
				detail_text += f"  {denominator:.4f}\n"
				detail_text += "─────────────────\n"
				detail_text += f"= {centroid:.4f}%\n"
				detail_text += "─────────────────\n"
				detail_text += f"≈ {centroid:.2f}%"
				
				ax_division_detail.text(0.5, 0.5, detail_text, 
							ha='center', va='center',
							fontsize=10, family='monospace', fontweight='bold',
							bbox=dict(boxstyle='round,pad=1.0', facecolor='#E3F2FD', 
								edgecolor='#2196F3', linewidth=2.5))
				
				# === CALCULATION FLOW ===
				flow_y = 0.5
				
				# Title
				ax_calc.text(0.5, 0.85, 'Centroid Calculation Process', 
						ha='center', va='center',
						fontsize=14, fontweight='bold', color='#424242',
						bbox=dict(boxstyle='round,pad=0.6', facecolor='#EEEEEE', 
							edgecolor='#757575', linewidth=2))
				
				# Flow boxes
				flow_items = [
					('①', 'Sum μ(x)', f'{denominator:.2f}', '#4CAF50', '#E8F5E9', 0.10),
					('②', 'Sum x·μ(x)', f'{numerator:.2f}', '#FF9800', '#FFF3E0', 0.32),
					('③', 'Divide', f'{numerator:.2f}/{denominator:.2f}', '#2196F3', '#E3F2FD', 0.57),
					('④', 'Result', f'{centroid:.2f}%', '#D32F2F', '#FFEBEE', 0.82)
				]
				
				for i, (num, label, value, color, bgcolor, x_pos) in enumerate(flow_items):
					# Number badge
					ax_calc.text(x_pos, flow_y + 0.13, num, 
							ha='center', va='center',
							fontsize=14, fontweight='bold', color='white',
							bbox=dict(boxstyle='circle,pad=0.4', facecolor=color, linewidth=0))
					
					# Label
					ax_calc.text(x_pos, flow_y - 0.02, label, 
							ha='center', va='center',
							fontsize=10, fontweight='bold', color=color)
					
					# Value box
					ax_calc.text(x_pos, flow_y - 0.18, value, 
							ha='center', va='center',
							fontsize=10, fontweight='bold', color=color,
							bbox=dict(boxstyle='round,pad=0.6', facecolor=bgcolor, 
								edgecolor=color, linewidth=2))
					
					# Arrow
					if i < len(flow_items) - 1:
						next_x = flow_items[i+1][5]
						ax_calc.annotate('', xy=(next_x - 0.03, flow_y), 
									xytext=(x_pos + 0.06, flow_y),
									arrowprops=dict(arrowstyle='->', lw=2.5, 
										color='#757575', alpha=0.6))
				
			else:
				ax_main.text(0.5, 0.5, 'No active rules - cannot calculate centroid', 
						ha='center', va='center', fontsize=16, color='gray',
						transform=ax_main.transAxes)
				centroid = 0
			
		except Exception as e:
			print(f"Error in centroid calculation: {e}")
			centroid = 0
			ax_main.text(0.5, 0.5, f'Error: {str(e)}', 
					ha='center', va='center', fontsize=12, color='red',
					transform=ax_main.transAxes)
		
		return fig, centroid	

if __name__ == "__main__":
	print("\n" + "=" * 60)
	print("DYNAMIC FUZZY INFERENCE SYSTEM VISUALIZER")
	print("Flood Risk Assessment - Synced with main2.py")
	print("=" * 60)
	
	# Initialize with calibration parameters
	# These should match your sensor setup
	GROUND_DISTANCE = 100  # cm - distance to ground when dry
	SIAGA_LEVEL = 130      # cm - alert threshold
	BANJIR_LEVEL = 100     # cm - flood threshold
	
	visualizer = DynamicFuzzyVisualizer(
		ground_distance=GROUND_DISTANCE,
		siaga_level=SIAGA_LEVEL,
		banjir_level=BANJIR_LEVEL
	)
	
	# Test with sample values
	water_distance = 100		# cm - sensor reading (closer = higher water)
	rate_change = 0.5			# cm/min - negative means water rising
	rainfall = 15	            # mm/hour
	
	print(f"\nTest Input Values:")
	print(f"  • Water Distance: {water_distance} cm")
	print(f"  • Rate of Change: {rate_change} cm/min")
	print(f"  • Rainfall: {rainfall} mm/hour")
	
	# Generate all visualizations
	defuzz_output = visualizer.visualize_all_steps(
		water_distance, 
		rate_change, 
		rainfall
	)
	
	print("\n" + "=" * 60)
	print("FINAL VISUALIZATION RESULT")
	print("=" * 60)
	print(f"Defuzzified Flood Risk Score: {defuzz_output:.2f}%")
	
	# Categorize the risk
	if defuzz_output >= 85:
		risk_category = "CRITICAL"
		color = "🔴"
	elif defuzz_output >= 60:
		risk_category = "HIGH"
		color = "🟠"
	elif defuzz_output >= 25:
		risk_category = "MEDIUM"
		color = "🟡"
	else:
		risk_category = "LOW"
		color = "🟢"
	
	print(f"Risk Category: {color} {risk_category}")
	print("=" * 60)
	print("\n✓ All visualizations saved successfully!")
	print("  - step1_fuzzification.png")
	print("  - step2_implication.png")
	print("  - step3_aggregation_defuzzification.png")
	print("  - step4_centroid_calculation.png")
	print("\n")
	
	# plt.show()