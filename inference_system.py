import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

plt.rcParams['font.family'] = 'DejaVu Sans'

class CleanFuzzyVisualizer:
	def __init__(self):
		"""Initialize fuzzy system"""
		self.create_fuzzy_system()
	
	def create_fuzzy_system(self):
		"""Create fuzzy system"""
		# Define fuzzy variables
		self.water_level = np.arange(0, 1.01, 0.01)
		self.rate_change = np.arange(-1, 1.01, 0.01)
		self.rainfall = np.arange(0, 1.01, 0.01)
		self.flood_risk = np.arange(0, 101, 1)
		
		# Water Level membership functions
		self.water_mf = {
			'normal': fuzz.trapmf(self.water_level, [0, 0, 0.1, 0.25]),
			'siaga': fuzz.trimf(self.water_level, [0.15, 0.4, 0.65]),
			'siaga_2': fuzz.trimf(self.water_level, [0.5, 0.7, 0.85]),
			'banjir': fuzz.trapmf(self.water_level, [0.75, 0.9, 1.0, 1.0])
		}
		
		# Rate of Change membership functions (matching main2.py - 8 levels)
		self.rate_mf = {
			'turun_sangat_cepat': fuzz.trapmf(self.rate_change, [-1, -1, -0.6, -0.4]),
			'turun_cepat': fuzz.trimf(self.rate_change, [-0.5, -0.3, -0.15]),
			'turun_lambat': fuzz.trimf(self.rate_change, [-0.2, -0.1, -0.03]),
			'stabil': fuzz.trimf(self.rate_change, [-0.05, 0, 0.05]),
			'naik_lambat': fuzz.trimf(self.rate_change, [0.03, 0.1, 0.2]),
			'naik_cepat': fuzz.trimf(self.rate_change, [0.15, 0.3, 0.5]),
			'naik_sangat_cepat': fuzz.trimf(self.rate_change, [0.4, 0.65, 0.85]),
			'naik_ekstrem': fuzz.trapmf(self.rate_change, [0.75, 0.9, 1, 1])
		}
		
		# Rainfall membership functions
		self.rain_mf = {
			'tidak_hujan': fuzz.trapmf(self.rainfall, [0, 0, 0.02, 0.04]),
			'ringan': fuzz.trimf(self.rainfall, [0.02, 0.1, 0.2]),
			'sedang': fuzz.trimf(self.rainfall, [0.15, 0.3, 0.45]),
			'lebat': fuzz.trimf(self.rainfall, [0.35, 0.65, 0.8]),
			'sangat_lebat': fuzz.trapmf(self.rainfall, [0.75, 0.9, 1, 1]),
		}
		
		# Output membership functions
		self.risk_mf = {
			'low': fuzz.trapmf(self.flood_risk, [0, 0, 15, 30]),
			'medium': fuzz.trimf(self.flood_risk, [25, 45, 65]),
			'high': fuzz.trimf(self.flood_risk, [60, 75, 88]),
			'critical': fuzz.trapmf(self.flood_risk, [85, 92, 100, 100])
		}
		
		# Define color schemes
		self.color_water = {'normal': '#2E86AB', 'siaga': '#A23B72', 
						   'siaga_2': '#F18F01', 'banjir': '#C73E1D'}
		self.color_rate = {'turun_sangat_cepat': '#0A2463', 'turun_cepat': '#1B4965', 
						  'turun_lambat': '#5FA8D3', 'stabil': '#62B6CB', 
						  'naik_lambat': '#CAE9FF', 'naik_cepat': '#FFB627', 
						  'naik_sangat_cepat': '#FF6B35', 'naik_ekstrem': '#C1121F'}
		self.color_rain = {'tidak_hujan': '#B8E0D2', 'ringan': '#95D5B2',
						  'sedang': '#74C69D', 'lebat': '#52B788',
						  'sangat_lebat': '#40916C'}
		self.color_risk = {'low': '#06D6A0', 'medium': '#FFD166', 
						  'high': '#EF476F', 'critical': '#9D0208'}
	
	def calculate_memberships(self, universe, mf_dict, value):
		"""Calculate membership degrees for all categories"""
		memberships = {}
		for term, mf in mf_dict.items():
			memberships[term] = fuzz.interp_membership(universe, mf, value)
		return memberships
	
	def visualize_all_steps(self, water_val, rate_val, rain_val):
		"""Generate all visualization steps separately"""
		print("\n" + "=" * 60)
		print("FUZZY INFERENCE VISUALIZATION")
		print("=" * 60)
		print(f"\nInput Values:")
		print(f"  • Water Level (normalized): {water_val}")
		print(f"  • Rate of Change (normalized): {rate_val}")
		print(f"  • Rainfall (normalized): {rain_val}\n")
		
		# Step 1: Fuzzification
		print("=" * 60)
		print("STEP 1: FUZZIFICATION")
		print("=" * 60)
		print("Converting crisp inputs to fuzzy membership values...")
		fig = self.plot_fuzzification_clean(
			water_val, rate_val, rain_val)
		fig.savefig('step1_fuzzification.png', dpi=150, bbox_inches='tight')
		print("✓ Saved: step1_fuzzification.png\n")
		
		# Step 2: Implication (Rule Firing)
		print("=" * 60)
		print("STEP 2: IMPLICATION (Rule Evaluation)")
		print("=" * 60)
		print("Evaluating fuzzy rules and clipping output functions...")
		fig2 = self.plot_imp(water_val, rate_val, rain_val)
		fig2.savefig('step2_implication.png', dpi=150, bbox_inches='tight')
		print("✓ Saved: step2_implication.png\n")

		# Step 3: Aggregation & Defuzzification
		print("=" * 60)
		print("STEP 3: AGGREGATION & DEFUZZIFICATION")
		print("=" * 60)
		print("Combining all rule outputs and computing final crisp value...")
		fig3, defuzz_value = self.plot_aggregation(
			water_val, rate_val, rain_val)
		fig3.savefig('step3_aggregation_defuzzification.png', dpi=150, bbox_inches='tight')
		print("✓ Saved: step3_aggregation_defuzzification.png\n")
		
		# Step 4: Detailed Centroid Calculation
		print("=" * 60)
		print("STEP 4: CENTROID CALCULATION (Detailed)")
		print("=" * 60)
		print("Visualizing the centroid defuzzification process...")
		fig4, centroid_value = self.plot_centroid_calculation(
			water_val, rate_val, rain_val)
		fig4.savefig('step4_centroid_calculation.png', dpi=150, bbox_inches='tight')
		print("✓ Saved: step4_centroid_calculation.png\n")
		
		return defuzz_value

	def plot_fuzzification_clean(self, water_val, rate_val, rain_val):
		"""Plot improved fuzzification - only active memberships"""
		fig, axes = plt.subplots(3, 1, figsize=(12, 10))
		
		# Calculate memberships
		water_mem = self.calculate_memberships(self.water_level, self.water_mf, water_val)
		rate_mem = self.calculate_memberships(self.rate_change, self.rate_mf, rate_val)
		rain_mem = self.calculate_memberships(self.rainfall, self.rain_mf, rain_val)
		
		# Get active memberships (> 0.01)
		active_water = {k: v for k, v in water_mem.items() if v > 0.01}
		active_rate = {k: v for k, v in rate_mem.items() if v > 0.01}
		active_rain = {k: v for k, v in rain_mem.items() if v > 0.01}
		
		# Plot Water Level
		ax = axes[0]
		for term in active_water.keys():
			ax.plot(self.water_level, self.water_mf[term], 
				   label=f'{term}',
				   linewidth=3, color=self.color_water[term], alpha=0.8)
			ax.fill_between(self.water_level, 0, self.water_mf[term], 
						   alpha=0.25, color=self.color_water[term])
		
		# Add input line and membership values
		ax.axvline(water_val, color='#FF0000', linestyle='-', linewidth=3, 
				  label=f'Input = {water_val:.2f}', zorder=10)
		
		# Add membership value markers
		for term in active_water.keys():
			y_val = water_mem[term]
			ax.plot(water_val, y_val, 'o', markersize=10, 
				   color=self.color_water[term], markeredgecolor='white', 
				   markeredgewidth=2, zorder=11)
			ax.text(water_val + 0.03, y_val, f'μ={y_val:.3f}', 
				   fontsize=10, fontweight='bold', color=self.color_water[term],
				   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
						   edgecolor=self.color_water[term], linewidth=1.5))
		
		ax.set_title('Water Level Fuzzification', fontweight='bold', fontsize=13, pad=12)
		ax.set_xlabel('Normalized Value', fontsize=11, fontweight='600')
		ax.set_ylabel('Membership Degree (μ)', fontsize=11, fontweight='600')
		ax.legend(fontsize=10, loc='best', framealpha=0.95, edgecolor='black')
		ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
		ax.set_ylim([0, 1.08])
		ax.set_facecolor('#FAFAFA')
		
		# Plot Rate of Change
		ax = axes[1]
		for term in active_rate.keys():
			ax.plot(self.rate_change, self.rate_mf[term], 
				   label=f'{term}',
				   linewidth=3, color=self.color_rate[term], alpha=0.8)
			ax.fill_between(self.rate_change, 0, self.rate_mf[term], 
						   alpha=0.25, color=self.color_rate[term])
		
		ax.axvline(rate_val, color='#FF0000', linestyle='-', linewidth=3, 
				  label=f'Input = {rate_val:.2f}', zorder=10)
		
		# Add membership value markers
		for term in active_rate.keys():
			y_val = rate_mem[term]
			ax.plot(rate_val, y_val, 'o', markersize=10, 
				   color=self.color_rate[term], markeredgecolor='white', 
				   markeredgewidth=2, zorder=11)
			ax.text(rate_val + 0.05, y_val, f'μ={y_val:.3f}', 
				   fontsize=10, fontweight='bold', color=self.color_rate[term],
				   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
						   edgecolor=self.color_rate[term], linewidth=1.5))
		
		ax.set_title('Rate of Change Fuzzification', fontweight='bold', fontsize=13, pad=12)
		ax.set_xlabel('Normalized Value', fontsize=11, fontweight='600')
		ax.set_ylabel('Membership Degree (μ)', fontsize=11, fontweight='600')
		ax.legend(fontsize=10, loc='best', framealpha=0.95, edgecolor='black')
		ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
		ax.set_ylim([0, 1.08])
		ax.set_facecolor('#FAFAFA')
		
		# Plot Rainfall
		ax = axes[2]
		for term in active_rain.keys():
			ax.plot(self.rainfall, self.rain_mf[term], 
				   label=f'{term}',
				   linewidth=3, color=self.color_rain[term], alpha=0.8)
			ax.fill_between(self.rainfall, 0, self.rain_mf[term], 
						   alpha=0.25, color=self.color_rain[term])
		
		ax.axvline(rain_val, color='#FF0000', linestyle='-', linewidth=3, 
				  label=f'Input = {rain_val:.2f}', zorder=10)
		
		# Add membership value markers
		for term in active_rain.keys():
			y_val = rain_mem[term]
			ax.plot(rain_val, y_val, 'o', markersize=10, 
				   color=self.color_rain[term], markeredgecolor='white', 
				   markeredgewidth=2, zorder=11)
			ax.text(rain_val + 0.03, y_val, f'μ={y_val:.3f}', 
				   fontsize=10, fontweight='bold', color=self.color_rain[term],
				   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
						   edgecolor=self.color_rain[term], linewidth=1.5))
		
		ax.set_title('Rainfall Fuzzification', fontweight='bold', fontsize=13, pad=12)
		ax.set_xlabel('Normalized Value', fontsize=11, fontweight='600')
		ax.set_ylabel('Membership Degree (μ)', fontsize=11, fontweight='600')
		ax.legend(fontsize=10, loc='best', framealpha=0.95, edgecolor='black')
		ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
		ax.set_ylim([0, 1.08])
		ax.set_facecolor('#FAFAFA')
		
		plt.tight_layout(pad=2.5)
		
		return fig

	def plot_imp(self, water_val, rate_val, rain_val):
		"""Plot implication - redesigned with better layout and clarity"""
		
		# Get active rules
		active_rules = self.evaluate_rules(water_val, rate_val, rain_val)
		
		if len(active_rules) == 0:
			print("No active rules to visualize!")
			fig, ax = plt.subplots(1, 1, figsize=(10, 6))
			ax.text(0.5, 0.5, 'No Active Rules', 
				   ha='center', va='center', fontsize=16, color='gray')
			ax.set_xlim([0, 100])
			ax.set_ylim([0, 1])
			ax.set_xlabel('Flood Risk (%)')
			ax.set_ylabel('Membership Degree (μ)')
			ax.grid(True, alpha=0.3)
			return fig
		
		# Calculate memberships for displaying antecedent values
		water_mem = self.calculate_memberships(self.water_level, self.water_mf, water_val)
		rate_mem = self.calculate_memberships(self.rate_change, self.rate_mf, rate_val)
		rain_mem = self.calculate_memberships(self.rainfall, self.rain_mf, rain_val)
		
		# Sort rules by output type for better organization
		output_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
		active_rules_sorted = sorted(active_rules, key=lambda x: (output_order[x['output']], -x['activation']))
		
		n_rules = len(active_rules_sorted)
		
		# Create smart layout: 1 column for vertical layout
		cols = 1
		rows = n_rules
		
		# Adjusted figure size for better proportions
		fig = plt.figure(figsize=(7 * cols, 4.5 * rows))
		
		print(f"\n=== Implication Step: {n_rules} Active Rules ===")
		
		for idx, rule in enumerate(active_rules_sorted):
			ax = plt.subplot(rows, cols, idx + 1)
			
			water_term = rule['water']
			rate_term = rule['rate']
			rain_term = rule['rain']
			output_term = rule['output']
			activation = rule['activation']
			
			# Plot original membership function with subtle styling
			ax.plot(self.flood_risk, self.risk_mf[output_term], 
				   linewidth=2, color='#CCCCCC', linestyle='-', 
				   label='Original MF', alpha=0.5, zorder=1)
			
			# Plot clipped output with prominent styling
			clipped_output = np.fmin(self.risk_mf[output_term], activation)
			ax.plot(self.flood_risk, clipped_output, 
				   linewidth=3.5, color=self.color_risk[output_term],
				   label=f'Clipped Output', zorder=3)
			ax.fill_between(self.flood_risk, 0, clipped_output,
						   alpha=0.35, color=self.color_risk[output_term], zorder=2)
			
			# Add firing strength line with better visibility
			ax.axhline(y=activation, color='#FF4444', linestyle='--', 
					  linewidth=2, alpha=0.8, zorder=4)
			
			# Add activation value annotation
			ax.text(95, activation, f'μ={activation:.3f}', 
				   verticalalignment='bottom', horizontalalignment='right',
				   fontsize=10, fontweight='bold', color='#FF4444',
				   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
						   edgecolor='#FF4444', alpha=0.9))
			
			# Create compact rule description
			antecedents = []
			if water_term:
				antecedents.append(f"W:{water_term}")
			if rate_term:
				antecedents.append(f"R:{rate_term}")
			if rain_term:
				antecedents.append(f"Rain:{rain_term}")
			
			rule_title = " & ".join(antecedents) if antecedents else "Priority Rule"
			
			# Set title with output risk prominently displayed
			title_color = self.color_risk[output_term]
			ax.set_title(f"Rule {idx + 1}: {rule_title}\n→ {output_term.upper()}", 
						fontweight='bold', fontsize=11, pad=12, color='#333333')
			
			# Add colored border to indicate output type
			for spine in ax.spines.values():
				spine.set_edgecolor(title_color)
				spine.set_linewidth(2.5)
			
			# Styling
			ax.set_xlabel('Flood Risk (%)', fontsize=10, fontweight='600')
			ax.set_ylabel('Membership (μ)', fontsize=10, fontweight='600')
			ax.legend(fontsize=8.5, loc='upper left', framealpha=0.95)
			ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
			ax.set_ylim([-0.02, 1.08])
			ax.set_xlim([0, 100])
			
			# Add subtle background color based on output type
			bg_colors = {'critical': '#FFE5E5', 'high': '#FFF0E5', 
						'medium': '#FFFACD', 'low': '#E5F5E5'}
			ax.set_facecolor(bg_colors.get(output_term, 'white'))
			
			# Print rule info
			antecedents_full = []
			if water_term:
				antecedents_full.append(f"water={water_term}({water_mem[water_term]:.3f})")
			if rate_term:
				antecedents_full.append(f"rate={rate_term}({rate_mem[rate_term]:.3f})")
			if rain_term:
				antecedents_full.append(f"rain={rain_term}({rain_mem[rain_term]:.3f})")
			
			print(f"  Rule {idx + 1}: {' AND '.join(antecedents_full) if antecedents_full else 'Priority'} "
				  f"→ {output_term} (μ={activation:.3f})")
		
		plt.tight_layout(pad=2.0)
		
		print(f"Total active rules visualized: {n_rules}\n")
		
		return fig
	
	def define_sample_rules(self):
		"""Define complete fuzzy rules matching main2.py"""
		rules = []
		
		# Format: (water_level, rate_change, rainfall, output_risk)
		
		# ========== PRIORITY RULES: Extreme rate regardless of water level ==========
		rules.append((None, 'naik_ekstrem', None, 'critical'))
		rules.append((None, 'naik_sangat_cepat', 'sangat_lebat', 'critical'))
		rules.append((None, 'naik_sangat_cepat', 'lebat', 'critical'))
		
		# ========== BANJIR PARAH LEVEL (0.75-1.0 normalized) ==========
		rules.append(('banjir', 'naik_ekstrem', None, 'critical'))
		rules.append(('banjir', 'naik_sangat_cepat', None, 'critical'))
		rules.append(('banjir', 'naik_cepat', None, 'critical'))
		rules.append(('banjir', 'naik_lambat', None, 'critical'))
		rules.append(('banjir', 'stabil', None, 'critical'))
		rules.append(('banjir', 'turun_lambat', None, 'critical'))
		rules.append(('banjir', 'turun_cepat', None, 'high'))
		rules.append(('banjir', 'turun_sangat_cepat', None, 'high'))
		
		# ========== BANJIR RINGAN LEVEL (0.5-0.85 normalized) ==========
		rules.append(('siaga_2', 'naik_ekstrem', None, 'critical'))
		rules.append(('siaga_2', 'naik_sangat_cepat', None, 'critical'))
		rules.append(('siaga_2', 'naik_cepat', None, 'critical'))
		rules.append(('siaga_2', 'naik_lambat', None, 'high'))
		rules.append(('siaga_2', 'stabil', None, 'high'))
		rules.append(('siaga_2', 'turun_lambat', None, 'high'))
		rules.append(('siaga_2', 'turun_cepat', None, 'high'))
		rules.append(('siaga_2', 'turun_sangat_cepat', None, 'medium'))
		
		# ========== SIAGA LEVEL (0.15-0.65 normalized) ==========
		rules.append(('siaga', 'naik_ekstrem', None, 'critical'))
		rules.append(('siaga', 'naik_sangat_cepat', 'sangat_lebat', 'critical'))
		rules.append(('siaga', 'naik_sangat_cepat', None, 'critical'))
		rules.append(('siaga', 'naik_cepat', 'lebat', 'critical'))
		rules.append(('siaga', 'naik_cepat', None, 'high'))
		rules.append(('siaga', 'naik_lambat', None, 'medium'))
		rules.append(('siaga', 'stabil', None, 'medium'))
		rules.append(('siaga', 'turun_lambat', None, 'medium'))
		rules.append(('siaga', 'turun_cepat', None, 'low'))
		rules.append(('siaga', 'turun_sangat_cepat', None, 'low'))
		
		# ========== NORMAL LEVEL (0-0.25 normalized) ==========
		rules.append(('normal', 'naik_ekstrem', None, 'critical'))
		rules.append(('normal', 'naik_sangat_cepat', 'sangat_lebat', 'critical'))
		rules.append(('normal', 'naik_sangat_cepat', None, 'high'))
		rules.append(('normal', 'naik_cepat', None, 'high'))
		rules.append(('normal', 'naik_lambat', None, 'low'))
		rules.append(('normal', 'stabil', None, 'low'))
		rules.append(('normal', 'turun_lambat', None, 'low'))
		rules.append(('normal', 'turun_cepat', None, 'low'))
		rules.append(('normal', 'turun_sangat_cepat', None, 'low'))
		
		return rules
	
	def evaluate_rules(self, water_val, rate_val, rain_val):
		"""Evaluate fuzzy rules and return active rules with their activation levels"""
		# Calculate memberships
		water_mem = self.calculate_memberships(self.water_level, self.water_mf, water_val)
		rate_mem = self.calculate_memberships(self.rate_change, self.rate_mf, rate_val)
		rain_mem = self.calculate_memberships(self.rainfall, self.rain_mf, rain_val)
		
		rules = self.define_sample_rules()
		active_rules = []
		
		for rule in rules:
			water_term, rate_term, rain_term, output_term = rule
			
			# Calculate rule activation (minimum of antecedents)
			activations = []
			
			# Check water level (if specified)
			if water_term is not None:
				if water_mem[water_term] > 0.01:
					activations.append(water_mem[water_term])
				else:
					continue  # Skip rule if water term not active
			
			# Check rate of change (if specified)
			if rate_term is not None:
				if rate_mem[rate_term] > 0.01:
					activations.append(rate_mem[rate_term])
				else:
					continue  # Skip rule if rate term not active
			
			# Check rainfall (if specified)
			if rain_term is not None:
				if rain_mem[rain_term] > 0.01:
					activations.append(rain_mem[rain_term])
				else:
					continue  # Skip rule if rain term not active
			
			# Calculate activation level (MIN of all antecedents)
			if len(activations) > 0:
				activation_level = min(activations)
				
				if activation_level > 0.01:
					active_rules.append({
						'water': water_term,
						'rate': rate_term,
						'rain': rain_term,
						'output': output_term,
						'activation': activation_level
					})
		
		return active_rules
	
	def plot_aggregation(self, water_val, rate_val, rain_val):
		"""Plot aggregation and defuzzification with improved design"""
		fig = plt.figure(figsize=(14, 7))
		
		# Create main plot and side info panel
		ax_main = plt.subplot(1, 1, 1)
		
		# Evaluate active rules
		active_rules = self.evaluate_rules(water_val, rate_val, rain_val)
		
		# Initialize aggregated output
		aggregated_output = np.zeros_like(self.flood_risk)
		
		# Aggregate outputs using MAX operator
		output_contributions = {'low': [], 'medium': [], 'high': [], 'critical': []}
		
		for rule in active_rules:
			output_term = rule['output']
			activation = rule['activation']
			clipped_mf = np.fmin(self.risk_mf[output_term], activation)
			output_contributions[output_term].append({
				'activation': activation,
				'clipped_mf': clipped_mf,
				'rule': rule
			})
			aggregated_output = np.fmax(aggregated_output, clipped_mf)
		
		# Plot individual clipped outputs with better visibility
		for output_term in ['low', 'medium', 'high', 'critical']:
			contributions = output_contributions[output_term]
			if contributions:
				max_activation = max(c['activation'] for c in contributions)
				max_clipped = np.zeros_like(self.flood_risk)
				for c in contributions:
					max_clipped = np.fmax(max_clipped, c['clipped_mf'])
				
				ax_main.plot(self.flood_risk, max_clipped, 
					   linewidth=2.5, color=self.color_risk[output_term], 
					   linestyle='-', alpha=0.7,
					   label=f'{output_term.upper()} (μ={max_activation:.3f})')
				ax_main.fill_between(self.flood_risk, 0, max_clipped,
							   alpha=0.25, color=self.color_risk[output_term])
		
		# Plot aggregated output with prominence
		ax_main.plot(self.flood_risk, aggregated_output, 
			   linewidth=4, color='#1a1a1a', zorder=10, alpha=0.9)
		ax_main.fill_between(self.flood_risk, 0, aggregated_output,
					   alpha=0.15, color='#000000', zorder=9)
		
		# Defuzzification
		if np.sum(aggregated_output) > 0:
			defuzzified_value = fuzz.defuzz(self.flood_risk, aggregated_output, 'centroid')


		else:
			defuzzified_value = 0
			ax_main.text(50, 0.5, 'No active rules', fontsize=14, 
				   ha='center', color='gray')
		
		# Styling for main plot
		ax_main.set_xlabel('Flood Risk (%)', fontsize=12, fontweight='600')
		ax_main.set_ylabel('Membership Degree (μ)', fontsize=12, fontweight='600')
		ax_main.legend(fontsize=10, loc='upper left', framealpha=0.95, 
					  edgecolor='black', fancybox=True)
		ax_main.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
		ax_main.set_ylim([0, 1.08])
		ax_main.set_xlim([0, 100])
		ax_main.set_facecolor('#FAFAFA')
		
		plt.tight_layout()
		return fig, defuzzified_value
	
	def plot_centroid_calculation(self, water_val, rate_val, rain_val):
		"""Detailed visualization of centroid defuzzification calculation"""
		fig = plt.figure(figsize=(16, 10))
		
		# Create grid layout
		gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.35, wspace=0.3)
		
		# Main plot - Aggregated function with centroid
		ax_main = fig.add_subplot(gs[0, :])
		
		# Bottom left - Area calculation visualization
		ax_area = fig.add_subplot(gs[1, 0])
		
		# Bottom right - Moment calculation visualization
		ax_moment = fig.add_subplot(gs[1, 1])
		
		# Bottom - Mathematical formula panel
		ax_formula = fig.add_subplot(gs[2, :])
		ax_formula.axis('off')
		
		# Evaluate active rules and get aggregated output
		active_rules = self.evaluate_rules(water_val, rate_val, rain_val)
		aggregated_output = np.zeros_like(self.flood_risk)
		
		for rule in active_rules:
			output_term = rule['output']
			activation = rule['activation']
			clipped_mf = np.fmin(self.risk_mf[output_term], activation)
			aggregated_output = np.fmax(aggregated_output, clipped_mf)
		
		# Calculate centroid
		if np.sum(aggregated_output) > 0:
			# Centroid formula: CoG = Σ(x * μ(x)) / Σ(μ(x))
			numerator = np.sum(self.flood_risk * aggregated_output)
			denominator = np.sum(aggregated_output)
			centroid = numerator / denominator
			
			# Also calculate using fuzz.defuzz for verification
			centroid_verify = fuzz.defuzz(self.flood_risk, aggregated_output, 'centroid')
			
			# === MAIN PLOT ===
			# Plot aggregated function
			ax_main.plot(self.flood_risk, aggregated_output, 
					   linewidth=4, color='#2E86AB', label='Aggregated Output', zorder=3)
			ax_main.fill_between(self.flood_risk, 0, aggregated_output,
						   alpha=0.35, color='#2E86AB', zorder=2)
			
			# Draw centroid line
			ax_main.axvline(centroid, color='#FF0000', linestyle='-', 
					  linewidth=4, label=f'Centroid (CoG)', zorder=5)
			
			# Mark the balance point
			y_centroid = fuzz.interp_membership(self.flood_risk, aggregated_output, centroid)
			ax_main.plot(centroid, y_centroid, 'o', markersize=20, 
					   color='#FF0000', markeredgecolor='white', 
					   markeredgewidth=3, zorder=6)
			
			# Add detailed annotation
			ax_main.annotate(f'Center of Gravity\n{centroid:.2f}%', 
					   xy=(centroid, y_centroid),
					   xytext=(centroid + 15, y_centroid + 0.15),
					   fontsize=14, fontweight='bold', color='#FF0000',
					   bbox=dict(boxstyle='round,pad=0.8', facecolor='white', 
						   edgecolor='#FF0000', linewidth=3),
					   arrowprops=dict(arrowstyle='->', color='#FF0000', lw=3))
			
			# Visual representation of "balance"
			# Draw support triangle at bottom
			triangle_x = [centroid - 5, centroid + 5, centroid, centroid - 5]
			triangle_y = [-0.08, -0.08, -0.02, -0.08]
			ax_main.fill(triangle_x, triangle_y, color='#FF0000', alpha=0.7, zorder=7)
			ax_main.plot([centroid, centroid], [-0.02, 0], 'r-', linewidth=3, zorder=7)
			
			ax_main.set_title('Centroid Defuzzification: Finding the "Balance Point"', 
						fontweight='bold', fontsize=16, pad=20)
			ax_main.set_xlabel('Flood Risk (%)', fontsize=13, fontweight='600')
			ax_main.set_ylabel('Membership Degree (μ)', fontsize=13, fontweight='600')
			ax_main.legend(fontsize=12, loc='upper left', framealpha=0.95, 
						  edgecolor='black', fancybox=True)
			ax_main.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
			ax_main.set_ylim([-0.12, 1.1])
			ax_main.set_xlim([0, 100])
			ax_main.set_facecolor('#FAFAFA')
			
			# === AREA CALCULATION (Denominator) ===
			ax_area.fill_between(self.flood_risk, 0, aggregated_output,
							   alpha=0.5, color='#4CAF50', label='Area under curve')
			ax_area.plot(self.flood_risk, aggregated_output, 
					   linewidth=2, color='#2E7D32')
			
			# Show sample rectangles for Riemann sum visualization
			sample_points = self.flood_risk[::10]  # Sample every 10 points
			for x in sample_points:
				y = fuzz.interp_membership(self.flood_risk, aggregated_output, x)
				if y > 0.01:
					ax_area.plot([x, x], [0, y], 'k-', alpha=0.2, linewidth=1)
			
			ax_area.text(0.5, 0.95, f'Area = Σμ(x) = {denominator:.2f}', 
					   transform=ax_area.transAxes, ha='center', va='top',
					   fontsize=12, fontweight='bold',
					   bbox=dict(boxstyle='round,pad=0.5', facecolor='#4CAF50', 
						   edgecolor='black', alpha=0.3))
			
			ax_area.set_title('Step 1: Calculate Total Area', fontweight='bold', fontsize=12)
			ax_area.set_xlabel('x (Risk %)', fontsize=10)
			ax_area.set_ylabel('μ(x)', fontsize=10)
			ax_area.grid(True, alpha=0.25)
			ax_area.set_facecolor('#FAFAFA')
			
			# === MOMENT CALCULATION (Numerator) ===
			# Weighted area visualization
			weighted_contribution = self.flood_risk * aggregated_output
			ax_moment.fill_between(self.flood_risk, 0, weighted_contribution,
								 alpha=0.5, color='#FF9800', label='x × μ(x)')
			ax_moment.plot(self.flood_risk, weighted_contribution, 
					   linewidth=2, color='#F57C00')
			
			# Mark centroid position
			ax_moment.axvline(centroid, color='#FF0000', linestyle='--', 
						  linewidth=2, alpha=0.7)
			
			ax_moment.text(0.5, 0.95, f'Moment = Σ(x·μ(x)) = {numerator:.2f}', 
					   transform=ax_moment.transAxes, ha='center', va='top',
					   fontsize=12, fontweight='bold',
					   bbox=dict(boxstyle='round,pad=0.5', facecolor='#FF9800', 
						   edgecolor='black', alpha=0.3))
			
			ax_moment.set_title('Step 2: Calculate Weighted Moment', fontweight='bold', fontsize=12)
			ax_moment.set_xlabel('x (Risk %)', fontsize=10)
			ax_moment.set_ylabel('x × μ(x)', fontsize=10)
			ax_moment.grid(True, alpha=0.25)
			ax_moment.set_facecolor('#FAFAFA')
			
			# === FORMULA PANEL ===
			y_pos = 0.85
			
			# Title
			ax_formula.text(0.5, y_pos, 'Centroid Defuzzification Formula', 
						  ha='center', va='top', fontsize=15, fontweight='bold')
			y_pos -= 0.18
			
			# Mathematical formula
			formula_text = r'$CoG = \frac{\sum_{i=1}^{n} x_i \cdot \mu(x_i)}{\sum_{i=1}^{n} \mu(x_i)}$'

			ax_formula.text(0.5, y_pos, formula_text, 
						  ha='center', va='top', fontsize=20)
			y_pos -= 0.25
			
			# Explanation
			explanation = (
				"Where:\n"
				f"• xᵢ = Risk values (0 to 100%)\n"
				f"• μ(xᵢ) = Membership degree at each point\n"
				f"• CoG = Center of Gravity (balance point)"
			)
			ax_formula.text(0.1, y_pos, explanation, 
						  ha='left', va='top', fontsize=11,
						  bbox=dict(boxstyle='round,pad=0.8', facecolor='#E3F2FD', 
							  edgecolor='#1976D2', linewidth=2))
			
			# Calculation steps
			calculation = (
				f"Calculation:\n"
				f"1. Numerator = Σ(x·μ(x)) = {numerator:.2f}\n"
				f"2. Denominator = Σμ(x) = {denominator:.2f}\n"
				f"3. CoG = {numerator:.2f} / {denominator:.2f}\n"
				f"4. Result = {centroid:.2f}%"
			)
			ax_formula.text(0.6, y_pos, calculation, 
						  ha='left', va='top', fontsize=11, family='monospace',
						  bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF3E0', 
							  edgecolor='#F57C00', linewidth=2))
			
			# Interpretation
			if centroid >= 85:
				interpretation = "⚠️ CRITICAL RISK - Immediate action required!"
				interp_color = '#9D0208'
			elif centroid >= 60:
				interpretation = "⚠️ HIGH RISK - Prepare for potential flooding"
				interp_color = '#EF476F'
			elif centroid >= 25:
				interpretation = "⚠️ MEDIUM RISK - Monitor situation closely"
				interp_color = '#FFD166'
			else:
				interpretation = "✓ LOW RISK - Situation under control"
				interp_color = '#06D6A0'
			
			ax_formula.text(0.5, 0.05, interpretation, 
						  ha='center', va='bottom', fontsize=13, fontweight='bold',
						  color=interp_color,
						  bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
							  edgecolor=interp_color, linewidth=3))
			
			print(f"\n{'='*60}")
			print(f"CENTROID DEFUZZIFICATION CALCULATION")
			print(f"{'='*60}")
			print(f"Numerator (Moment):   Σ(x·μ(x)) = {numerator:.4f}")
			print(f"Denominator (Area):   Σμ(x)     = {denominator:.4f}")
			print(f"Centroid (CoG):       {numerator:.4f} / {denominator:.4f}")
			print(f"Final Result:         {centroid:.2f}%")
			print(f"Verification:         {centroid_verify:.2f}% (using fuzz.defuzz)")
			print(f"{'='*60}\n")
			
		else:
			ax_main.text(0.5, 0.5, 'No active rules - cannot calculate centroid', 
					   ha='center', va='center', fontsize=16, color='gray',
					   transform=ax_main.transAxes)
			centroid = 0
		
		plt.suptitle('Understanding Centroid Defuzzification Method', 
				   fontsize=18, fontweight='bold', y=0.98)
		
		return fig, centroid
		
if __name__ == "__main__":
	print("\n" + "=" * 60)
	print("FUZZY INFERENCE SYSTEM VISUALIZER")
	print("Flood Risk Assessment")
	print("=" * 60)
	
	visualizer = CleanFuzzyVisualizer()
	
	# Input values
	water_input = 0.45
	rate_input = 0.7
	rain_input = 0.85

	# Generate all visualizations
	defuzz_output = visualizer.visualize_all_steps(water_input, rate_input, rain_input)
	
	print("\n" + "=" * 60)
	print("FINAL RESULT")
	print("=" * 60)
	print(f"Input Values:")
	print(f"  • Water Level (normalized): {water_input}")
	print(f"  • Rate of Change (normalized): {rate_input}")
	print(f"  • Rainfall (normalized): {rain_input}")
	print(f"\nOutput:")
	print(f"  • Flood Risk Score: {defuzz_output:.2f}%")
	
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
	
	print(f"  • Risk Category: {color} {risk_category}")
	print("=" * 60)
	print("\n✓ All visualizations saved successfully!")
	print("  - step1_fuzzification.png")
	print("  - step2_implication.png")
	print("  - step3_aggregation_defuzzification.png")
	print("\n")
	
	plt.show()