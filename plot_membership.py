import matplotlib.pyplot as plt

def plot_all_membership(system, save_path=None):
	if system.fuzzy_system is None:
		raise ValueError("System not calibrated. Call calibrate() first.")
	
	fig, axes = plt.subplots(2, 2, figsize=(16, 12))
	fig.suptitle('Fuzzy Flood Warning System - Membership Functions', fontsize=16, fontweight='bold')
	
	water_level = avg_rate_change = rainfall = flood_risk = None
	for var in system.fuzzy_system.ctrl.fuzzy_variables:
		if var.label == 'water_level': water_level = var
		elif var.label == 'avg_rate_change': avg_rate_change = var
		elif var.label == 'rainfall': rainfall = var
		elif var.label == 'flood_risk': flood_risk = var
	
	# Water Level
	ax1 = axes[0, 0]
	for label in water_level.terms:
		ax1.plot(water_level.universe, water_level[label].mf, linewidth=2, label=label)
	ax1.axvline(x=system.siaga_level, color='orange', linestyle='--', linewidth=2, label=f'Siaga ({system.siaga_level}cm)')
	ax1.axvline(x=system.banjir_level, color='red', linestyle='--', linewidth=2, label=f'Banjir ({system.banjir_level}cm)')
	ax1.set_title('Water Level', fontweight='bold', fontsize=12)
	ax1.set_xlabel('Distance (cm)', fontsize=10)
	ax1.set_ylabel('Membership', fontsize=10)
	ax1.legend(loc='best', fontsize=9)
	ax1.grid(True, alpha=0.3)
	
	# Rate of Change
	ax2 = axes[0, 1]
	for label in avg_rate_change.terms:
		ax2.plot(avg_rate_change.universe, avg_rate_change[label].mf, linewidth=2, label=label)
	ax2.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
	ax2.axvline(x=0.67, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='Guideline (±0.67)')
	ax2.axvline(x=-0.67, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
	ax2.set_title('Rate of Change (Guideline: 0.67 cm/min)', fontweight='bold', fontsize=12)
	ax2.set_xlabel('Rate (cm/min)', fontsize=10)
	ax2.set_ylabel('Membership', fontsize=10)
	ax2.legend(loc='best', fontsize=8)
	ax2.grid(True, alpha=0.3)
	
	# Rainfall
	ax3 = axes[1, 0]
	for label in rainfall.terms:
		ax3.plot(rainfall.universe, rainfall[label].mf, linewidth=2, label=label)
	for val in [1, 5, 10, 20]:
		ax3.axvline(x=val, color='gray', linestyle=':', linewidth=1, alpha=0.5)
	ax3.set_title('Rainfall (BMKG)', fontweight='bold', fontsize=12)
	ax3.set_xlabel('Rainfall (mm/h)', fontsize=10)
	ax3.set_ylabel('Membership', fontsize=10)
	ax3.legend(loc='best', fontsize=9)
	ax3.grid(True, alpha=0.3)
	
	# Flood Risk
	ax4 = axes[1, 1]
	for label in flood_risk.terms:
		ax4.plot(flood_risk.universe, flood_risk[label].mf, linewidth=2, label=label)
	ax4.set_title('Flood Risk Output', fontweight='bold', fontsize=12)
	ax4.set_xlabel('Risk (%)', fontsize=10)
	ax4.set_ylabel('Membership', fontsize=10)
	ax4.legend(loc='best', fontsize=9)
	ax4.grid(True, alpha=0.3)
	
	plt.tight_layout()
	if save_path:
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"Plot saved to: {save_path}")
	else:
		plt.show()

def plot_single_membership(system, variable_name, save_path=None):
	if system.fuzzy_system is None:
		raise ValueError("System not calibrated. Call calibrate() first.")
	
	target_var = None
	for var in system.fuzzy_system.ctrl.fuzzy_variables:
		if var.label == variable_name:
			target_var = var
			break
	
	if target_var is None:
		raise ValueError(f"Variable '{variable_name}' not found. Choose: 'water_level', 'avg_rate_change', 'rainfall', 'flood_risk'")
	
	plt.figure(figsize=(12, 6))
	
	for label in target_var.terms:
		plt.plot(target_var.universe, target_var[label].mf, linewidth=2.5, label=label)
	
	if variable_name == 'water_level':
		plt.axvline(x=system.siaga_level, color='orange', linestyle='--', linewidth=2, label=f'Siaga ({system.siaga_level}cm)')
		plt.axvline(x=system.banjir_level, color='red', linestyle='--', linewidth=2, label=f'Banjir ({system.banjir_level}cm)')
		plt.xlabel('Distance (cm)', fontsize=12)
		plt.title('Water Level Membership Functions', fontsize=14, fontweight='bold')
	elif variable_name == 'avg_rate_change':
		plt.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
		plt.axvline(x=0.67, color='green', linestyle=':', linewidth=2, alpha=0.7, label='Guideline (±0.67 cm/min)')
		plt.axvline(x=-0.67, color='green', linestyle=':', linewidth=2, alpha=0.7)
		plt.xlabel('Rate (cm/min)', fontsize=12)
		plt.title('Rate of Change (Guideline: 0.67 cm/min)', fontsize=14, fontweight='bold')
	elif variable_name == 'rainfall':
		for val in [1, 5, 10, 20]:
			plt.axvline(x=val, color='gray', linestyle=':', linewidth=1.5, alpha=0.6, label='BMKG' if val==1 else None)
		plt.xlabel('Rainfall (mm/h)', fontsize=12)
		plt.title('Rainfall (BMKG Guidelines)', fontsize=14, fontweight='bold')
	elif variable_name == 'flood_risk':
		plt.xlabel('Risk (%)', fontsize=12)
		plt.title('Flood Risk Output', fontsize=14, fontweight='bold')
	
	plt.ylabel('Membership Degree', fontsize=12)
	plt.legend(loc='best', fontsize=10)
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	
	if save_path:
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"Plot saved to: {save_path}")
	else:
		plt.show()

if __name__ == "__main__":
	from main2 import DynamicFuzzyFloodWarningSystem
	
	system = DynamicFuzzyFloodWarningSystem(reading_interval_seconds=1)
	system.calibrate(ground_distance=100, siaga_level_override=130, banjir_level_override=100)
	
	print("Generating plots...")
	plot_all_membership(system)