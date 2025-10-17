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
            'banjir_ringan': fuzz.trimf(self.water_level, [0.5, 0.7, 0.85]),
            'banjir_parah': fuzz.trapmf(self.water_level, [0.75, 0.9, 1.0, 1.0])
        }
        
        # Rate of Change membership functions
        self.rate_mf = {
            'turun_cepat': fuzz.trapmf(self.rate_change, [-1, -1, -0.5, -0.2]),
            'turun_lambat': fuzz.trimf(self.rate_change, [-0.3, -0.1, -0.03]),
            'stabil': fuzz.trimf(self.rate_change, [-0.05, 0, 0.05]),
            'naik_lambat': fuzz.trimf(self.rate_change, [0.03, 0.1, 0.2]),
            'naik_cepat': fuzz.trimf(self.rate_change, [0.15, 0.35, 0.6]),
            'naik_sangat_cepat': fuzz.trimf(self.rate_change, [0.5, 0.75, 0.9]),
            'naik_ekstrem': fuzz.trapmf(self.rate_change, [0.85, 0.95, 1, 1])
        }
        
        # Rainfall membership functions
        self.rain_mf = {
            'tidak_hujan': fuzz.trapmf(self.rainfall, [0, 0, 0.02, 0.04]),
            'ringan': fuzz.trimf(self.rainfall, [0.02, 0.1, 0.2]),
            'sedang': fuzz.trimf(self.rainfall, [0.15, 0.3, 0.45]),
            'lebat': fuzz.trimf(self.rainfall, [0.35, 0.55, 0.7]),
            'sangat_lebat': fuzz.trimf(self.rainfall, [0.6, 0.75, 0.88]),
            'ekstrem': fuzz.trapmf(self.rainfall, [0.8, 0.92, 1.0, 1.0])
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
                           'banjir_ringan': '#F18F01', 'banjir_parah': '#C73E1D'}
        self.color_rate = {'turun_cepat': '#1B4965', 'turun_lambat': '#5FA8D3',
                          'stabil': '#62B6CB', 'naik_lambat': '#CAE9FF',
                          'naik_cepat': '#FFB627', 'naik_sangat_cepat': '#FF6B35',
                          'naik_ekstrem': '#C1121F'}
        self.color_rain = {'tidak_hujan': '#B8E0D2', 'ringan': '#95D5B2',
                          'sedang': '#74C69D', 'lebat': '#52B788',
                          'sangat_lebat': '#40916C', 'ekstrem': '#2D6A4F'}
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
        print(f"\nGenerating visualizations for:")
        print(f"  Water Level: {water_val}")
        print(f"  Rate Change: {rate_val}")
        print(f"  Rainfall: {rain_val}\n")
        
        # Step 1: Fuzzification
        fig = self.plot_fuzzification_clean(
            water_val, rate_val, rain_val)
        fig.savefig('fuzzification.png')
        
        # Step 2: Clipping
        fig2 = self.plot_imp(water_val, rate_val, rain_val)
        fig2.savefig('clipping.png')

    def plot_fuzzification_clean(self, water_val, rate_val, rain_val):
        """Plot clean fuzzification - only active memberships"""
        fig, axes = plt.subplots(3, 1, figsize=(6, 12))
        
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
                   label=f'{term} (μ={water_mem[term]:.3f})',
                   linewidth=2.5, color=self.color_water[term])
            ax.fill_between(self.water_level, 0, self.water_mf[term], 
                           alpha=0.2, color=self.color_water[term])
        
        ax.axvline(water_val, color='red', linestyle='--', linewidth=2, label=f'Input={water_val:.2f}')
        ax.set_title('Water Level', fontweight='bold', fontsize=12)
        ax.set_xlabel('Normalized Value')
        ax.set_ylabel('Membership Degree (μ)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Plot Rate of Change
        ax = axes[1]
        for term in active_rate.keys():
            ax.plot(self.rate_change, self.rate_mf[term], 
                   label=f'{term} (μ={rate_mem[term]:.3f})',
                   linewidth=2.5, color=self.color_rate[term])
            ax.fill_between(self.rate_change, 0, self.rate_mf[term], 
                           alpha=0.2, color=self.color_rate[term])
        
        ax.axvline(rate_val, color='red', linestyle='--', linewidth=2, label=f'Input={rate_val:.2f}')
        ax.set_title('Rate of Change', fontweight='bold', fontsize=12)
        ax.set_xlabel('Normalized Value')
        ax.set_ylabel('Membership Degree (μ)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Plot Rainfall
        ax = axes[2]
        for term in active_rain.keys():
            ax.plot(self.rainfall, self.rain_mf[term], 
                   label=f'{term} (μ={rain_mem[term]:.3f})',
                   linewidth=2.5, color=self.color_rain[term])
            ax.fill_between(self.rainfall, 0, self.rain_mf[term], 
                           alpha=0.2, color=self.color_rain[term])
        
        ax.axvline(rain_val, color='red', linestyle='--', linewidth=2, label=f'Input={rain_val:.2f}')
        ax.set_title('Rainfall', fontweight='bold', fontsize=12)
        ax.set_xlabel('Normalized Value')
        ax.set_ylabel('Membership Degree (μ)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        return fig

    def plot_imp(self, water_val, rate_val, rain_val):
        """Plot implication - clipping of membership functions"""
        fig, axes = plt.subplots(3, 1, figsize=(6, 16))
        
        # Calculate memberships
        water_mem = self.calculate_memberships(self.water_level, self.water_mf, water_val)
        rate_mem = self.calculate_memberships(self.rate_change, self.rate_mf, rate_val)
        rain_mem = self.calculate_memberships(self.rainfall, self.rain_mf, rain_val)
        
        # Get active memberships (> 0.01)
        active_water = {k: v for k, v in water_mem.items() if v > 0.01}
        active_rate = {k: v for k, v in rate_mem.items() if v > 0.01}
        active_rain = {k: v for k, v in rain_mem.items() if v > 0.01}
        
        # Calculate representative clip values for each input FIRST
        # For each input: if multiple memberships are active, take the HIGHEST
        water_representative = max(active_water.values()) if active_water else 0
        rate_representative = max(active_rate.values()) if active_rate else 0
        rain_representative = max(active_rain.values()) if active_rain else 0
        
        # Take the MINIMUM of the three representatives
        min_clip = min(water_representative, rate_representative, rain_representative)
        
        print(f"\nClip value selection:")
        print(f"  Water representative (max of active): {water_representative:.3f}")
        print(f"  Rate representative (max of active): {rate_representative:.3f}")
        print(f"  Rain representative (max of active): {rain_representative:.3f}")
        print(f"  Final clip value (min of representatives): {min_clip:.3f}\n")
        
        # Plot Water Level - ALL clipped at min_clip
        ax = axes[0]
        for term in active_water.keys():
            individual_membership = water_mem[term]
            ax.plot(self.water_level, np.fmin(self.water_mf[term], min_clip), 
                    label=f'{term} (μ={individual_membership:.2f})',
                    linewidth=2.5, color=self.color_water[term])
            ax.fill_between(self.water_level, 0, np.fmin(self.water_mf[term], min_clip), 
                            alpha=0.2, color=self.color_water[term])
        ax.set_title(f'Water Level Clipping (clipped at μ={min_clip:.2f})', fontweight='bold', fontsize=12)
        ax.set_xlabel('Normalized Value')
        ax.set_ylabel('Membership Degree (μ)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Plot Rate of Change - ALL clipped at min_clip
        ax = axes[1]
        for term in active_rate.keys():
            individual_membership = rate_mem[term]
            ax.plot(self.rate_change, np.fmin(self.rate_mf[term], min_clip), 
                    label=f'{term} (μ={individual_membership:.2f})',
                    linewidth=2.5, color=self.color_rate[term])
            ax.fill_between(self.rate_change, 0, np.fmin(self.rate_mf[term], min_clip), 
                            alpha=0.2, color=self.color_rate[term])
        ax.set_title(f'Rate of Change Clipping (clipped at μ={min_clip:.2f})', fontweight='bold', fontsize=12)
        ax.set_xlabel('Normalized Value')
        ax.set_ylabel('Membership Degree (μ)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Plot Rainfall - ALL clipped at min_clip
        ax = axes[2]
        for term in active_rain.keys():
            individual_membership = rain_mem[term]
            ax.plot(self.rainfall, np.fmin(self.rain_mf[term], min_clip), 
                    label=f'{term} (μ={individual_membership:.2f})',
                    linewidth=2.5, color=self.color_rain[term])
            ax.fill_between(self.rainfall, 0, np.fmin(self.rain_mf[term], min_clip), 
                            alpha=0.2, color=self.color_rain[term])
        ax.set_title(f'Rainfall Clipping (clipped at μ={min_clip:.2f})', fontweight='bold', fontsize=12)
        ax.set_xlabel('Normalized Value')
        ax.set_ylabel('Membership Degree (μ)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        # Create a combined figure showing all clipped inputs in one plot
        fig_combined, ax = plt.subplots(figsize=(10, 6))
        
        # Plot Water Level clipped memberships
        for term in active_water.keys():
            ax.plot(self.water_level, np.fmin(self.water_mf[term], min_clip), 
                    label=f'Water: {term}',
                    linewidth=2.5, color=self.color_water[term])
            ax.fill_between(self.water_level, 0, np.fmin(self.water_mf[term], min_clip), 
                            alpha=0.15, color=self.color_water[term])
        
        # Plot Rate of Change clipped memberships (scaled to 0-1 range)
        for term in active_rate.keys():
            # Scale rate_change from [-1, 1] to [0, 1] for combined visualization
            rate_scaled_x = (self.rate_change + 1) / 2
            ax.plot(rate_scaled_x, np.fmin(self.rate_mf[term], min_clip), 
                    label=f'Rate: {term}',
                    linewidth=2.5, color=self.color_rate[term], linestyle='--')
            ax.fill_between(rate_scaled_x, 0, np.fmin(self.rate_mf[term], min_clip), 
                            alpha=0.15, color=self.color_rate[term])
        
        # Plot Rainfall clipped memberships
        for term in active_rain.keys():
            ax.plot(self.rainfall, np.fmin(self.rain_mf[term], min_clip), 
                    label=f'Rain: {term}',
                    linewidth=2.5, color=self.color_rain[term], linestyle=':')
            ax.fill_between(self.rainfall, 0, np.fmin(self.rain_mf[term], min_clip), 
                            alpha=0.15, color=self.color_rain[term])
        
        # Add horizontal line at min_clip level
        ax.axhline(min_clip, color='red', linestyle='-', linewidth=2.5, 
                label=f'Clip level = {min_clip:.2f}', alpha=0.8, zorder=10)
        
        ax.set_title(f'Combined View: All Clipped Memberships (clipped at μ={min_clip:.2f})', 
                    fontweight='bold', fontsize=13)
        ax.set_xlabel('Normalized Input Scale (0-1)')
        ax.set_ylabel('Membership Degree (μ)')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        ax.set_xlim([0, 1])
        
        fig_combined.tight_layout()
        fig_combined.savefig('combined_clipping.png')
        
        return fig

# Main execution
if __name__ == "__main__":
    visualizer = CleanFuzzyVisualizer()
    
    # Input values
    water_input = 0.45
    rate_input = 0.7
    rain_input = 0.85

    # Generate all visualizations
    result = visualizer.visualize_all_steps(water_input, rate_input, rain_input)
    
    plt.show()