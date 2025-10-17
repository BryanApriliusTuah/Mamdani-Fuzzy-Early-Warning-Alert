import numpy as np
import pandas as pd

class FloodScenarioGenerator:
    """Generate realistic flood warning scenarios for system testing"""
    
    def __init__(self, ground_distance=100, siaga_level=130, banjir_level=100):
        self.ground_distance = ground_distance
        self.siaga_level = siaga_level
        self.banjir_level = banjir_level
        
    def generate_scenario(self, scenario_type):
        """Generate a single scenario based on type"""
        
        if scenario_type == "Normal_Stabil":
            # Normal water level, stable or slowly changing
            distance = np.random.uniform(self.siaga_level + 10, self.siaga_level + 50)
            rate_change = np.random.uniform(-0.5, 0.5)
            rainfall = np.random.uniform(0, 3)
            expected_label = "NORMAL"
            
        elif scenario_type == "Normal_Naik_Lambat":
            # Normal level but water rising slowly
            distance = np.random.uniform(self.siaga_level + 5, self.siaga_level + 30)
            rate_change = np.random.uniform(-5, -1)
            rainfall = np.random.uniform(2, 8)
            expected_label = "NORMAL"
            
        elif scenario_type == "Normal_Naik_Cepat":
            # Normal level but water rising fast (should trigger warning)
            distance = np.random.uniform(self.siaga_level + 5, self.siaga_level + 20)
            rate_change = np.random.uniform(-15, -8)
            rainfall = np.random.uniform(10, 20)
            expected_label = "SIAGA"  # Fast rise should trigger alert
            
        elif scenario_type == "Siaga_Stabil":
            # At siaga level, stable
            distance = np.random.uniform(self.banjir_level + 5, self.siaga_level - 5)
            rate_change = np.random.uniform(-2, 2)
            rainfall = np.random.uniform(0, 5)
            expected_label = "SIAGA"
            
        elif scenario_type == "Siaga_Naik_Lambat":
            # At siaga level, rising slowly
            distance = np.random.uniform(self.banjir_level + 10, self.siaga_level - 5)
            rate_change = np.random.uniform(-8, -3)
            rainfall = np.random.uniform(5, 12)
            expected_label = "SIAGA"
            
        elif scenario_type == "Siaga_Naik_Cepat":
            # At siaga level, rising fast (critical)
            distance = np.random.uniform(self.banjir_level + 5, self.siaga_level - 5)
            rate_change = np.random.uniform(-20, -10)
            rainfall = np.random.uniform(12, 25)
            expected_label = "BANJIR"  # Fast rise at siaga = flood imminent
            
        elif scenario_type == "Siaga_Menurun":
            # At siaga level, water dropping (recovery)
            distance = np.random.uniform(self.banjir_level + 10, self.siaga_level - 5)
            rate_change = np.random.uniform(3, 10)
            rainfall = np.random.uniform(0, 3)
            expected_label = "SIAGA"  # Still in siaga zone
            
        elif scenario_type == "Banjir_Stabil":
            # At flood level, stable
            distance = np.random.uniform(self.banjir_level - 10, self.banjir_level + 3)
            rate_change = np.random.uniform(-2, 2)
            rainfall = np.random.uniform(0, 8)
            expected_label = "BANJIR"
            
        elif scenario_type == "Banjir_Naik":
            # At flood level, still rising
            distance = np.random.uniform(self.banjir_level - 15, self.banjir_level + 2)
            rate_change = np.random.uniform(-15, -3)
            rainfall = np.random.uniform(5, 25)
            expected_label = "BANJIR"
            
        elif scenario_type == "Banjir_Turun_Lambat":
            # At flood level, dropping slowly
            distance = np.random.uniform(self.banjir_level - 5, self.banjir_level + 3)
            rate_change = np.random.uniform(2, 8)
            rainfall = np.random.uniform(0, 3)
            expected_label = "BANJIR"  # Still at flood level
            
        elif scenario_type == "Banjir_Turun_Cepat":
            # At flood level, dropping fast
            distance = np.random.uniform(self.banjir_level - 3, self.banjir_level + 5)
            rate_change = np.random.uniform(8, 20)
            rainfall = np.random.uniform(0, 2)
            expected_label = "BANJIR"  # Still technically at flood level
            
        elif scenario_type == "Normal → Banjir":
            # Extreme water rise scenario
            distance = np.random.uniform(self.banjir_level - 10, self.siaga_level + 10)
            rate_change = np.random.uniform(-30, -20)
            rainfall = np.random.uniform(20, 30)
            expected_label = "BANJIR"  # Extreme rise = flood warning
            
        else:  # RECOVERY scenario
            # Water dropped back to normal
            distance = np.random.uniform(self.banjir_level - 10, self.siaga_level + 30)
            rate_change = np.random.uniform(5, 15)
            rainfall = np.random.uniform(0, 2)
            expected_label = "NORMAL"
        
        return {
            'distance_cm': round(distance, 2),
            'rate_change_cm_per_min': round(rate_change, 2),
            'rainfall_mm_per_hour': round(rainfall, 2),
            'expected_warning': expected_label,
            'scenario_type': scenario_type
        }
    
    def generate_dataset(self, total_scenarios=1000):
        """Generate a balanced dataset of scenarios"""
        
        # Define scenario distribution
        scenario_types = [
            "Normal_Stabil",           # 20%
            "Normal_Naik_Lambat",      # 15%
            "Normal_Naik_Cepat",      # 8%
            "Siaga_Stabil",            # 12%
            "Siaga_Naik_Lambat",       # 10%
            "Siaga_Naik_Cepat",       # 8%
            "Siaga_Menurun",          # 7%
            "Banjir_Stabil",           # 8%
            "Banjir_Naik",           # 7%
            "Banjir_Turun_Lambat",    # 3%
            "Banjir_Turun_Cepat",    # 2%
            "Normal → Banjir",            # 5%
            "RECOVERY"                 # 5%
        ]
        
        # Calculate number of scenarios per type
        distribution = [0.20, 0.15, 0.08, 0.12, 0.10, 0.08, 0.07, 0.08, 0.07, 0.03, 0.02, 0.05, 0.05]
        scenarios_per_type = [int(total_scenarios * dist) for dist in distribution]
        
        # Adjust last element to ensure exact total
        scenarios_per_type[-1] = total_scenarios - sum(scenarios_per_type[:-1])
        
        # Generate scenarios
        all_scenarios = []
        scenario_id = 1
        
        for scenario_type, count in zip(scenario_types, scenarios_per_type):
            for _ in range(count):
                scenario = self.generate_scenario(scenario_type)
                scenario['scenario_id'] = scenario_id
                all_scenarios.append(scenario)
                scenario_id += 1
        
        # Shuffle scenarios
        np.random.shuffle(all_scenarios)
        
        # Create DataFrame
        df = pd.DataFrame(all_scenarios)
        
        # Reorder columns
        df = df[['scenario_id', 'distance_cm', 'rate_change_cm_per_min', 
                'rainfall_mm_per_hour', 'expected_warning', 'scenario_type']]
        
        return df
    
    def save_dataset(self, df, filename_base='flood_scenarios'):
        """Save dataset in multiple formats"""
                
        # Save as CSV
        csv_file = f'{filename_base}.csv'
        df.to_csv(csv_file, index=False)
        print(f"✓ Saved CSV: {csv_file}")
        
        # Save as JSON
        json_file = f'{filename_base}.json'
        df.to_json(json_file, orient='records', indent=2)
        print(f"✓ Saved JSON: {json_file}")
        
        # Save statistics
        stats_file = f'{filename_base}.txt'
        with open(stats_file, 'w') as f:
            f.write("=== FLOOD SCENARIO DATASET STATISTICS ===\n\n")
            f.write(f"Total Scenarios: {len(df)}\n")            
            f.write("--- Expected Warning Distribution ---\n")
            f.write(df['expected_warning'].value_counts().to_string())
            f.write("\n\n--- Scenario Type Distribution ---\n")
            f.write(df['scenario_type'].value_counts().to_string())
            f.write("\n\n--- Statistical Summary ---\n")
            f.write(df[['distance_cm', 'rate_change_cm_per_min', 'rainfall_mm_per_hour']].describe().to_string())
            
        print(f"✓ Saved Statistics: {stats_file}")
        
        return csv_file, json_file, stats_file

def main():
    """Generate test scenarios"""
    
    print("=== Flood Scenario Generator ===\n")
    
    # Initialize generator with system parameters
    generator = FloodScenarioGenerator(
        ground_distance=100,
        siaga_level=130,
        banjir_level=100
    )
    
    # Generate dataset
    print("Generating 1200 scenarios...")
    df = generator.generate_dataset(total_scenarios=1200)
    
    # Display sample
    print("\n--- Sample Scenarios ---")
    print(df.head(10).to_string(index=False))
    
    # Display statistics
    print("\n--- Dataset Statistics ---")
    print(f"Total scenarios: {len(df)}")
    print("\nExpected Warning Distribution:")
    print(df['expected_warning'].value_counts())
    print("\nScenario Type Distribution:")
    print(df['scenario_type'].value_counts())
    
    # Save dataset
    print("\n--- Saving Dataset ---")
    csv_file, json_file, stats_file = generator.save_dataset(df)
    
    print("\n✓ Dataset generation complete!")
    print(f"\nFiles created:")
    print(f"  - {csv_file}")
    print(f"  - {json_file}")
    print(f"  - {stats_file}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main()