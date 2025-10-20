import numpy as np
import pandas as pd

class FloodScenarioGenerator:
    def __init__(self, ground_distance=100, siaga_level=130, banjir_level=100):
        self.ground, self.siaga, self.banjir = ground_distance, siaga_level, banjir_level
        
    def generate_scenario(self, stype):
        # FLOOD SCENARIOS
        scenarios = {
            "Flash_Flood": (self.banjir-15, self.banjir+5, -25, -15, 18, 30, True),
            "Monsoon_Flood": (self.banjir-10, self.banjir+3, -18, -10, 15, 25, True),
            "Slow_Flood": (self.banjir-8, self.banjir+2, -12, -6, 10, 20, True),
            "Dam_Flood": (self.banjir-12, self.banjir+2, -22, -12, 5, 15, True),
            "Tidal_Flood": (self.banjir-8, self.banjir+3, -15, -8, 8, 18, True),
            "Very_Slow_Flood": (self.banjir-5, self.banjir+5, -5, -2, 3, 10, True),
            "Seepage_Flood": (self.banjir-8, self.banjir+2, -8, -3, 0, 8, True),
            "Delayed_Flood": (self.banjir-6, self.banjir+3, -10, -5, 2, 8, True),
            # SAFE SCENARIOS
            "Normal_Dry": (self.siaga+15, self.siaga+60, -1, 1, 0, 2, False),
            "Light_Rain": (self.siaga+10, self.siaga+40, -3, 2, 2, 8, False),
            "Brief_Shower": (self.siaga+8, self.siaga+35, -4, 3, 5, 15, False),
            "Receding": (self.siaga+5, self.siaga+50, 3, 15, 0, 5, False),
            "Stable_Low": (self.siaga+5, self.siaga+30, -2, 2, 0, 5, False),
            "Heavy_Rain_Safe": (self.siaga-5, self.siaga+10, -8, -4, 15, 25, False),
            "Sensor_Noise": (self.siaga+5, self.siaga+25, -5, 5, 3, 12, False),
            "Near_Threshold": (self.siaga-10, self.siaga+5, -6, 1, 5, 15, False),
        }
        
        if stype in scenarios:
            d1, d2, r1, r2, rain1, rain2, flood = scenarios[stype]
        else:
            d1, d2, r1, r2, rain1, rain2 = self.banjir, self.siaga+20, -15, 8, 0, 20
            flood = np.random.rand() < 0.5
        
        return {
            'distance_cm': round(np.random.uniform(d1, d2), 2),
            'rate_change_cm_per_min': round(np.random.uniform(r1, r2), 2),
            'rainfall_mm_per_hour': round(np.random.uniform(rain1, rain2), 2),
            'actual_flood': flood,
            'scenario_type': stype
        }
    
    def generate_dataset(self, n=1200):
        types = [
            ("Flash_Flood", 0.15), ("Monsoon_Flood", 0.12), ("Slow_Flood", 0.08),
            ("Dam_Flood", 0.06), ("Tidal_Flood", 0.05), ("Very_Slow_Flood", 0.03),
            ("Seepage_Flood", 0.02), ("Delayed_Flood", 0.02),
            ("Normal_Dry", 0.15), ("Light_Rain", 0.12), ("Brief_Shower", 0.08),
            ("Receding", 0.05), ("Stable_Low", 0.05), ("Heavy_Rain_Safe", 0.04),
            ("Sensor_Noise", 0.03), ("Near_Threshold", 0.03)
        ]
        
        data = []
        for i, (stype, prop) in enumerate(types, 1):
            for _ in range(int(n * prop)):
                scenario = self.generate_scenario(stype)
                scenario['scenario_id'] = len(data) + 1
                data.append(scenario)
        
        # Fill to exact total
        while len(data) < n:
            scenario = self.generate_scenario("Normal_Dry")
            scenario['scenario_id'] = len(data) + 1
            data.append(scenario)
        
        np.random.shuffle(data)
        df = pd.DataFrame(data)
        return df[['scenario_id', 'distance_cm', 'rate_change_cm_per_min', 
                   'rainfall_mm_per_hour', 'actual_flood', 'scenario_type']]

if __name__ == "__main__":
    np.random.seed(42)
    gen = FloodScenarioGenerator(100, 130, 100)
    df = gen.generate_dataset(1200)
    
    print(f"Generated {len(df)} scenarios")
    print(f"Flood: {df['actual_flood'].sum()} ({df['actual_flood'].sum()/len(df):.1%})")
    print(f"Safe: {(~df['actual_flood']).sum()} ({(~df['actual_flood']).sum()/len(df):.1%})")
    
    df.to_csv('flood_scenarios.csv', index=False)
    print("✓ Saved: flood_scenarios.csv")