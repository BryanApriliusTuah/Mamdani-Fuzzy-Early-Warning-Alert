import numpy as np
import pandas as pd

class FloodScenarioGenerator:
    def __init__(self, ground_distance=100, siaga_level=130, banjir_level=100):
        self.ground, self.siaga, self.banjir = ground_distance, siaga_level, banjir_level
        
    def generate_scenario(self, stype):
        scenarios = {
            # SKENARIO BANJIR (4 jenis)
            "Banjir_Bandang": (self.banjir-30, self.banjir+10, 10, 17.6, 0, 4, True),
            "Banjir_Luapan": (self.banjir-25, self.banjir+5, 0, 0.08, 8, 15, True),
            "Banjir_Rob": (self.banjir-10, self.banjir+8, 0, 0.16, 1, 5, True),
            "Banjir_Genangan": (self.banjir-5, self.banjir+3, 0, 0.5, 55, 73, True),
            
            # SKENARIO AMAN (8 jenis - diperbaiki)
            "Tidak hujan": (self.siaga+15, self.siaga+60, -0.67, 0.67, 0, 5, False),
            "Gerimis": (self.siaga+10, self.siaga+40, 1, 1, 1, 5, False),
            "Hujan sedang": (self.siaga+8, self.siaga+35, -1, 1, 5, 10, False),
            "Air pasang": (self.siaga+5, self.siaga+25, -5, 5, 1, 5, False),
        }
        
        if stype in scenarios:
            d1, d2, r1, r2, rain1, rain2, flood = scenarios[stype]
        else:
            d1, d2, r1, r2, rain1, rain2 = self.banjir, self.siaga+20, -15, 8, 0, 20
            flood = np.random.rand() < 0.5
        
        return {
            'jarak_cm': round(np.random.uniform(d1, d2), 2),
            'laju_perubahan_cm_per_menit': round(np.random.uniform(r1, r2), 2),
            'curah_hujan_mm_per_jam': round(np.random.uniform(rain1, rain2), 2),
            'banjir_aktual': flood,
            'tipe_skenario': stype
        }
    
    def generate_dataset(self, n=1200):
        tipe = [
            # Skenario Banjir (50%)
            ("Banjir_Bandang", 0.125),   # 150 data
			("Banjir_Luapan", 0.125),    # 150 data
			("Banjir_Rob", 0.125),       # 150 data
			("Banjir_Genangan", 0.125),  # 150 data
            
            # Skenario Aman (50%)
            ("Tidak hujan", 0.125),       # 150 data
			("Gerimis", 0.125),          # 150 data
			("Hujan sedang", 0.125),     # 150 data
			("Air pasang", 0.125),    # 150 data
        ]
        
        data = []
        for i, (stype, prop) in enumerate(tipe, 1):
            for _ in range(int(n * prop)):
                scenario = self.generate_scenario(stype)
                scenario['id_skenario'] = len(data) + 1
                data.append(scenario)
        
        # Lengkapi sampai jumlah total
        while len(data) < n:
            scenario = self.generate_scenario("Tidak hujan")
            scenario['id_skenario'] = len(data) + 1
            data.append(scenario)
        
        np.random.shuffle(data)
        df = pd.DataFrame(data)
        return df[['id_skenario', 'jarak_cm', 'laju_perubahan_cm_per_menit', 
                   'curah_hujan_mm_per_jam', 'banjir_aktual', 'tipe_skenario']]

if __name__ == "__main__":
    np.random.seed(42)
    gen = FloodScenarioGenerator(100, 130, 100)
    df = gen.generate_dataset(1200)
    
    print(f"Menghasilkan {len(df)} skenario")
    print(f"Banjir: {df['banjir_aktual'].sum()} ({df['banjir_aktual'].sum()/len(df):.1%})")
    print(f"Aman: {(~df['banjir_aktual']).sum()} ({(~df['banjir_aktual']).sum()/len(df):.1%})")
    
    df.to_csv('skenario_banjir.csv', index=False)
    print("✓ Tersimpan: skenario_banjir.csv")