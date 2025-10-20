import numpy as np
import pandas as pd

class FloodScenarioGenerator:
    def __init__(self, ground_distance=100, siaga_level=130, banjir_level=100):
        self.ground, self.siaga, self.banjir = ground_distance, siaga_level, banjir_level
        
    def generate_scenario(self, stype):
        scenarios = {
            # SKENARIO BANJIR (4 jenis)
            "Banjir_Bandang": (self.banjir-15, self.banjir+5, -25, -15, 18, 30, True),
            "Banjir_Luapan": (self.banjir-10, self.banjir+3, -18, -10, 15, 25, True),
            "Banjir_Rob": (self.banjir-8, self.banjir+3, -15, -8, 8, 18, True),
            "Banjir_Genangan": (self.banjir-8, self.banjir+2, -12, -6, 10, 20, True),
            
            # SKENARIO AMAN (8 jenis - diperbaiki)
            "Cuaca_Cerah": (self.siaga+15, self.siaga+60, -1, 1, 0, 2, False),
            "Gerimis": (self.siaga+10, self.siaga+40, -3, 2, 2, 8, False),
            "Hujan_Sebentar": (self.siaga+8, self.siaga+35, -4, 3, 5, 15, False),
            "Pasca_Hujan": (self.siaga+5, self.siaga+50, 3, 15, 0, 5, False),
            "Air_Tenang": (self.siaga+5, self.siaga+30, -2, 2, 0, 5, False),
            "Hujan_Terserap": (self.siaga-5, self.siaga+10, -8, -4, 15, 25, False),
            "Pasang_Normal": (self.siaga+5, self.siaga+25, -5, 5, 3, 12, False),
            "Hampir_Siaga": (self.siaga-10, self.siaga+5, -6, 1, 5, 15, False),
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
            # Skenario Banjir (53%)
            ("Banjir_Bandang", 0.106),    # 127 data
            ("Banjir_Luapan", 0.212),     # 254 data
            ("Banjir_Rob", 0.106),        # 127 data
            ("Banjir_Genangan", 0.106),   # 127 data
            
            # Skenario Aman (47%)
            ("Cuaca_Cerah", 0.15),        # 180 data
            ("Gerimis", 0.10),            # 120 data
            ("Hujan_Sebentar", 0.07),     # 84 data
            ("Pasca_Hujan", 0.05),        # 60 data
            ("Air_Tenang", 0.05),         # 60 data
            ("Hujan_Terserap", 0.04),     # 48 data
            ("Pasang_Normal", 0.03),      # 36 data
            ("Hampir_Siaga", 0.03),       # 36 data
        ]
        
        data = []
        for i, (stype, prop) in enumerate(tipe, 1):
            for _ in range(int(n * prop)):
                scenario = self.generate_scenario(stype)
                scenario['id_skenario'] = len(data) + 1
                data.append(scenario)
        
        # Lengkapi sampai jumlah total
        while len(data) < n:
            scenario = self.generate_scenario("Cuaca_Cerah")
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