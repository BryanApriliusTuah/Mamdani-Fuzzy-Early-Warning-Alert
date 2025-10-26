import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from main2 import DynamicFuzzyFloodWarningSystem
from scenario_generator import FloodScenarioGenerator

class FloodDetectionAnalyzer:
    def __init__(self, ground=100, siaga=130, banjir=100):
        self.ground, self.siaga, self.banjir = ground, siaga, banjir
        self.results = None
        
    def simulate(self, system, jarak, laju, hujan):
        """Simulasi skenario dan deteksi peringatan banjir"""
        system.reset_history()
        max_warning = 'NORMAL'
        
        # Bangun riwayat 60 detik
        for i in range(60):
            past_dist = jarak - (laju * (60-i) / 60.0)
            system.add_distance_reading(past_dist)
        
        # Dapatkan prediksi akhir
        result = system.calculate_risk(jarak, hujan)
        
        # Cek apakah ada peringatan banjir (SIAGA I, SIAGA II, atau BANJIR)
        if result['warning_level'] in ['SIAGA I', 'SIAGA II', 'BANJIR']:
            max_warning = result['warning_level']
        
        return max_warning, result
    
    def analyze(self, n=1200):
        print(f"Menghasilkan {n} skenario banjir...")
        gen = FloodScenarioGenerator(self.ground, self.siaga, self.banjir)
        df = gen.generate_dataset(n)
        
        print("Menguji skenario...")
        sys = DynamicFuzzyFloodWarningSystem(1)
        sys.calibrate(self.ground, self.siaga, self.banjir)
        
        # Uji setiap skenario
        predictions = []
        for _, row in df.iterrows():
            warning, _ = self.simulate(sys, row.jarak_cm, row.laju_perubahan_cm_per_menit, 
                                      row.curah_hujan_mm_per_jam)
            banjir_prediksi = warning in ['SIAGA I', 'SIAGA II', 'BANJIR']
            predictions.append(banjir_prediksi)
        
        df['banjir_prediksi'] = predictions
        self.results = df
        
        # Hitung metrik klasifikasi biner
        y_true = df['banjir_aktual'].astype(int)
        y_pred = df['banjir_prediksi'].astype(int)
        
        # Komponen confusion matrix
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        
        accuracy = (tp + tn) / n
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        print("\n" + "="*60)
        print("HASIL KLASIFIKASI BINER")
        print("="*60)
        print(f"Akurasi:              {accuracy:.2%}")
        print(f"Presisi:              {precision:.2%}")
        print(f"Recall (TPR):         {recall:.2%}")
        print(f"Tingkat Alarm Palsu:  {false_alarm_rate:.2%} {'✅' if false_alarm_rate < 0.2 else '⚠️'}")
        print(f"Tingkat Terlewat:     {miss_rate:.2%} {'✅' if miss_rate < 0.15 else '❌'}")
        
        print("\n" + "="*60)
        print("CONFUSION MATRIX")
        print("="*60)
        print(f"True Positive  (TP): {tp:4d}  |  False Positive (FP): {fp:4d}")
        print(f"False Negative (FN): {fn:4d}  |  True Negative  (TN): {tn:4d}")
        
        # Tampilkan kasus bermasalah
        fp_cases = df[~df.banjir_aktual & df.banjir_prediksi].head(5)
        fn_cases = df[df.banjir_aktual & ~df.banjir_prediksi].head(5)
        
        if len(fp_cases) > 0:
            print("\n⚠️  FALSE POSITIVES (contoh):")
            for _, row in fp_cases.iterrows():
                print(f"  {row.tipe_skenario}: jarak={row.jarak_cm}cm, laju={row.laju_perubahan_cm_per_menit:.1f}, hujan={row.curah_hujan_mm_per_jam:.1f}")
        
        if len(fn_cases) > 0:
            print("\n🚨 FALSE NEGATIVES (contoh):")
            for _, row in fn_cases.iterrows():
                print(f"  {row.tipe_skenario}: jarak={row.jarak_cm}cm, laju={row.laju_perubahan_cm_per_menit:.1f}, hujan={row.curah_hujan_mm_per_jam:.1f}")
        
        return df
    
    def plot(self, save='confusion_matrix.png'):
        if self.results is None: return
        
        y_true = self.results['banjir_aktual'].astype(int)
        y_pred = self.results['banjir_prediksi'].astype(int)
        
        labels = ['Tidak Banjir', 'Banjir']
        cm = confusion_matrix(y_true, y_pred)
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Matriks hitungan
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, 
                   yticklabels=labels, ax=ax1, square=True, cbar_kws={'label': 'Jumlah'})
        ax1.set_title('Confusion Matrix (Jumlah)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Aktual', fontweight='bold')
        ax1.set_xlabel('Prediksi', fontweight='bold')
        
        # Tambah label TP, TN, FP, FN
        ax1.text(0.5, 0.1, 'TN', ha='center', va='bottom', fontsize=10, color='darkblue', fontweight='bold')
        ax1.text(1.5, 0.1, 'FP', ha='center', va='bottom', fontsize=10, color='darkred', fontweight='bold')
        ax1.text(0.5, 1.1, 'FN', ha='center', va='bottom', fontsize=10, color='darkred', fontweight='bold')
        ax1.text(1.5, 1.1, 'TP', ha='center', va='bottom', fontsize=10, color='darkgreen', fontweight='bold')
        
        # Matriks persentase
        annot = [[f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)' for j in range(2)] for i in range(2)]
        sns.heatmap(cm_pct, annot=annot, fmt='', cmap='RdYlGn_r', xticklabels=labels, 
                   yticklabels=labels, ax=ax2, square=True, vmin=0, vmax=100,
                   cbar_kws={'label': 'Persentase (%)'})
        ax2.set_title('Confusion Matrix (Persentase)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Aktual', fontweight='bold')
        ax2.set_xlabel('Prediksi', fontweight='bold')
        
        # Tambah label TP, TN, FP, FN
        ax2.text(0.5, 0.1, 'TN', ha='center', va='bottom', fontsize=10, color='darkblue', fontweight='bold')
        ax2.text(1.5, 0.1, 'FP', ha='center', va='bottom', fontsize=10, color='darkred', fontweight='bold')
        ax2.text(0.5, 1.1, 'FN', ha='center', va='bottom', fontsize=10, color='darkred', fontweight='bold')
        ax2.text(1.5, 1.1, 'TP', ha='center', va='bottom', fontsize=10, color='darkgreen', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save, dpi=300, bbox_inches='tight')
        print(f"\n✓ Tersimpan: {save}")
        plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    analyzer = FloodDetectionAnalyzer(ground=100, siaga=130, banjir=100)
    analyzer.analyze(1200)
    analyzer.plot()