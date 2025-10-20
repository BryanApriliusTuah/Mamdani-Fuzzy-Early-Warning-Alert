import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from main2 import DynamicFuzzyFloodWarningSystem
from scenario_generator import FloodScenarioGenerator

class FloodDetectionAnalyzer:
    def __init__(self, ground=100, siaga=130, banjir=100):
        self.ground, self.siaga, self.banjir = ground, siaga, banjir
        self.results = None
        
    def simulate(self, system, distance, rate, rainfall):
        """Simulate scenario and detect if flood warning triggered"""
        system.reset_history()
        max_warning = 'NORMAL'
        
        # Build 60-second history
        for i in range(60):
            past_dist = distance - (rate * (60-i) / 60.0)
            system.add_distance_reading(past_dist)
        
        # Get final prediction
        result = system.calculate_risk(distance, rainfall)
        
        # Check if any flood warning triggered (SIAGA or BANJIR)
        if result['warning_level'] in ['SIAGA', 'BANJIR']:
            max_warning = result['warning_level']
        
        return max_warning, result
    
    def analyze(self, n=1200):
        print(f"Generating {n} flood scenarios...")
        gen = FloodScenarioGenerator(self.ground, self.siaga, self.banjir)
        df = gen.generate_dataset(n)
        
        print("Testing scenarios...")
        sys = DynamicFuzzyFloodWarningSystem(1)
        sys.calibrate(self.ground, self.siaga, self.banjir)
        
        # Test each scenario
        predictions = []
        for _, row in df.iterrows():
            warning, _ = self.simulate(sys, row.distance_cm, row.rate_change_cm_per_min, 
                                      row.rainfall_mm_per_hour)
            # Binary: warned (SIAGA or BANJIR) or not warned (NORMAL)
            predicted_flood = warning in ['SIAGA', 'BANJIR']
            predictions.append(predicted_flood)
        
        df['predicted_flood'] = predictions
        self.results = df
        
        # Calculate binary classification metrics
        y_true = df['actual_flood'].astype(int)
        y_pred = df['predicted_flood'].astype(int)
        
        # Confusion matrix components
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
        print("BINARY CLASSIFICATION RESULTS")
        print("="*60)
        print(f"Accuracy:         {accuracy:.2%}")
        print(f"Precision:        {precision:.2%}")
        print(f"Recall (TPR):     {recall:.2%}")
        print(f"False Alarm Rate: {false_alarm_rate:.2%} {'✅' if false_alarm_rate < 0.2 else '⚠️'}")
        print(f"Miss Rate (FNR):  {miss_rate:.2%} {'✅' if miss_rate < 0.15 else '❌'}")
        
        print("\n" + "="*60)
        print("CONFUSION MATRIX")
        print("="*60)
        print(f"True Positive  (TP): {tp:4d}  |  False Positive (FP): {fp:4d}")
        print(f"False Negative (FN): {fn:4d}  |  True Negative  (TN): {tn:4d}")
        
        # Show problem cases
        fp_cases = df[~df.actual_flood & df.predicted_flood].head(5)
        fn_cases = df[df.actual_flood & ~df.predicted_flood].head(5)
        
        if len(fp_cases) > 0:
            print("\n⚠️  FALSE POSITIVES (sample):")
            for _, row in fp_cases.iterrows():
                print(f"  {row.scenario_type}: dist={row.distance_cm}cm, rate={row.rate_change_cm_per_min:.1f}, rain={row.rainfall_mm_per_hour:.1f}")
        
        if len(fn_cases) > 0:
            print("\n🚨 FALSE NEGATIVES (sample):")
            for _, row in fn_cases.iterrows():
                print(f"  {row.scenario_type}: dist={row.distance_cm}cm, rate={row.rate_change_cm_per_min:.1f}, rain={row.rainfall_mm_per_hour:.1f}")
        
        return df
    
    def plot(self, save='confusion_matrix.png'):
        if self.results is None: return
        
        y_true = self.results['actual_flood'].astype(int)
        y_pred = self.results['predicted_flood'].astype(int)
        
        labels = ['No Flood', 'Flood']
        cm = confusion_matrix(y_true, y_pred)
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, 
                   yticklabels=labels, ax=ax1, square=True, cbar_kws={'label': 'Count'})
        ax1.set_title('Confusion Matrix (Counts)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Actual', fontweight='bold')
        ax1.set_xlabel('Predicted', fontweight='bold')
        
        # Add TP, TN, FP, FN labels
        ax1.text(0.5, 0.1, 'TN', ha='center', va='bottom', fontsize=10, color='darkblue', fontweight='bold')
        ax1.text(1.5, 0.1, 'FP', ha='center', va='bottom', fontsize=10, color='darkred', fontweight='bold')
        ax1.text(0.5, 1.1, 'FN', ha='center', va='bottom', fontsize=10, color='darkred', fontweight='bold')
        ax1.text(1.5, 1.1, 'TP', ha='center', va='bottom', fontsize=10, color='darkgreen', fontweight='bold')
        
        # Percentage matrix
        annot = [[f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)' for j in range(2)] for i in range(2)]
        sns.heatmap(cm_pct, annot=annot, fmt='', cmap='RdYlGn_r', xticklabels=labels, 
                   yticklabels=labels, ax=ax2, square=True, vmin=0, vmax=100,
                   cbar_kws={'label': 'Percentage (%)'})
        ax2.set_title('Confusion Matrix (Percentage)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Actual', fontweight='bold')
        ax2.set_xlabel('Predicted', fontweight='bold')
        
        # Add TP, TN, FP, FN labels
        ax2.text(0.5, 0.1, 'TN', ha='center', va='bottom', fontsize=10, color='darkblue', fontweight='bold')
        ax2.text(1.5, 0.1, 'FP', ha='center', va='bottom', fontsize=10, color='darkred', fontweight='bold')
        ax2.text(0.5, 1.1, 'FN', ha='center', va='bottom', fontsize=10, color='darkred', fontweight='bold')
        ax2.text(1.5, 1.1, 'TP', ha='center', va='bottom', fontsize=10, color='darkgreen', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {save}")
        plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    analyzer = FloodDetectionAnalyzer(ground=100, siaga=130, banjir=100)
    analyzer.analyze(1200)
    analyzer.plot()