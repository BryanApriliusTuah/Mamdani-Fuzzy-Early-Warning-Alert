import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import the fuzzy system
from main2 import DynamicFuzzyFloodWarningSystem


class FuzzySystemEvaluator:
    """Evaluate fuzzy flood warning system using confusion matrix and metrics"""
    
    def __init__(self, system, scenario_file):
        self.system = system
        self.scenario_file = scenario_file
        self.scenarios_df = None
        self.results_df = None
        self.y_true = []
        self.y_pred = []
        
    def load_scenarios(self):
        """Load test scenarios from CSV or JSON file"""
        try:
            if self.scenario_file.endswith('.csv'):
                self.scenarios_df = pd.read_csv(self.scenario_file)
            elif self.scenario_file.endswith('.json'):
                self.scenarios_df = pd.read_json(self.scenario_file)
            else:
                raise ValueError("File must be CSV or JSON")
            
            print(f"✓ Loaded {len(self.scenarios_df)} scenarios from {self.scenario_file}")
            return True
            
        except Exception as e:
            print(f"✗ Error loading scenarios: {e}")
            return False
    
    def prepare_system_for_scenario(self, scenario):
        """
        Prepare the fuzzy system to simulate a specific scenario
        This involves creating a history that would result in the given rate of change
        """
        distance = scenario['distance_cm']
        rate_change = scenario['rate_change_cm_per_min']
        
        # Reset system for new scenario
        self.system.reset_history()
        
        # Calculate what the starting distance should be to achieve the desired rate
        # We'll simulate 60 readings over 60 seconds
        time_span_minutes = 59 / 60.0  # 59 intervals of 1 second
        
        # rate = (final_distance - initial_distance) / time_span_minutes
        # Therefore: initial_distance = final_distance - (rate * time_span_minutes)
        initial_distance = distance - (rate_change * time_span_minutes)
        
        # Generate intermediate distances with some realistic variation
        num_readings = 60
        distances = np.linspace(initial_distance, distance, num_readings)
        
        # Add small random noise to make it more realistic (±0.5 cm)
        noise = np.random.normal(0, 0.3, num_readings)
        distances = distances + noise
        
        # Ensure monotonic progression if rate is significant
        if abs(rate_change) > 1:
            if rate_change < 0:  # Water rising (distance decreasing)
                distances = np.sort(distances)[::-1]
            else:  # Water dropping (distance increasing)
                distances = np.sort(distances)
        
        # Add all readings except the last one
        for i in range(num_readings - 1):
            self.system.add_distance_reading(distances[i])
        
        return distances
    
    def run_single_scenario(self, scenario, verbose=False):
        """Run a single test scenario"""
        try:
            # Prepare system with history
            self.prepare_system_for_scenario(scenario)
            
            # Run the actual test reading
            result = self.system.calculate_risk(
                current_distance=scenario['distance_cm'],
                current_rainfall_mm_per_hour=scenario['rainfall_mm_per_hour']
            )
            
            # Extract prediction
            predicted_warning = result['warning_level']
            expected_warning = scenario['expected_warning']
            
            if verbose:
                print(f"\nScenario #{scenario['scenario_id']}: {scenario['scenario_type']}")
                print(f"  Distance: {scenario['distance_cm']:.2f} cm")
                print(f"  Rate: {scenario['rate_change_cm_per_min']:.2f} cm/min")
                print(f"  Rainfall: {scenario['rainfall_mm_per_hour']:.2f} mm/h")
                print(f"  Expected: {expected_warning}")
                print(f"  Predicted: {predicted_warning}")
                print(f"  Risk Score: {result['risk_score']:.2f}%")
                print(f"  Match: {'✓' if predicted_warning == expected_warning else '✗'}")
            
            return {
                'scenario_id': scenario['scenario_id'],
                'scenario_type': scenario['scenario_type'],
                'distance_cm': scenario['distance_cm'],
                'rate_change_cm_per_min': scenario['rate_change_cm_per_min'],
                'rainfall_mm_per_hour': scenario['rainfall_mm_per_hour'],
                'expected_warning': expected_warning,
                'predicted_warning': predicted_warning,
                'risk_score': result['risk_score'],
                'match': predicted_warning == expected_warning,
                'water_level_normalized': result['water_level_normalized'],
                'rate_normalized': result['avg_rate_normalized']
            }
            
        except Exception as e:
            print(f"✗ Error in scenario #{scenario['scenario_id']}: {e}")
            return None
    
    def evaluate_all_scenarios(self, verbose=False, sample_size=None):
        """Evaluate all scenarios and collect results"""
        
        if self.scenarios_df is None:
            print("✗ No scenarios loaded. Call load_scenarios() first.")
            return False
        
        print("\n=== Evaluating Fuzzy System ===")
        
        # Sample scenarios if requested
        if sample_size and sample_size < len(self.scenarios_df):
            scenarios_to_test = self.scenarios_df.sample(n=sample_size, random_state=42)
            print(f"Testing {sample_size} random scenarios...")
        else:
            scenarios_to_test = self.scenarios_df
            print(f"Testing all {len(scenarios_to_test)} scenarios...")
        
        results = []
        total = len(scenarios_to_test)
        
        for idx, (_, scenario) in enumerate(scenarios_to_test.iterrows(), 1):
            if idx % 100 == 0 or idx == total:
                print(f"Progress: {idx}/{total} ({idx/total*100:.1f}%)", end='\r')
            
            result = self.run_single_scenario(scenario, verbose=verbose)
            if result:
                results.append(result)
                self.y_true.append(result['expected_warning'])
                self.y_pred.append(result['predicted_warning'])
        
        print(f"\n✓ Completed {len(results)} scenarios")
        
        self.results_df = pd.DataFrame(results)
        return True
    
    def generate_confusion_matrix(self, save_plot=True):
        """Generate and visualize confusion matrix"""
        
        if not self.y_true or not self.y_pred:
            print("✗ No results to analyze. Run evaluate_all_scenarios() first.")
            return None
        
        # Define label order
        labels = ['NORMAL', 'SIAGA', 'BANJIR']
        
        # Create confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred, labels=labels)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix - Fuzzy Flood Warning System', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('Expected Warning', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Warning', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_plot:
            filename = f'confusion_matrix.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Saved confusion matrix: {filename}")
        
        plt.show()
        
        return cm
    
    def generate_detailed_report(self, save_report=True):
        """Generate detailed evaluation report"""
        
        if not self.y_true or not self.y_pred:
            print("✗ No results to analyze.")
            return None
        
        labels = ['NORMAL', 'SIAGA', 'BANJIR']
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_true, self.y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_true, self.y_pred, labels=labels, zero_division=0
        )
        
        # Generate classification report
        class_report = classification_report(
            self.y_true, self.y_pred, labels=labels, zero_division=0
        )
        
        # Analyze misclassifications
        misclassified = self.results_df[~self.results_df['match']]
        
        # Create report
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("FUZZY FLOOD WARNING SYSTEM - EVALUATION REPORT")
        report_lines.append("="*70)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Scenarios Tested: {len(self.y_true)}")
        report_lines.append("")
        
        # Overall metrics
        report_lines.append("--- OVERALL PERFORMANCE ---")
        report_lines.append(f"Accuracy: {accuracy*100:.2f}%")
        report_lines.append(f"Correct Predictions: {sum(self.results_df['match'])}/{len(self.results_df)}")
        report_lines.append(f"Misclassifications: {len(misclassified)}")
        report_lines.append("")
        
        # Per-class metrics
        report_lines.append("--- PER-CLASS METRICS ---")
        for i, label in enumerate(labels):
            report_lines.append(f"\n{label}:")
            report_lines.append(f"  Precision: {precision[i]*100:.2f}%")
            report_lines.append(f"  Recall:    {recall[i]*100:.2f}%")
            report_lines.append(f"  F1-Score:  {f1[i]*100:.2f}%")
            report_lines.append(f"  Support:   {support[i]}")
        
        report_lines.append("\n" + "="*70)
        report_lines.append("CLASSIFICATION REPORT")
        report_lines.append("="*70)
        report_lines.append(class_report)
        
        # Misclassification analysis
        if len(misclassified) > 0:
            report_lines.append("\n" + "="*70)
            report_lines.append("MISCLASSIFICATION ANALYSIS")
            report_lines.append("="*70)
            
            # Group by expected-predicted pairs
            error_groups = misclassified.groupby(['expected_warning', 'predicted_warning']).size()
            report_lines.append("\nError Distribution:")
            for (expected, predicted), count in error_groups.items():
                pct = count / len(misclassified) * 100
                report_lines.append(f"  {expected} → {predicted}: {count} ({pct:.1f}%)")
            
            # Show worst scenarios
            report_lines.append("\nTop 10 Most Problematic Scenarios:")
            worst_cases = misclassified.nlargest(10, 'risk_score')[
                ['scenario_id', 'scenario_type', 'expected_warning', 
                 'predicted_warning', 'risk_score']
            ]
            report_lines.append(worst_cases.to_string(index=False))
        
        # Scenario type performance
        report_lines.append("\n" + "="*70)
        report_lines.append("PERFORMANCE BY SCENARIO TYPE")
        report_lines.append("="*70)
        scenario_performance = self.results_df.groupby('scenario_type').agg({
            'match': ['sum', 'count', 'mean']
        }).round(3)
        scenario_performance.columns = ['Correct', 'Total', 'Accuracy']
        scenario_performance['Accuracy'] = scenario_performance['Accuracy'] * 100
        report_lines.append(scenario_performance.to_string())
        
        # Join all lines
        report_text = "\n".join(report_lines)
        
        # Print report
        print(report_text)
        
        # Save report
        if save_report:
            filename = f'evaluation_report.txt'
            with open(filename, 'w') as f:
                f.write(report_text)
            print(f"\n✓ Saved report: {filename}")
            
            # Save detailed results CSV
            csv_filename = f'evaluation_results.csv'
            self.results_df.to_csv(csv_filename, index=False)
            print(f"✓ Saved detailed results: {csv_filename}")
        
        return report_text
    
    def plot_performance_by_scenario(self, save_plot=True):
        """Plot accuracy by scenario type"""
        
        if self.results_df is None:
            print("✗ No results to plot.")
            return
        
        # Calculate accuracy by scenario type
        scenario_acc = self.results_df.groupby('scenario_type')['match'].mean() * 100
        scenario_acc = scenario_acc.sort_values(ascending=True)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(scenario_acc)), scenario_acc.values, 
                       color=['red' if x < 70 else 'orange' if x < 85 else 'green' 
                             for x in scenario_acc.values])
        
        plt.yticks(range(len(scenario_acc)), scenario_acc.index)
        plt.xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
        plt.ylabel('Scenario Type', fontsize=12, fontweight='bold')
        plt.title('Fuzzy System Accuracy by Scenario Type', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlim(0, 105)
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(scenario_acc.values):
            plt.text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_plot:
            filename = f'scenario_performance.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Saved performance plot: {filename}")
        
        plt.show()

def main():
    """Main evaluation script"""
    
    print("="*70)
    print("FUZZY FLOOD WARNING SYSTEM - EVALUATOR")
    print("="*70)
    
    # Check if scenario file is provided
    if len(sys.argv) < 2:
        print("\nUsage: python fuzzy_system_evaluator.py <scenario_file.csv>")
        print("\nSearching for recent scenario files...")
        
        import glob
        csv_files = glob.glob("flood_scenarios.csv")
        if csv_files:
            # Use most recent file
            scenario_file = max(csv_files)
            print(f"Found: {scenario_file}")
        else:
            print("✗ No scenario files found. Please generate scenarios first.")
            return
    else:
        scenario_file = sys.argv[1]
    
    # Initialize fuzzy system
    print("\n--- Initializing Fuzzy System ---")
    system = DynamicFuzzyFloodWarningSystem(reading_interval_seconds=1)
    system.calibrate(ground_distance=100, siaga_level_override=130, banjir_level_override=100)
    
    # Initialize evaluator
    print("\n--- Loading Scenarios ---")
    evaluator = FuzzySystemEvaluator(system, scenario_file)
    
    if not evaluator.load_scenarios():
        return
    
    # Run evaluation
    print("\n--- Running Evaluation ---")
    # Set verbose=True to see details for first few scenarios
    evaluator.evaluate_all_scenarios(verbose=False)
    
    # Generate confusion matrix
    print("\n--- Generating Confusion Matrix ---")
    evaluator.generate_confusion_matrix(save_plot=True)
    
    # Generate performance plot
    print("\n--- Generating Performance Analysis ---")
    evaluator.plot_performance_by_scenario(save_plot=True)
    
    # Generate detailed report
    print("\n--- Generating Detailed Report ---")
    evaluator.generate_detailed_report(save_report=True)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()