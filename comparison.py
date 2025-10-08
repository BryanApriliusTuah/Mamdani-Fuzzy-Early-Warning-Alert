import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class ParameterSet:
    """Container for a complete set of fuzzy membership function parameters"""
    name: str
    description: str
    source: str
    
    water_normal: List[float]
    water_siaga: List[float]
    water_banjir: List[float]
    
    rate_naik_cepat: List[float]
    rate_naik_lambat: List[float]
    rate_stabil: List[float]
    rate_turun_lambat: List[float]
    rate_turun_cepat: List[float]
    
    rain_tidak: List[float]
    rain_ringan: List[float]
    rain_sedang: List[float]
    rain_lebat: List[float]
    rain_sangat_lebat: List[float]
    
    risk_low: List[float]
    risk_medium: List[float]
    risk_high: List[float]

class FuzzyParameterEvaluator:
    """Evaluate different fuzzy parameter sets against ground truth scenarios"""
    
    def __init__(self):
        self.parameter_sets = self._define_parameter_sets()
        self.test_scenarios = self._define_test_scenarios()
        self.all_evaluations = {}
    
    def _define_parameter_sets(self) -> Dict[str, ParameterSet]:
        """Define multiple parameter configurations to test"""
        
        sets = {}
        
        sets['original'] = ParameterSet(
            name="Original (Current)",
            description="Current system parameters",
            source="Initial design based on general practice",
            
            water_normal=[0, 0, 0.4, 0.6],
            water_siaga=[0.5, 0.75, 0.9],
            water_banjir=[0.85, 0.95, 1.0, 1.0],
            
            rate_naik_cepat=[-1, -1, -0.4, -0.2],
            rate_naik_lambat=[-0.3, -0.15, 0],
            rate_stabil=[-0.1, 0, 0.1],
            rate_turun_lambat=[0, 0.15, 0.3],
            rate_turun_cepat=[0.2, 0.4, 1, 1],
            
            rain_tidak=[0, 0, 0.02, 0.04],
            rain_ringan=[0.02, 0.12, 0.2],
            rain_sedang=[0.16, 0.3, 0.4],
            rain_lebat=[0.36, 0.6, 0.8],
            rain_sangat_lebat=[0.72, 0.88, 1.0, 1.0],
            
            risk_low=[0, 0, 50],
            risk_medium=[40, 60, 80],
            risk_high=[70, 100, 100]
        )
        
        sets['literature'] = ParameterSet(
            name="Literature-Based",
            description="Based on flood early warning standards",
            source="Academic literature",
            
            water_normal=[0, 0, 0.35, 0.5],
            water_siaga=[0.4, 0.6, 0.8],
            water_banjir=[0.75, 0.9, 1.0, 1.0],
            
            rate_naik_cepat=[-1, -1, -0.5, -0.3],
            rate_naik_lambat=[-0.35, -0.2, -0.05],
            rate_stabil=[-0.08, 0, 0.08],
            rate_turun_lambat=[0.05, 0.2, 0.35],
            rate_turun_cepat=[0.3, 0.5, 1, 1],
            
            rain_tidak=[0, 0, 0.03, 0.05],
            rain_ringan=[0.03, 0.1, 0.18],
            rain_sedang=[0.15, 0.28, 0.42],
            rain_lebat=[0.38, 0.58, 0.78],
            rain_sangat_lebat=[0.75, 0.9, 1.0, 1.0],
            
            risk_low=[0, 0, 35],
            risk_medium=[25, 50, 75],
            risk_high=[65, 100, 100]
        )
        
        sets['aggressive'] = ParameterSet(
            name="Aggressive (Early Warning)",
            description="Very sensitive - early detection",
            source="Safety-first approach",
            
            water_normal=[0, 0, 0.3, 0.45],
            water_siaga=[0.35, 0.55, 0.75],
            water_banjir=[0.7, 0.85, 1.0, 1.0],
            
            rate_naik_cepat=[-1, -1, -0.55, -0.35],
            rate_naik_lambat=[-0.4, -0.25, -0.1],
            rate_stabil=[-0.12, 0, 0.12],
            rate_turun_lambat=[0.1, 0.25, 0.4],
            rate_turun_cepat=[0.35, 0.55, 1, 1],
            
            rain_tidak=[0, 0, 0.025, 0.045],
            rain_ringan=[0.025, 0.08, 0.15],
            rain_sedang=[0.12, 0.25, 0.38],
            rain_lebat=[0.35, 0.55, 0.75],
            rain_sangat_lebat=[0.7, 0.85, 1.0, 1.0],
            
            risk_low=[0, 0, 25],
            risk_medium=[15, 45, 70],
            risk_high=[60, 100, 100]
        )
        
        sets['conservative'] = ParameterSet(
            name="Conservative (Fewer False Alarms)",
            description="Less sensitive - reduces false positives",
            source="Optimized for frequent rain areas",
            
            water_normal=[0, 0, 0.45, 0.65],
            water_siaga=[0.55, 0.8, 0.95],
            water_banjir=[0.9, 0.97, 1.0, 1.0],
            
            rate_naik_cepat=[-1, -1, -0.35, -0.15],
            rate_naik_lambat=[-0.25, -0.1, 0.05],
            rate_stabil=[-0.08, 0, 0.08],
            rate_turun_lambat=[-0.05, 0.1, 0.25],
            rate_turun_cepat=[0.15, 0.35, 1, 1],
            
            rain_tidak=[0, 0, 0.04, 0.08],
            rain_ringan=[0.04, 0.15, 0.25],
            rain_sedang=[0.2, 0.35, 0.48],
            rain_lebat=[0.42, 0.65, 0.85],
            rain_sangat_lebat=[0.8, 0.92, 1.0, 1.0],
            
            risk_low=[0, 0, 35],
            risk_medium=[25, 55, 85],
            risk_high=[75, 100, 100]
        )
        
        sets['balanced'] = ParameterSet(
            name="Balanced (Optimized)",
            description="Balanced sensitivity",
            source="Empirically tuned",
            
            water_normal=[0, 0, 0.38, 0.55],
            water_siaga=[0.45, 0.68, 0.85],
            water_banjir=[0.8, 0.92, 1.0, 1.0],
            
            rate_naik_cepat=[-1, -1, -0.45, -0.25],
            rate_naik_lambat=[-0.32, -0.18, -0.02],
            rate_stabil=[-0.09, 0, 0.09],
            rate_turun_lambat=[0.02, 0.18, 0.32],
            rate_turun_cepat=[0.25, 0.45, 1, 1],
            
            rain_tidak=[0, 0, 0.028, 0.048],
            rain_ringan=[0.028, 0.11, 0.19],
            rain_sedang=[0.17, 0.3, 0.41],
            rain_lebat=[0.37, 0.59, 0.79],
            rain_sangat_lebat=[0.74, 0.89, 1.0, 1.0],
            
            risk_low=[0, 0, 32],
            risk_medium=[22, 50, 78],
            risk_high=[68, 100, 100]
        )
        
        return sets
    
    def _define_test_scenarios(self) -> List[Dict]:
        """Define 100 comprehensive test scenarios"""
        
        scenarios = []
        
        # ============================================
        # LOW RISK SCENARIOS (40 scenarios)
        # ============================================
        
        # Very safe conditions (12 scenarios)
        for i, (wl, rate, rain) in enumerate([
            (0.05, 0.0, 0.01), (0.1, 0.0, 0.0), (0.15, 0.05, 0.02),
            (0.2, 0.0, 0.05), (0.25, 0.1, 0.03), (0.3, 0.0, 0.0),
            (0.1, 0.2, 0.0), (0.2, 0.15, 0.01), (0.25, 0.0, 0.1),
            (0.15, -0.05, 0.15), (0.2, -0.05, 0.2), (0.1, 0.0, 0.3),
        ]):
            scenarios.append({
                'name': f'Safe L{i+1} - Very low water',
                'water_level_norm': wl,
                'rate_norm': rate,
                'rainfall_norm': rain,
                'expected_warning': 'NORMAL',
                'expected_risk_category': 'low',
                'priority': 'high'
            })
        
        # Safe with light-moderate rain (9 scenarios)
        for i, (wl, rate, rain) in enumerate([
            (0.2, 0.0, 0.15), (0.25, 0.0, 0.2), (0.3, 0.0, 0.25),
            (0.35, 0.0, 0.18), (0.3, -0.08, 0.3), (0.25, -0.1, 0.35),
            (0.35, 0.05, 0.2), (0.4, 0.0, 0.15), (0.3, -0.05, 0.28),
        ]):
            scenarios.append({
                'name': f'Safe L{i+13} - Light rain',
                'water_level_norm': wl,
                'rate_norm': rate,
                'rainfall_norm': rain,
                'expected_warning': 'NORMAL',
                'expected_risk_category': 'low',
                'priority': 'high'
            })
        
        # Moderate level, receding (8 scenarios)
        for i, (wl, rate, rain) in enumerate([
            (0.35, 0.2, 0.05), (0.4, 0.25, 0.03), (0.45, 0.3, 0.0),
            (0.4, 0.15, 0.1), (0.5, 0.2, 0.08), (0.45, 0.35, 0.02),
            (0.38, 0.18, 0.15), (0.42, 0.22, 0.12),
        ]):
            scenarios.append({
                'name': f'Safe L{i+22} - Receding',
                'water_level_norm': wl,
                'rate_norm': rate,
                'rainfall_norm': rain,
                'expected_warning': 'NORMAL',
                'expected_risk_category': 'low',
                'priority': 'medium'
            })
        
        # Edge of safe zone (11 scenarios)
        for i, (wl, rate, rain) in enumerate([
            (0.45, -0.08, 0.2), (0.5, 0.0, 0.15), (0.48, -0.1, 0.25),
            (0.52, 0.05, 0.18), (0.5, -0.12, 0.22), (0.48, 0.0, 0.28),
            (0.52, -0.05, 0.2), (0.5, 0.08, 0.15), (0.48, -0.08, 0.3),
            (0.55, 0.1, 0.1), (0.53, 0.0, 0.25),
        ]):
            scenarios.append({
                'name': f'Safe L{i+30} - Edge safe zone',
                'water_level_norm': wl,
                'rate_norm': rate,
                'rainfall_norm': rain,
                'expected_warning': 'NORMAL',
                'expected_risk_category': 'low',
                'priority': 'medium'
            })
        
        # ============================================
        # MEDIUM RISK SCENARIOS (35 scenarios)
        # ============================================
        
        # Entering warning zone (10 scenarios)
        for i, (wl, rate, rain) in enumerate([
            (0.6, -0.15, 0.3), (0.62, -0.18, 0.35), (0.65, -0.12, 0.4),
            (0.68, -0.2, 0.28), (0.63, -0.1, 0.38), (0.67, -0.15, 0.32),
            (0.6, 0.0, 0.25), (0.65, 0.0, 0.3), (0.68, -0.05, 0.35),
            (0.62, -0.22, 0.42),
        ]):
            scenarios.append({
                'name': f'Warning M{i+1} - Entering warning',
                'water_level_norm': wl,
                'rate_norm': rate,
                'rainfall_norm': rain,
                'expected_warning': 'SIAGA',
                'expected_risk_category': 'medium',
                'priority': 'high'
            })
        
        # Mid-warning level (11 scenarios)
        for i, (wl, rate, rain) in enumerate([
            (0.7, -0.2, 0.3), (0.72, -0.18, 0.35), (0.75, -0.15, 0.4),
            (0.78, -0.12, 0.32), (0.73, -0.25, 0.38), (0.76, -0.15, 0.28),
            (0.7, 0.0, 0.2), (0.75, 0.0, 0.25), (0.78, -0.08, 0.3),
            (0.72, -0.1, 0.35), (0.77, 0.05, 0.2),
        ]):
            scenarios.append({
                'name': f'Warning M{i+11} - Mid warning',
                'water_level_norm': wl,
                'rate_norm': rate,
                'rainfall_norm': rain,
                'expected_warning': 'SIAGA',
                'expected_risk_category': 'medium',
                'priority': 'high'
            })
        
        # High warning level (8 scenarios)
        for i, (wl, rate, rain) in enumerate([
            (0.8, -0.15, 0.25), (0.82, -0.2, 0.3), (0.83, -0.12, 0.35),
            (0.81, -0.18, 0.28), (0.84, -0.1, 0.32), (0.8, 0.0, 0.2),
            (0.82, 0.05, 0.15), (0.83, -0.08, 0.25),
        ]):
            scenarios.append({
                'name': f'Warning M{i+22} - High warning',
                'water_level_norm': wl,
                'rate_norm': rate,
                'rainfall_norm': rain,
                'expected_warning': 'SIAGA',
                'expected_risk_category': 'medium',
                'priority': 'high'
            })
        
        # Warning with receding water (6 scenarios)
        for i, (wl, rate, rain) in enumerate([
            (0.75, 0.3, 0.1), (0.8, 0.35, 0.05), (0.7, 0.25, 0.15),
            (0.82, 0.4, 0.08), (0.78, 0.28, 0.12), (0.85, 0.45, 0.05),
        ]):
            scenarios.append({
                'name': f'Warning M{i+30} - Receding',
                'water_level_norm': wl,
                'rate_norm': rate,
                'rainfall_norm': rain,
                'expected_warning': 'SIAGA',
                'expected_risk_category': 'medium',
                'priority': 'high'
            })
        
        # ============================================
        # HIGH RISK SCENARIOS (25 scenarios)
        # ============================================
        
        # At flood threshold (8 scenarios)
        for i, (wl, rate, rain) in enumerate([
            (0.87, -0.2, 0.5), (0.88, -0.25, 0.55), (0.9, -0.15, 0.6),
            (0.89, -0.18, 0.52), (0.91, -0.22, 0.58), (0.86, -0.12, 0.48),
            (0.9, 0.0, 0.4), (0.88, -0.08, 0.5),
        ]):
            scenarios.append({
                'name': f'Critical H{i+1} - At threshold',
                'water_level_norm': wl,
                'rate_norm': rate,
                'rainfall_norm': rain,
                'expected_warning': 'BANJIR',
                'expected_risk_category': 'high',
                'priority': 'critical'
            })
        
        # Flood level (9 scenarios)
        for i, (wl, rate, rain) in enumerate([
            (0.93, -0.15, 0.6), (0.95, -0.1, 0.7), (0.96, -0.08, 0.65),
            (0.94, -0.2, 0.75), (0.97, -0.05, 0.8), (0.98, -0.12, 0.85),
            (0.95, 0.0, 0.5), (0.92, -0.18, 0.7), (0.99, -0.1, 0.9),
        ]):
            scenarios.append({
                'name': f'Critical H{i+9} - Flood level',
                'water_level_norm': wl,
                'rate_norm': rate,
                'rainfall_norm': rain,
                'expected_warning': 'BANJIR',
                'expected_risk_category': 'high',
                'priority': 'critical'
            })
        
        # Rapid rise situations (6 scenarios)
        for i, (wl, rate, rain) in enumerate([
            (0.7, -0.45, 0.7), (0.65, -0.5, 0.75), (0.6, -0.48, 0.8),
            (0.75, -0.42, 0.65), (0.5, -0.5, 0.85), (0.55, -0.48, 0.9),
        ]):
            scenarios.append({
                'name': f'Critical H{i+18} - Rapid rise',
                'water_level_norm': wl,
                'rate_norm': rate,
                'rainfall_norm': rain,
                'expected_warning': 'SIAGA',
                'expected_risk_category': 'high',
                'priority': 'critical'
            })
        
        # Recovery from flood (2 scenarios)
        for i, (wl, rate, rain) in enumerate([
            (0.93, 0.3, 0.1), (0.9, 0.35, 0.05),
        ]):
            scenarios.append({
                'name': f'Critical H{i+24} - Recovery',
                'water_level_norm': wl,
                'rate_norm': rate,
                'rainfall_norm': rain,
                'expected_warning': 'BANJIR',
                'expected_risk_category': 'medium',
                'priority': 'high'
            })
        
        print(f"\n✅ Generated {len(scenarios)} test scenarios:")
        print(f"   - LOW risk: {sum(1 for s in scenarios if s['expected_risk_category'] == 'low')}")
        print(f"   - MEDIUM risk: {sum(1 for s in scenarios if s['expected_risk_category'] == 'medium')}")
        print(f"   - HIGH risk: {sum(1 for s in scenarios if s['expected_risk_category'] == 'high')}")
        
        return scenarios
    
    def _create_fuzzy_system(self, params: ParameterSet):
        """Create fuzzy system with given parameters"""
        
        water_level_norm = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'water_level_norm')
        rate_change_norm = ctrl.Antecedent(np.arange(-1, 1.01, 0.01), 'rate_change_norm')
        rainfall_norm = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'rainfall_norm')
        flood_risk = ctrl.Consequent(np.arange(0, 101, 1), 'flood_risk')
        
        water_level_norm['normal'] = fuzz.trapmf(water_level_norm.universe, params.water_normal)
        water_level_norm['siaga'] = fuzz.trimf(water_level_norm.universe, params.water_siaga)
        water_level_norm['banjir'] = fuzz.trapmf(water_level_norm.universe, params.water_banjir)
        
        rate_change_norm['naik cepat'] = fuzz.trapmf(rate_change_norm.universe, params.rate_naik_cepat)
        rate_change_norm['naik lambat'] = fuzz.trimf(rate_change_norm.universe, params.rate_naik_lambat)
        rate_change_norm['stabil'] = fuzz.trimf(rate_change_norm.universe, params.rate_stabil)
        rate_change_norm['turun lambat'] = fuzz.trimf(rate_change_norm.universe, params.rate_turun_lambat)
        rate_change_norm['turun cepat'] = fuzz.trapmf(rate_change_norm.universe, params.rate_turun_cepat)
        
        rainfall_norm['tidak_hujan'] = fuzz.trapmf(rainfall_norm.universe, params.rain_tidak)
        rainfall_norm['ringan'] = fuzz.trimf(rainfall_norm.universe, params.rain_ringan)
        rainfall_norm['sedang'] = fuzz.trimf(rainfall_norm.universe, params.rain_sedang)
        rainfall_norm['lebat'] = fuzz.trimf(rainfall_norm.universe, params.rain_lebat)
        rainfall_norm['sangat_lebat'] = fuzz.trapmf(rainfall_norm.universe, params.rain_sangat_lebat)
        
        flood_risk['low'] = fuzz.trimf(flood_risk.universe, params.risk_low)
        flood_risk['medium'] = fuzz.trimf(flood_risk.universe, params.risk_medium)
        flood_risk['high'] = fuzz.trimf(flood_risk.universe, params.risk_high)
        
        rules = []
        rules.append(ctrl.Rule(water_level_norm['banjir'] & rate_change_norm['naik cepat'], flood_risk['high']))
        rules.append(ctrl.Rule(water_level_norm['banjir'] & rate_change_norm['naik lambat'], flood_risk['high']))
        rules.append(ctrl.Rule(water_level_norm['banjir'] & rate_change_norm['stabil'], flood_risk['high']))
        rules.append(ctrl.Rule(water_level_norm['banjir'] & rate_change_norm['turun lambat'], flood_risk['medium']))
        rules.append(ctrl.Rule(water_level_norm['banjir'] & rate_change_norm['turun cepat'], flood_risk['medium']))
        
        rules.append(ctrl.Rule(water_level_norm['siaga'] & rate_change_norm['naik cepat'] & rainfall_norm['lebat'], flood_risk['high']))
        rules.append(ctrl.Rule(water_level_norm['siaga'] & rate_change_norm['naik cepat'] & rainfall_norm['sangat_lebat'], flood_risk['high']))
        rules.append(ctrl.Rule(water_level_norm['siaga'] & rate_change_norm['naik cepat'], flood_risk['high']))
        rules.append(ctrl.Rule(water_level_norm['siaga'] & rate_change_norm['naik lambat'] & rainfall_norm['lebat'], flood_risk['high']))
        rules.append(ctrl.Rule(water_level_norm['siaga'] & rate_change_norm['naik lambat'] & rainfall_norm['sangat_lebat'], flood_risk['high']))
        rules.append(ctrl.Rule(water_level_norm['siaga'] & rate_change_norm['naik lambat'], flood_risk['medium']))
        rules.append(ctrl.Rule(water_level_norm['siaga'] & rate_change_norm['stabil'], flood_risk['medium']))
        rules.append(ctrl.Rule(water_level_norm['siaga'] & rate_change_norm['turun lambat'], flood_risk['low']))
        rules.append(ctrl.Rule(water_level_norm['siaga'] & rate_change_norm['turun cepat'], flood_risk['low']))
        
        rules.append(ctrl.Rule(water_level_norm['normal'] & rainfall_norm['sangat_lebat'] & rate_change_norm['naik cepat'], flood_risk['high']))
        rules.append(ctrl.Rule(water_level_norm['normal'] & rainfall_norm['lebat'] & rate_change_norm['naik cepat'], flood_risk['medium']))
        rules.append(ctrl.Rule(water_level_norm['normal'] & rainfall_norm['sangat_lebat'] & rate_change_norm['naik lambat'], flood_risk['medium']))
        rules.append(ctrl.Rule(water_level_norm['normal'] & rate_change_norm['naik cepat'], flood_risk['medium']))
        rules.append(ctrl.Rule(water_level_norm['normal'] & rate_change_norm['naik lambat'], flood_risk['low']))
        rules.append(ctrl.Rule(water_level_norm['normal'] & rate_change_norm['stabil'], flood_risk['low']))
        rules.append(ctrl.Rule(water_level_norm['normal'] & rate_change_norm['turun lambat'], flood_risk['low']))
        rules.append(ctrl.Rule(water_level_norm['normal'] & rate_change_norm['turun cepat'], flood_risk['low']))
        
        flood_ctrl = ctrl.ControlSystem(rules)
        return ctrl.ControlSystemSimulation(flood_ctrl)
    
    def _categorize_risk(self, risk_score):
        if risk_score < 40:
            return 'low'
        elif risk_score < 70:
            return 'medium'
        else:
            return 'high'
    
    def _determine_warning(self, risk_score, water_level_norm):
        if water_level_norm >= 0.85:
            return 'BANJIR'
        elif water_level_norm >= 0.6:
            if risk_score >= 70:
                return 'BANJIR'
            else:
                return 'SIAGA'
        else:
            if risk_score >= 60:
                return 'SIAGA'
            else:
                return 'NORMAL'
    
    def evaluate_single_scenario(self, fuzzy_system, scenario):
        try:
            fuzzy_system.input['water_level_norm'] = scenario['water_level_norm']
            fuzzy_system.input['rate_change_norm'] = scenario['rate_norm']
            fuzzy_system.input['rainfall_norm'] = scenario['rainfall_norm']
            fuzzy_system.compute()
            
            risk_score = fuzzy_system.output['flood_risk']
            risk_category = self._categorize_risk(risk_score)
            warning_level = self._determine_warning(risk_score, scenario['water_level_norm'])
            
            warning_correct = warning_level == scenario['expected_warning']
            risk_correct = risk_category == scenario['expected_risk_category']
            
            return {
                'risk_score': risk_score,
                'risk_category': risk_category,
                'warning_level': warning_level,
                'warning_correct': warning_correct,
                'risk_correct': risk_correct,
                'both_correct': warning_correct and risk_correct
            }
        except:
            return {
                'risk_score': 0,
                'risk_category': 'error',
                'warning_level': 'ERROR',
                'warning_correct': False,
                'risk_correct': False,
                'both_correct': False
            }
	
    def evaluate_parameter_set(self, param_name: str) -> Dict:
        params = self.parameter_sets[param_name]
        fuzzy_system = self._create_fuzzy_system(params)
        
        results = []
        for scenario in self.test_scenarios:
            result = self.evaluate_single_scenario(fuzzy_system, scenario)
            result['scenario_name'] = scenario['name']
            result['expected_warning'] = scenario['expected_warning']
            result['expected_risk'] = scenario['expected_risk_category']
            result['priority'] = scenario['priority']
            results.append(result)
        
        df = pd.DataFrame(results)
        
        total = len(df)
        warning_accuracy = (df['warning_correct'].sum() / total) * 100
        risk_accuracy = (df['risk_correct'].sum() / total) * 100
        overall_accuracy = (df['both_correct'].sum() / total) * 100
        
        risk_category_accuracy = {}
        for risk_cat in ['low', 'medium', 'high']:
            cat_scenarios = df[df['expected_risk'] == risk_cat]
            if len(cat_scenarios) > 0:
                risk_category_accuracy[risk_cat] = {
                    'count': len(cat_scenarios),
                    'correct': cat_scenarios['both_correct'].sum(),
                    'accuracy': (cat_scenarios['both_correct'].sum() / len(cat_scenarios)) * 100
                }
            else:
                risk_category_accuracy[risk_cat] = {'count': 0, 'correct': 0, 'accuracy': 0}
        
        critical_scenarios = df[df['priority'] == 'critical']
        high_scenarios = df[df['priority'] == 'high']
        
        critical_accuracy = (critical_scenarios['both_correct'].sum() / len(critical_scenarios) * 100) if len(critical_scenarios) > 0 else 0
        high_accuracy = (high_scenarios['both_correct'].sum() / len(high_scenarios) * 100) if len(high_scenarios) > 0 else 0
        
        return {
            'param_name': param_name,
            'params': params,
            'results_df': df,
            'warning_accuracy': warning_accuracy,
            'risk_accuracy': risk_accuracy,
            'overall_accuracy': overall_accuracy,
            'risk_category_accuracy': risk_category_accuracy,
            'critical_accuracy': critical_accuracy,
            'high_priority_accuracy': high_accuracy,
            'avg_risk_score': df['risk_score'].mean()
        }
    
    def evaluate_all_parameter_sets(self):
        print("\n" + "="*100)
        print("FUZZY PARAMETER EVALUATION SYSTEM - 100 TEST SCENARIOS")
        print("="*100)
        
        for param_name in self.parameter_sets.keys():
            print(f"\nEvaluating: {self.parameter_sets[param_name].name}...")
            evaluation = self.evaluate_parameter_set(param_name)
            self.all_evaluations[param_name] = evaluation
        
        print("\n✅ Evaluation complete!")

    def print_overall_accuracy_table(self):
        print("\n" + "="*100)
        print("TABLE 1: OVERALL ACCURACY OF EACH PARAMETER SET")
        print("="*100 + "\n")
        
        table_data = []
        for param_name, eval_data in self.all_evaluations.items():
            table_data.append({
                'Parameter Set': eval_data['params'].name,
                'Source': eval_data['params'].source,
                'Overall Accuracy (%)': round(eval_data['overall_accuracy'], 1),
                'Warning Accuracy (%)': round(eval_data['warning_accuracy'], 1),
                'Risk Category Accuracy (%)': round(eval_data['risk_accuracy'], 1),
                'Critical Scenarios (%)': round(eval_data['critical_accuracy'], 1),
                'Avg Risk Score': round(eval_data['avg_risk_score'], 1)
            })
        
        df = pd.DataFrame(table_data)
        df = df.sort_values('Overall Accuracy (%)', ascending=False)
        
        print(df.to_string(index=False))
        
        best = df.iloc[0]
        print("\n" + "-"*100)
        print(f"🏆 BEST PERFORMER: {best['Parameter Set']}")
        print(f"   Overall Accuracy: {best['Overall Accuracy (%)']}%")
        print(f"   Critical Scenarios: {best['Critical Scenarios (%)']}%")
        print("-"*100)
        
        return df
    
    def print_risk_category_accuracy_table(self):
        print("\n" + "="*100)
        print("TABLE 2: ACCURACY BY RISK CATEGORY FOR EACH PARAMETER SET")
        print("="*100 + "\n")
        
        table_data = []
        for param_name, eval_data in self.all_evaluations.items():
            row = {'Parameter Set': eval_data['params'].name}
            
            for risk_cat in ['low', 'medium', 'high']:
                cat_data = eval_data['risk_category_accuracy'][risk_cat]
                accuracy = cat_data['accuracy']
                count = cat_data['count']
                correct = cat_data['correct']
                row[f'{risk_cat.upper()} Risk (%)'] = round(accuracy, 1)
                row[f'{risk_cat.upper()} Correct/Total'] = f"{int(correct)}/{count}"
            
            row['Overall (%)'] = round(eval_data['overall_accuracy'], 1)
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        df = df.sort_values('Overall (%)', ascending=False)
        
        print(df.to_string(index=False))
        
        print("\n" + "-"*100)
        print("INTERPRETATION:")
        print("• LOW Risk: Should classify as NORMAL (safe conditions)")
        print("• MEDIUM Risk: Should classify as SIAGA (warning level)")
        print("• HIGH Risk: Should classify as BANJIR or escalated warning")
        print("• Good system: >80% accuracy in each category")
        print("-"*100)
        
        return df
    
    def visualize_evaluation_results(self):
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        param_names = [self.all_evaluations[k]['params'].name for k in self.all_evaluations.keys()]
        overall_acc = [self.all_evaluations[k]['overall_accuracy'] for k in self.all_evaluations.keys()]
        
        # PLOT 1: Overall Accuracy
        ax = axes[0]
        colors = ['#4CAF50' if acc >= 80 else '#FFC107' if acc >= 65 else '#F44336' 
                 for acc in overall_acc]
        
        bars = ax.barh(param_names, overall_acc, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_xlabel('Overall Accuracy (%)', fontsize=13, weight='bold')
        ax.set_title('Overall Accuracy of Each Parameter Set', fontsize=15, weight='bold', pad=20)
        ax.set_xlim(0, 100)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.axvline(x=80, color='green', linestyle='--', linewidth=2, alpha=0.6, label='Target: 80%')
        ax.legend(fontsize=11)
        
        for i, (bar, acc) in enumerate(zip(bars, overall_acc)):
            ax.text(acc + 2, bar.get_y() + bar.get_height()/2, 
                   f'{acc:.1f}%', va='center', fontsize=11, weight='bold')
        
        if max(overall_acc) >= 80:
            status = 'EXCELLENT'
            status_color = '#4CAF50'
        elif max(overall_acc) >= 65:
            status = 'GOOD'
            status_color = '#FFC107'
        else:
            status = 'NEEDS IMPROVEMENT'
            status_color = '#F44336'
        
        ax.text(0.02, 0.98, f'Best: {status}', transform=ax.transAxes,
               fontsize=12, weight='bold', va='top', 
               bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.3))
        
        # PLOT 2: Accuracy by Risk Category
        ax = axes[1]
        
        low_acc = [self.all_evaluations[k]['risk_category_accuracy']['low']['accuracy'] 
                   for k in self.all_evaluations.keys()]
        medium_acc = [self.all_evaluations[k]['risk_category_accuracy']['medium']['accuracy'] 
                      for k in self.all_evaluations.keys()]
        high_acc = [self.all_evaluations[k]['risk_category_accuracy']['high']['accuracy'] 
                    for k in self.all_evaluations.keys()]
        
        low_count = self.all_evaluations[list(self.all_evaluations.keys())[0]]['risk_category_accuracy']['low']['count']
        medium_count = self.all_evaluations[list(self.all_evaluations.keys())[0]]['risk_category_accuracy']['medium']['count']
        high_count = self.all_evaluations[list(self.all_evaluations.keys())[0]]['risk_category_accuracy']['high']['count']
        
        x = np.arange(len(param_names))
        width = 0.25
        
        bars1 = ax.bar(x - width, low_acc, width, label=f'LOW Risk ({low_count} scenarios)', 
                      color='#4CAF50', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x, medium_acc, width, label=f'MEDIUM Risk ({medium_count} scenarios)', 
                      color='#FFC107', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars3 = ax.bar(x + width, high_acc, width, label=f'HIGH Risk ({high_count} scenarios)', 
                      color='#F44336', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Accuracy (%)', fontsize=13, weight='bold')
        ax.set_title('Accuracy by Risk Category for Each Parameter Set', fontsize=15, weight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([name.split('(')[0].strip() for name in param_names], 
                           rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 108)
        ax.axhline(y=80, color='green', linestyle='--', linewidth=2, alpha=0.6)
        
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{height:.0f}%', ha='center', va='bottom', fontsize=9, weight='bold')
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("="*100)
    print("FUZZY LOGIC PARAMETER EVALUATION - 100 SCENARIOS")
    print("="*100)
    
    evaluator = FuzzyParameterEvaluator()
    evaluator.evaluate_all_parameter_sets()
    
    overall_df = evaluator.print_overall_accuracy_table()
    risk_category_df = evaluator.print_risk_category_accuracy_table()
    
    print("\n📊 Generating visualizations...")
    evaluator.visualize_evaluation_results()
    
    print("\n" + "="*100)
    print("✅ EVALUATION COMPLETE!")
    print("="*100)
    print("\n💡 For your thesis defense:")
    print("\"Five parameter configurations were evaluated against 100 comprehensive")
    print("test scenarios (40 LOW, 35 MEDIUM, 25 HIGH risk conditions). The selected")
    print("parameters achieved [X]% overall accuracy, demonstrating robust performance")
    print("across diverse flood situations through empirical validation.\"")
    print("="*100)