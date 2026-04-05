import numpy as np
import scipy.stats as stats
from typing import Any, Dict, List, Tuple
import pandas as pd

class StatisticalTests:
    """
    Computes statistical significance between evaluation runs.
    """
    def paired_ttest(self, baseline_results: List[float], system_results: List[float], alpha: float = 0.05) -> Dict[str, Any]:
        """
        Calculates T-Statistic and P-Value.
        """
        if len(baseline_results) == 0 or len(system_results) == 0:
            return {'t_stat': 0.0, 'p_value': 1.0, 'significant': False, 'effect_size_cohens_d': 0.0}
            
        t_stat, p_value = stats.ttest_rel(baseline_results, system_results)
        
        # Cohen's d for paired samples
        diff = np.array(baseline_results) - np.array(system_results)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1) if len(diff) > 1 else 1.0
        
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0
        
        return {
            't_stat': float(t_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < alpha),
            'effect_size_cohens_d': float(cohens_d)
        }
        
    def confidence_interval(self, data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """
        Returns +/- bounds around the mean.
        """
        if len(data) < 2:
            val = data[0] if len(data) == 1 else 0.0
            return (val, val)
            
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), stats.sem(a)
        h = se * stats.t.ppf((1 + confidence) / 2., n-1)
        return float(m - h), float(m + h)
        
    def run_full_comparison(self, all_baseline_results: Dict[str, List[float]], system_results: List[float]) -> pd.DataFrame:
        """
        Iterates over all baselines and tests against the target system results.
        Returns a Pandas DF suitable for LaTeX generation.
        """
        rows = []
        
        sys_mean = np.mean(system_results)
        sys_ci = self.confidence_interval(system_results)
        
        for b_name, b_data in all_baseline_results.items():
            b_mean = np.mean(b_data)
            b_ci = self.confidence_interval(b_data)
            
            test_res = self.paired_ttest(b_data, system_results)
            
            rows.append({
                "Baseline": b_name,
                "Baseline Mean TTF (Yrs)": f"{b_mean:.2f} ± {(b_mean - b_ci[0]):.2f}",
                "Ours Mean TTF (Yrs)": f"{sys_mean:.2f} ± {(sys_mean - sys_ci[0]):.2f}",
                "P-Value": f"{test_res['p_value']:.2e}",
                "Significant": "Yes" if test_res['significant'] else "No",
                "Cohen's d": f"{test_res['effect_size_cohens_d']:.2f}"
            })
            
        df = pd.DataFrame(rows)
        return df
