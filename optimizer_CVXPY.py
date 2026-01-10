import pandas as pd
import numpy as np
import cvxpy as cp
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')

"""
Quadratic Programming optimizer, il permet de trouver le portfeuille qui minimize les spreads definits initialement
(duration, YTM, maturity, distribution geographique). Cela est utile si nous voulons repliquer physiquement un indice
obligataire en baissant le nombre de eliminant differentes regions et en conservant les parametres de l'indice de base.
"""


class BondPortfolioOptimizer:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self._clean_data()

        self.selection_pool = self.data.copy() #sans exclusion = benchmark universe
        self.excluded_countries = []

        self._calculate_benchmark()

    def _clean_data(self):
        """ fonction pour:
            > selectionner des colonnes qui sont utilisees pour l'optimisation
            > test si colonnes sont absentes
            > clean les data (conversion + drop) """
        required_cols = ['ISIN', 'Country', 'YLD_YTM_MID', 'DUR_ADJ_MID', 'MTY_YEARS', 'Region',
                         'w ytm', 'w dur', 'w maturity', 'Benchmark_Weight']

        missing = [c for c in required_cols if c not in self.data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        self.data = self.data.dropna(subset=required_cols)
        cols_to_float = ['w ytm', 'w dur', 'w maturity', 'MTY_YEARS',
                         'YLD_YTM_MID', 'DUR_ADJ_MID', 'Benchmark_Weight']
        for col in cols_to_float:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        self.data['Region'] = self.data['Region'].astype(str)
        self.data['Country'] = self.data['Country'].astype(str)
        self.data['ISIN'] = self.data['ISIN'].astype(str)

        self.data = self.data.dropna()

    def _calculate_benchmark(self):
        """fonction pour calculer les poids de l'indice immutable de reference, important car une obligation
        peut compter 1 mais peser bien plus"""
        self.benchmark_w_ytm = self.data['w ytm'].sum()
        self.benchmark_w_dur = self.data['w dur'].sum()
        self.benchmark_w_maturity = self.data['w maturity'].sum()

        region_weights = self.data.groupby('Region')['Benchmark_Weight'].sum()
        self.benchmark_region_dist = (region_weights / region_weights.sum()).to_dict()
        self.regions = list(self.benchmark_region_dist.keys())

    def apply_country_exclusions(self, countries_to_exclude: List[str]):
        """ fonction qui permet d'exclure des pays dans l'indice synthetique qui est important pour repliquer
        efficacement l'indice de reference """
        self.excluded_countries = countries_to_exclude
        mask = ~self.data['Country'].isin(countries_to_exclude)
        self.selection_pool = self.data[mask].copy().reset_index(drop=True)
        print(f"\n[Tool Active] Excluding Countries: {countries_to_exclude}")
        print(f"              Pool reduced from {len(self.data)} to {len(self.selection_pool)} bonds.")

    def optimize_portfolio(self, num_bonds, lambda_ytm=1.0, lambda_dur=1.0,
                           lambda_maturity=1.0, lambda_region=1.0,
                           min_weight=0.0, max_weight=1.0, random_state=None) -> Dict:
        """
        fonction pour otpimiser le portfeuille, on choisit une optimization convexe CVXPY car elle permet d'eviter
        que l'optimizer ne reste bloquer dans des optimums locaux.
        La fonction s'articule autour de plusieurs etapes importantes:
        > sampling, on selectionne aleatoirement les obligations dans le panier vide des exclusions
        > clean les data pour qu'elles soient compatibles avec l'algorithme d'optimization CVXPY
        > definir le probleme que l'algorithme doit resoudre : mimiser les differences au carrees + les contraintes
        > solve avec l'algo definit sinon le plus proche possible
        > check si resultat est optimal sinon handicap le resultat
        > clean resultats pour l'export
        """

        if len(self.selection_pool) < num_bonds:
            raise ValueError(f"Requested {num_bonds} bonds, but only {len(self.selection_pool)} available.")

        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(self.selection_pool), num_bonds, replace=False)
        selected_bonds = self.selection_pool.iloc[idx].reset_index(drop=True)

        n = len(selected_bonds)

        ytm = selected_bonds['YLD_YTM_MID'].values
        dur = selected_bonds['DUR_ADJ_MID'].values
        mat = selected_bonds['MTY_YEARS'].values

        region_matrix = np.zeros((n, len(self.regions)))
        for i, r in enumerate(self.regions):
            region_matrix[:, i] = (selected_bonds['Region'] == r).astype(float)
        bench_region_vec = np.array([self.benchmark_region_dist.get(r, 0) for r in self.regions])

        w = cp.Variable(n)
        ytm_diff = w @ ytm - self.benchmark_w_ytm
        dur_diff = w @ dur - self.benchmark_w_dur
        mat_diff = w @ mat - self.benchmark_w_maturity
        region_diff = w @ region_matrix - bench_region_vec

        objective = cp.Minimize(
            lambda_ytm * cp.square(ytm_diff) +
            lambda_dur * cp.square(dur_diff) +
            lambda_maturity * cp.square(mat_diff) +
            lambda_region * cp.sum_squares(region_diff)
        )

        constraints = [
            cp.sum(w) == 1.0,
            w >= min_weight,
            w <= max_weight
        ]

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.OSQP, verbose=False)
        except cp.SolverError:
            prob.solve(verbose=False)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return {
                'success': False,
                'objective_value': np.inf,
                'weights': np.zeros(n),
                'w_ytm_spread': 999, 'w_dur_spread': 999, 'w_maturity_spread': 999, 'region_spread': {}
            }

        weights = w.value
        weights[weights < 1e-7] = 0
        weights /= weights.sum()  # on normalise les poids pour avoir 1

        optimized_w_ytm = np.dot(weights, ytm)
        optimized_w_dur = np.dot(weights, dur)
        optimized_w_maturity = np.dot(weights, mat)

        optimized_region_dist = {}
        for region in self.regions:
            mask = selected_bonds['Region'] == region
            optimized_region_dist[region] = np.sum(weights[mask])

        return {
            'success': True,
            'objective_value': prob.value,
            'weights': weights,
            'bonds': selected_bonds,
            'optimized_w_ytm': optimized_w_ytm,
            'optimized_w_dur': optimized_w_dur,
            'optimized_w_maturity': optimized_w_maturity,
            'optimized_region_dist': optimized_region_dist,
            'w_ytm_spread': optimized_w_ytm - self.benchmark_w_ytm,
            'w_dur_spread': optimized_w_dur - self.benchmark_w_dur,
            'w_maturity_spread': optimized_w_maturity - self.benchmark_w_maturity,
            'region_spread': {r: optimized_region_dist.get(r, 0) - self.benchmark_region_dist.get(r, 0) for r in
                              self.regions},
            'num_bonds': num_bonds
        }

    def find_optimal_bond_count(self, min_bonds, max_bonds, trials_per_count, **kwargs) -> Tuple[Dict, pd.DataFrame]:
        results = []
        best_result = None
        best_objective = float('inf')

        for num_bonds in tqdm(range(min_bonds, max_bonds + 1), desc="Testing bond counts"):
            trial_results = []
            for trial in range(trials_per_count):
                seed = trial * 1000 + num_bonds
                res = self.optimize_portfolio(num_bonds=num_bonds, random_state=seed, **kwargs)
                if res['success']:
                    trial_results.append(res)

            if not trial_results:
                continue

            best_trial = min(trial_results, key=lambda x: x['objective_value'])

            results.append({
                'num_bonds': num_bonds,
                'objective_value': best_trial['objective_value'],
                'w_ytm_spread': best_trial['w_ytm_spread'],
                'w_dur_spread': best_trial['w_dur_spread'],
                'w_maturity_spread': best_trial['w_maturity_spread'],
                'avg_region_spread': np.mean([abs(v) for v in best_trial['region_spread'].values()])
            })

            if best_trial['objective_value'] < best_objective:
                best_objective = best_trial['objective_value']
                best_result = best_trial

        return best_result, pd.DataFrame(results)

    def plot_optimization_results(self, results_df: pd.DataFrame):
        """Fonction pour plot les resultats"""
        if results_df.empty:
            print("No valid results to plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].plot(results_df['num_bonds'], results_df['objective_value'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Number of Bonds')
        axes[0, 0].set_ylabel('Objective Value')
        axes[0, 0].set_title('Total Tracking Error vs Portfolio Size')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(results_df['num_bonds'], results_df['w_ytm_spread'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Number of Bonds')
        axes[0, 1].set_ylabel('Weighted YTM Spread')
        axes[0, 1].set_title('Weighted YTM Deviation')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)

        axes[1, 0].plot(results_df['num_bonds'], results_df['w_dur_spread'], 'r-', linewidth=2)
        axes[1, 0].set_xlabel('Number of Bonds')
        axes[1, 0].set_ylabel('Weighted Duration Spread')
        axes[1, 0].set_title('Weighted Duration Deviation')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)

        axes[1, 1].plot(results_df['num_bonds'], results_df['avg_region_spread'], 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Number of Bonds')
        axes[1, 1].set_ylabel('Avg Region Spread')
        axes[1, 1].set_title('Region Distribution Deviation')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('optimization_results.png', dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    file_path = 'synthetic_bond_data.csv' # VERIFIER NOM DU FICHIER

    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please check the path.")
        exit()

    print("Initializing Bond Portfolio Optimizer (CVXPY Version)...")
    optimizer = BondPortfolioOptimizer(data)
    optimizer.apply_country_exclusions(['Mexico', 'Israel']) # ajouter pays selon impact sur liquidite, ...

    print(f"\nBenchmark Characteristics (Weighted Sums - Fixed):")
    print(f"Weighted YTM: {optimizer.benchmark_w_ytm:.6f}")
    print(f"Weighted Duration: {optimizer.benchmark_w_dur:.6f}")
    print(f"Weighted Maturity: {optimizer.benchmark_w_maturity:.6f}")
    print(f"Region Distribution: {optimizer.benchmark_region_dist}")

    print("\n" + "*" * 80)
    print("Finding optimal portfolio size...")
    print("*" * 80)

    # a changer selon l'horizon d'investissement, nous avons realiser un fichier de test pour rentrer les parametres
    # 'optimaux', check tests.py pour plus d'infos
    best_result, results_df = optimizer.find_optimal_bond_count(
        min_bonds=300,
        max_bonds=400,
        trials_per_count=5,
        lambda_ytm=1.0,
        lambda_dur=1.0,
        lambda_maturity=1.0,
        lambda_region=1.0,
        min_weight=0.0,
        max_weight=1.0
    )

    if best_result is None:
        print("Optimization failed to find a valid solution. Check constraints or bond count.")
        exit()

    print("\n" + "*" * 80)
    print("OPTIMIZATION RESULTS")
    print("*" * 80)

    print(f"\nBest Number of Bonds: {best_result['num_bonds']}")
    print(f"Objective Value: {best_result['objective_value']:.6f}")

    print(f"\nBenchmark vs Optimized (Weighted Sums):")
    print(
        f"  Weighted YTM - Benchmark: {optimizer.benchmark_w_ytm:.6f}, Optimized: {best_result['optimized_w_ytm']:.6f}, Spread: {best_result['w_ytm_spread']:.6f}")
    print(
        f"  Weighted Duration - Benchmark: {optimizer.benchmark_w_dur:.6f}, Optimized: {best_result['optimized_w_dur']:.6f}, Spread: {best_result['w_dur_spread']:.6f}")
    print(
        f"  Weighted Maturity - Benchmark: {optimizer.benchmark_w_maturity:.6f}, Optimized: {best_result['optimized_w_maturity']:.6f}, Spread: {best_result['w_maturity_spread']:.6f}")

    print(f"\nRegion Distribution Spread:")
    for region, spread in best_result['region_spread'].items():
        benchmark_pct = optimizer.benchmark_region_dist.get(region, 0) * 100
        optimized_pct = best_result['optimized_region_dist'].get(region, 0) * 100
        print(
            f"  {region}: Benchmark={benchmark_pct:.2f}%, Optimized={optimized_pct:.2f}%, Spread={spread * 100:+.2f}%")

    optimized_portfolio = pd.DataFrame({
        'ISIN': best_result['bonds']['ISIN'].values,
        'Country': best_result['bonds']['Country'].values,
        'Weight': best_result['weights'],
        'YTM': best_result['bonds']['YLD_YTM_MID'].values,
        'Duration': best_result['bonds']['DUR_ADJ_MID'].values,
        'Maturity': best_result['bonds']['MTY_YEARS'].values,
        'Region': best_result['bonds']['Region'].values,
        'w_ytm': best_result['weights'] * best_result['bonds']['YLD_YTM_MID'].values,
        'w_dur': best_result['weights'] * best_result['bonds']['DUR_ADJ_MID'].values,
        'w_maturity': best_result['weights'] * best_result['bonds']['MTY_YEARS'].values
    })


    print("\n" + "*" * 80)
    print("VERIFICATION")
    print("*" * 80)
    print(f"Sum of weights: {optimized_portfolio['Weight'].sum():.6f} (should be 1.0)")
    print(f"Computed weighted YTM: {optimized_portfolio['w_ytm'].sum():.6f}")
    print(f"Computed weighted Duration: {optimized_portfolio['w_dur'].sum():.6f}")
    print(f"Computed weighted Maturity: {optimized_portfolio['w_maturity'].sum():.6f}")

    banned_check = optimized_portfolio[optimized_portfolio['Country'].isin(['Mexico', 'Israel'])]
    if not banned_check.empty:
        print("CRITICAL WARNING: Banned countries found in portfolio!")
    else:
        print("Country Exclusion Check: PASSED")

    optimized_portfolio.to_csv('optimized_portfolio.csv', index=False)
    results_df.to_csv('optimization_summary.csv', index=False)
    optimizer.plot_optimization_results(results_df)