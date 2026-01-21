import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

try:
    from optimizer_CVXPY import BondPortfolioOptimizer
except ImportError:
    print("ERREUR CRITIQUE : Impossible d'importer 'BondPortfolioOptimizer'.")
    sys.exit(1)

warnings.filterwarnings('ignore')


class PortfolioValidator:
    def __init__(self, data_path='synthetic_bond_data.csv',
                 liq_path='country_liquidity_costs.csv'):
        """Initialisation du validateur et chargement de l'optimiseur."""
        print("Initialisation du validateur...")
        self.optimizer = BondPortfolioOptimizer(data_path, liq_path)

    def run_frontier_analysis(self, min_n=20, max_n=300, step=20, trials=5):
        """Test 1 : Analyse de la frontière efficiente (Taille vs Erreur)."""
        print(
            f"\n[Test 1] Analyse de la frontière ({min_n} à {max_n} titres)..."
        )
        results = []
        bond_counts = range(min_n, max_n + 1, step)

        for n in tqdm(bond_counts, desc="Calcul de la frontière"):
            res, _ = self.optimizer.find_optimal_bond_count(
                min_bonds=n, max_bonds=n, trials_per_count=trials
            )
            results.append({
                'N': n,
                'Tracking_Error': res['objective_value'],
                'YTM_Spread': abs(res['w_ytm_spread']),
                'Dur_Spread': abs(res['w_dur_spread'])
            })

        df = pd.DataFrame(results)

        # Tracé du graphique
        fig, ax1 = plt.subplots(figsize=(10, 6))
        color = 'tab:blue'
        ax1.set_xlabel("Nombre d'obligations")
        ax1.set_ylabel('Erreur Totale', color=color)
        ax1.plot(df['N'], df['Tracking_Error'], color=color, marker='o')

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Spread Absolu', color=color)
        ax2.plot(
            df['N'], df['YTM_Spread'],
            color=color, linestyle='--', label='YTM'
        )
        ax2.plot(
            df['N'], df['Dur_Spread'],
            color='green', linestyle=':', label='Duration'
        )

        plt.title('Frontière efficiente : taille vs qualité')
        plt.tight_layout()
        plt.savefig('test_frontier_plot.png')
        print("sauvegardé 'test_frontier_plot.png'")
        return df

    def run_stability_test(self, target_n=100, num_trials=50):
        """Test 2 : Test de stabilité du solveur (Histogramme)."""
        print(f"\n[Test 2] Test de stabilité (N={target_n})...")
        objectives = []
        for i in tqdm(range(num_trials)):
            res = self.optimizer.optimize_portfolio(
                num_bonds=target_n,
                random_state=i * 999
            )
            if res['success']:
                objectives.append(res['objective_value'])

        plt.figure(figsize=(10, 6))
        sns.histplot(objectives, kde=True, color='purple')
        plt.title(f'Distribution de stabilité (N={target_n})')
        plt.savefig('test_stability.png')
        print("sauvegardé 'test_stability.png'")

    def run_sampling_convergence_test(self, target_n=100, max_trials=100):
        """Test 4 : Convergence de l'échantillonnage."""
        print(f"\n[Test 4] Convergence ({1} à {max_trials} essais)...")

        # 1. Exécution massive de simulations
        objectives = []
        for i in tqdm(range(max_trials), desc="Simulations"):
            res = self.optimizer.optimize_portfolio(
                num_bonds=target_n,
                random_state=i * 42
            )
            if res['success']:
                objectives.append(res['objective_value'])

        if not objectives:
            print("Aucun essai réussi.")
            return

        # 2. Calcul du minimum cumulatif
        running_min = np.minimum.accumulate(objectives)

        # 3. Graphique
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(running_min) + 1),
            running_min,
            'b-', linewidth=2
        )
        plt.xlabel("Nombre d'essais réalisés")
        plt.ylabel('Meilleur objectif trouvé (min cumulatif)')
        plt.title('Convergence : quand arrêter les calculs ?')
        plt.grid(True, alpha=0.3)

        final_val = running_min[-1]
        plt.axhline(
            final_val, color='r', linestyle='--', alpha=0.5,
            label=f'Meilleur : {final_val:.2e}'
        )
        plt.legend()

        plt.savefig('test_convergence_sampling.png')
        print("sauvegardé 'test_convergence_sampling.png'")

    def run_lambda_sensitivity_test(self, target_n=100,
                                    param_name='lambda_dur',
                                    values=[0.1, 1.0, 10.0, 100.0, 500.0]):
        """Test 5 : Analyse de sensibilité des paramètres Lambda."""
        print(f"\n[Test 5] Sensibilité pour '{param_name}'...")

        results = []
        base_params = {
            'lambda_ytm': 1.0, 'lambda_dur': 1.0,
            'lambda_maturity': 1.0, 'lambda_region': 1.0
        }

        for val in tqdm(values, desc="Test des lambdas"):
            current_params = base_params.copy()
            current_params[param_name] = val

            res, _ = self.optimizer.find_optimal_bond_count(
                min_bonds=target_n, max_bonds=target_n, trials_per_count=10,
                **current_params
            )

            avg_reg_spread = np.mean([
                abs(v) for v in res['region_spread'].values()
            ])

            results.append({
                'Value': val,
                'Total_Error': res['objective_value'],
                'Dur_Spread': abs(res['w_dur_spread']),
                'YTM_Spread': abs(res['w_ytm_spread']),
                'Region_Spread': avg_reg_spread
            })

        df = pd.DataFrame(results)

        # Graphique des compromis (Trade-offs)
        fig, ax1 = plt.subplots(figsize=(10, 6))

        if 'dur' in param_name:
            target_metric = 'Dur_Spread'
            color = 'tab:green'
        elif 'ytm' in param_name:
            target_metric = 'YTM_Spread'
            color = 'tab:blue'
        else:
            target_metric = 'Region_Spread'
            color = 'tab:purple'

        ax1.set_xlabel(f'Valeur de {param_name} (Log Scale)')
        ax1.set_ylabel(
            f'{target_metric} (Cible)', color=color, fontweight='bold'
        )
        ax1.plot(
            df['Value'], df[target_metric],
            color=color, marker='o', linewidth=2
        )
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        color2 = 'gray'
        ax2.set_ylabel('Erreur Totale (Coût)', color=color2)
        ax2.plot(
            df['Value'], df['Total_Error'],
            color=color2, linestyle='--', marker='x'
        )
        ax2.tick_params(axis='y', labelcolor=color2)

        plt.title(f'Sensibilité : impact de {param_name}')
        plt.tight_layout()
        plt.savefig(f'test_sensitivity_{param_name}.png')
        print(f"sauvegardé 'test_sensitivity_{param_name}.png'")

    def run_stress_test_impossible_constraints(self):
        """Test 6 : Stress Test (contraintes impossibles)."""
        print("\n[Test 6] Stress test (contraintes impossibles)...")

        # 1. Simulation d'exclusion extrême (Top 3 régions)
        regions = list(self.optimizer.benchmark_region_dist.keys())
        excluded_regions = regions[:3]
        print(f"   Exclusion de : {excluded_regions}")

        self.optimizer.selection_pool = self.optimizer.data.copy()
        original_count = len(self.optimizer.selection_pool)

        # Filtre manuel par région
        self.optimizer.selection_pool = self.optimizer.data[
            ~self.optimizer.data['Region'].isin(excluded_regions)
        ].copy()
        new_count = len(self.optimizer.selection_pool)

        print(f"   Univers réduit de {original_count} à {new_count} titres.")

        # 2. Exécution de l'optimisation
        res = self.optimizer.optimize_portfolio(num_bonds=50)

        print(f"   Succès du solveur : {res['success']}")
        print(f"   Valeur objectif : {res['objective_value']:.4f}")

        # 3. Analyse
        if not res['success']:
            print("   -> SUCCÈS : le solveur a échoué comme prévu.")
        elif res['objective_value'] > 1.0:
            print("   -> SUCCÈS : solution trouvée mais erreur massive.")
        else:
            print("   -> ATTENTION : Le solveur a trouvé une solution.")

    def run_sanity_check_holdings(self):
        """Test 7 : Vérification de la qualité des positions (Poids min)."""
        print("\n[Test 7] Contrôle qualité portefeuille...")

        self.optimizer.selection_pool = self.optimizer.data.copy()
        res = self.optimizer.optimize_portfolio(num_bonds=100)

        if not res['success']:
            print("   Annulé (échec optimisation).")
            return

        holdings = res['bonds'].copy()
        holdings['Final_Weight'] = res['weights']
        active_holdings = holdings[holdings['Final_Weight'] > 1e-6]

        # Check 1 : dust (pos < 1bp)
        min_w = active_holdings['Final_Weight'].min()
        print(f"   Plus petite position : {min_w * 100:.4f}%")
        if min_w < 0.0001:
            print("   -> ATTENTION : positions insignifiantes détectées !")
        else:
            print("   -> OK : taille minimale respectée.")

        # Check 2 : concentration (> 10%)
        max_w = active_holdings['Final_Weight'].max()
        print(f"   Plus grosse position :  {max_w * 100:.2f}%")
        if max_w > 0.1:
            print("   -> ATTENTION : risque de concentration élevé !")
        else:
            print("   -> OK : concentration OK.")

        # Check 3 : Rendements nuls/faibles
        low_yield_threshold = 0.1
        junk_bonds = active_holdings[
            active_holdings['YLD_YTM_MID'] < low_yield_threshold
        ]

        if not junk_bonds.empty:
            print(f"   -> WARN : {len(junk_bonds)} titres sans rendement!")
        else:
            print("   -> PASS : Rendements valides :)")

    def run_determinism_test(self):
        """Test 8 : vérification du déterminisme en prod."""
        print("\n[Test 9] Test de déterminisme (Seed=42)...")

        # Opti 1
        res1 = self.optimizer.optimize_portfolio(
            num_bonds=100, random_state=42
        )
        isins1 = set(res1['bonds']['ISIN'].values)

        # Opti 2
        res2 = self.optimizer.optimize_portfolio(
            num_bonds=100, random_state=42
        )
        isins2 = set(res2['bonds']['ISIN'].values)

        diff = isins1.symmetric_difference(isins2)

        if len(diff) == 0:
            print("   -> SUCCÈS : portefeuilles 100% identiques.")
        else:
            print(f"   -> ÉCHEC : {len(diff)} différences détectées.")


if __name__ == "__main__":
    print("*" * 60)
    print("DÉMARRAGE DE LA SUITE DE VALIDATION")
    print("*" * 60)
    validator = PortfolioValidator()

    # PERFORMANCE
    validator.run_frontier_analysis(min_n=50, max_n=300, step=50, trials=5)
    validator.run_stability_test(target_n=100, num_trials=30)

    # TUNING
    validator.run_sampling_convergence_test(target_n=100, max_trials=50)

    validator.run_lambda_sensitivity_test(
        target_n=100,
        param_name='lambda_dur',
        values=[0.1, 1.0, 10.0, 100.0]
    )

    validator.run_stress_test_impossible_constraints()
    validator.run_sanity_check_holdings()
    validator.run_determinism_test()

    print("\n" + "*" * 60)
    print("TOUS LES TESTS TERMINÉS")
    print("*" * 60)