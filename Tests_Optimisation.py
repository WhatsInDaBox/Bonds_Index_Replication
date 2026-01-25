import os
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from optimizer_CVXPY import ImprovedBondOptimizer, OptimizationConfig
except ImportError:
    print("ERREUR CRITIQUE : Impossible d'importer 'ImprovedBondOptimizer' "
          "depuis 'optimizer_CVXPY.py'.")
    sys.exit(1)

warnings.filterwarnings('ignore')


class PortfolioValidatorV2:
    def __init__(self, data_path='synthetic_bond_data.csv',
                 liq_path='country_liquidity_costs.csv'):
        """Initialisation du validateur avec l'optimiseur"""
        print("Initialisation du validateur ...")
        self.optimizer = ImprovedBondOptimizer(data_path, liq_path)

        # Configuration de base pour les tests
        self.base_config = OptimizationConfig(
            lambda_ytm=1.0,
            lambda_dur=1.0,
            lambda_maturity=1.0,
            lambda_region=2.0,  # Priorité élevée pour les tests régionaux
            lambda_cost=0.5,
            max_weight=0.10,
            max_country_weight=0.20,
            region_tolerance=0
        )

    def run_frontier_analysis(self, min_n=50, max_n=900, step=50):
        """
        Test 1 : Analyse de la frontière efficiente (taille vs tracking error).
        """
        print(f"\n[Test 1] Analyse de la frontière efficiente "
              f"({min_n} à {max_n} titres)...")
        results = []
        bond_counts = range(min_n, max_n + 1, step)

        for n in tqdm(bond_counts, desc="Calcul de la frontière"):
            try:
                res = self.optimizer.optimize_portfolio(
                    num_bonds=n, config=self.base_config
                )

                te = res['tracking_error']

                max_reg_dev = 0
                for r, weight in res['portfolio_regions'].items():
                    bench = self.optimizer.benchmark_regions.get(r, 0)
                    dev = abs(weight - bench)
                    if dev > max_reg_dev:
                        max_reg_dev = dev

                results.append({
                    'N': n,
                    'Objective_Value': res['objective_value'],
                    'YTM_Error_bps': abs(te['ytm']) * 10000,
                    'Dur_Error': abs(te['duration']),
                    'Max_Region_Dev_Pct': max_reg_dev * 100
                })
            except Exception as e:
                print(f"  Erreur à N={n} : {e}")

        df = pd.DataFrame(results)

        # Génération du graphique
        fig, ax1 = plt.subplots(figsize=(12, 7))

        color = 'tab:blue'
        ax1.set_xlabel("Nombre d'obligations (N)")
        ax1.set_ylabel(
            'Valeur objectif (erreur totale)',
            color=color, fontweight='bold'
        )
        ax1.plot(
            df['N'], df['Objective_Value'],
            color=color, marker='o', linewidth=2, label='Erreur totale'
        )
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel(
            'Déviation régionale max (%)',
            color=color, fontweight='bold'
        )
        ax2.plot(
            df['N'], df['Max_Region_Dev_Pct'],
            color=color, linestyle='--', marker='x', label='Déviation Rég.'
        )
        ax2.tick_params(axis='y', labelcolor=color)

        ax2.axhline(
            y=1.0, color='gray', linestyle=':', alpha=0.7,
            label='Tolérance 1%'
        )

        plt.title('Frontière efficiente V2 : taille du portefeuille vs qualité')
        fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.85))
        plt.tight_layout()
        plt.savefig('test_v2_frontier.png')
        print("Sauvegardé 'test_v2_frontier.png'")
        return df


    def run_stress_test_universe_scarcity(self):
        """Test 2 : Stress test - rareté de l'univers d'investissement."""
        print("\n[Test 3] Stress test : rareté de l'univers...")

        # Instance temporaire pour le test
        crippled_optimizer = ImprovedBondOptimizer(
            'synthetic_bond_data.csv', 'country_liquidity_costs.csv'
        )

        # Filtrage brutal : suppression des 3 premières régions
        regions = list(crippled_optimizer.regions)
        excluded = regions[:3]
        print(f"   Régions exclues : {excluded}")

        crippled_optimizer.selection_pool = crippled_optimizer.data[
            ~crippled_optimizer.data['Region'].isin(excluded)
        ].copy()

        remaining = len(crippled_optimizer.selection_pool)
        print(f"   Univers réduit à {remaining} titres.")

        try:
            res = crippled_optimizer.optimize_portfolio(
                num_bonds=400, config=self.base_config
            )

            print(f"   Statut du solveur : "
                  f"{'Succès' if res['success'] else 'Échec'}")
            print(f"   Valeur Objectif : {res['objective_value']:.4f}")

            if res['success']:
                print("   -> SUCCÈS : Le solveur a géré la rareté "
                      "(les contraintes souples ont fonctionné).")
                print(f"   -> Allocations régionales : "
                      f"{res['portfolio_regions']}")
            else:
                print("   -> ÉCHEC : Le solveur a planté.")

        except Exception as e:
            print(f"   -> EXCEPTION : {e}")

    def run_sanity_check_holdings(self):
        """Test 3 : Contrôle qualité des positions (poids min/max, etc)."""
        print("\n[Test 4] Contrôle qualité des positions...")

        res = self.optimizer.optimize_portfolio(
            num_bonds=400, config=self.base_config
        )

        if not res['success']:
            print("   Annulé (échec de l'optimisation).")
            return

        weights = res['weights']
        active_mask = weights > 1e-6
        active_weights = weights[active_mask]

        min_w = active_weights.min()
        print(f"   Plus petite position : {min_w * 100:.5f}%")
        if min_w < 0.0001:  # 0.01%
            print("   -> ATTENTION : Positions infimes détectées (< 1bp). "
                  "Logique de troncature requise.")
        else:
            print("   -> OK : Taille minimale respectée.")

        max_w = active_weights.max()
        print(f"   Plus grande position : {max_w * 100:.2f}%")
        limit = self.base_config.max_weight
        if max_w > limit + 0.001:  # Tolérance float
            print(f"   -> ÉCHEC : Contrainte violée (Max autorisé : "
                  f"{limit * 100}%)")
        else:
            print("   -> OK : Limites de concentration respectées.")

        print("   Conformité régionale :")
        for r in self.optimizer.regions:
            bench = self.optimizer.benchmark_regions.get(r, 0)
            port = res['portfolio_regions'].get(r, 0)
            diff = port - bench
            status = "OK" if abs(diff) < 0.01 else "DRIFT"
            print(f"     - {r:<15}: {port * 100:6.2f}% "
                  f"(Bench: {bench * 100:6.2f}%) -> "
                  f"{diff * 100:+5.2f}% [{status}]")

    def run_determinism_test(self):
        """
        Test 4 : Vérification du déterminisme
        (Deux exécutions donnent le meme résultat EXACT).
        """
        print("\n[Test 5] Vérification du déterminisme...")

        # Exec 1
        res1 = self.optimizer.optimize_portfolio(
            num_bonds=400, config=self.base_config
        )
        w1 = res1['weights']

        # Exec 2
        res2 = self.optimizer.optimize_portfolio(
            num_bonds=400, config=self.base_config
        )
        w2 = res2['weights']

        if np.allclose(w1, w2, atol=1e-8):
            print("   -> SUCCÈS : L'optimisation est déterministe.")
        else:
            diff = np.sum(np.abs(w1 - w2))
            print(f"   -> ÉCHEC : Les résultats diffèrent ! "
                  f"Delta total des poids : {diff}")


if __name__ == "__main__":
    print("-" * 60)
    print("Démarrage des tests")
    print("-" * 60)

    if not os.path.exists('synthetic_bond_data.csv'):
        print("Erreur : 'synthetic_bond_data.csv' introuvable. "
              "Veuillez générer/updater les données.")
        sys.exit(1)

    validator = PortfolioValidatorV2()

    validator.run_frontier_analysis(min_n=50, max_n=500, step=50)

    validator.run_stress_test_universe_scarcity()
    validator.run_sanity_check_holdings()
    validator.run_determinism_test()

    print("\n" + "-" * 60)
    print("Tests termines")
    print("-" * 60)