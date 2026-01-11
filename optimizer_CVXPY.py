import pandas as pd
import numpy as np
import cvxpy as cp
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
import warnings
import os

warnings.filterwarnings('ignore') # pour ignorer les avertissements non critiques de pd/CVXPY

"""
Ce script permet de construire un portefeuille obligataire optimisé qui minimise la tracking error
par rapport à un indice de référence (Benchmark), tout en réduisant considérablement le nombre de lignes.

Il utilise la programmation convexe pour aligner :
1. Duration
2. YTM
3. Maturité
4. Répartition géographique

Fonctionnalités clés :
- Exclusion dynamique (pays bannis, ESG, liquidité)
- Minimisation des coûts de transaction via pénalités
- Exportation des résultats en Excel
"""

class BondPortfolioOptimizer:
    def __init__(self, bond_file: str, liquidity_file: str):
        """
        Initialise l'optimiseur en chargeant et fusionnant les données de marché et de liquidité.
        """
        # On vérifie l'existence des fichiers
        if not os.path.exists(bond_file) or not os.path.exists(liquidity_file):
            raise FileNotFoundError(f"Au moins un fichier introuvable, check : {bond_file} ou {liquidity_file}")

        self.data = pd.read_csv(bond_file)
        self.liquidity = pd.read_csv(liquidity_file)

        # merge des données de marché avec coûts d'exec + scores de liquidité (contraintes de trading réel)
        self.data = pd.merge(self.data, self.liquidity, on='Country', how='left')

        # Si pas de données, on applique une forte pénalité (coût élevé, score nul)
        self.data['Execution_Cost_bps'].fillna(50, inplace=True)
        self.data['Liquidity_Score'].fillna(0, inplace=True)

        # Data clean
        self._clean_data()

        self.selection_pool = self.data.copy()  # univers complet AvANT filtres
        self.excluded_countries = []

        # calcul des parametres de référence
        self._calculate_benchmark()

    def _clean_data(self):
        """
        Nettoie le DataFrame :
        1. Vérifie la présence des colonnes obligatoires.
        2. Convertit les colonnes numériques.
        3. Supprime les lignes contenant des valeurs nulles critiques.
        """
        #1.
        required_cols = ['ISIN', 'Country', 'YLD_YTM_MID', 'DUR_ADJ_MID', 'MTY_YEARS', 'Region',
                         'w ytm', 'w dur', 'w maturity', 'Benchmark_Weight',
                         'Execution_Cost_bps', 'Liquidity_Score']

        missing = [c for c in required_cols if c not in self.data.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes dans le fichier source : {missing}")

        #2.
        cols_to_float = ['w ytm', 'w dur', 'w maturity', 'MTY_YEARS',
                         'YLD_YTM_MID', 'DUR_ADJ_MID', 'Benchmark_Weight',
                         'Execution_Cost_bps', 'Liquidity_Score']

        for col in cols_to_float:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        self.data['Region'] = self.data['Region'].astype(str)
        self.data['Country'] = self.data['Country'].astype(str)
        self.data['ISIN'] = self.data['ISIN'].astype(str)

        #3.
        self.data.dropna(subset=required_cols, inplace=True)

    def _calculate_benchmark(self):
        """
        Calcule les caractéristiques cibles de l'indice de référence (Benchmark).
        Ces valeurs serviront de cibles pour l'algorithme d'optimisation.
        Note : la somme des colonnes pondérées (w_) donne la moyenne pondérée de l'indice.
        """
        self.benchmark_w_ytm = self.data['w ytm'].sum()
        self.benchmark_w_dur = self.data['w dur'].sum()
        self.benchmark_w_maturity = self.data['w maturity'].sum()

        # Coût moyen et score de liquidité moyen de l'indice = pour comparaison
        if 'Benchmark_Weight' in self.data.columns:
            weights = self.data['Benchmark_Weight'] / self.data['Benchmark_Weight'].sum()
            self.benchmark_cost_bps = (self.data['Execution_Cost_bps'] * weights).sum()
            self.benchmark_liq_score = (self.data['Liquidity_Score'] * weights).sum()
        else:
            self.benchmark_cost_bps = self.data['Execution_Cost_bps'].mean()
            self.benchmark_liq_score = self.data['Liquidity_Score'].mean()

        # répartition géographique de l'indice
        region_weights = self.data.groupby('Region')['Benchmark_Weight'].sum()
        self.benchmark_region_dist = (region_weights / region_weights.sum()).to_dict()
        self.regions = list(self.benchmark_region_dist.keys())

    def apply_screens(self, countries_to_exclude: List[str], min_liquidity_score: int = 0):
        """
        Applique les filtres d'exclusion à l'univers d'investissement.

        Args:
            countries_to_exclude: liste des pays à bannir (ex: sanctions, politique interne, regles ESG)
            min_liquidity_score: score minimum (0-10) pour qu'une obligation soit éligible
        """
        self.excluded_countries = countries_to_exclude

        # 1. filtre géo
        mask_country = ~self.data['Country'].isin(countries_to_exclude)

        # 2. filtre de liquidité
        mask_liquidity = self.data['Liquidity_Score'] >= min_liquidity_score

        mask_final = mask_country & mask_liquidity
        self.selection_pool = self.data[mask_final].copy().reset_index(drop=True)

        # Log exclusion
        print(f"\nFiltres:")
        print(f" > Pays exclus : {countries_to_exclude}")
        print(f" > Score liquidité min : {min_liquidity_score}/10")
        print(f" > Univers final : {len(self.selection_pool)} obligations")

    def optimize_portfolio(self, num_bonds, lambda_ytm=1.0, lambda_dur=1.0,
                           lambda_maturity=1.0, lambda_region=1.0,
                           min_weight=0.0, max_weight=1.0, random_state=None) -> Dict:
        """
        Exécute une optimisation convexe pour un nombre donné d'obligations.

        Args:
            num_bonds: nombre d'actifs à sélectionner (cardinalité).
            lambda_*: pénalités pour chaque facteur de risque (importance relative).
            min_weight/max_weight: contraintes de poids par ligne (diversification).
            random_state: graine pour la sélection aléatoire initiale (reproductibilité).

        Returns:
            Dictionnaire contenant le statut de l'optimisation, les poids et les métriques.
        """

        # dummy filter : si l'univers filtré est plus petit que la demande = cancel
        if len(self.selection_pool) < num_bonds:
            return {'success': False, 'objective_value': np.inf}

        # Échantillonnage aléatoire pour réduire la dimensionnalité (Heuristique)
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(self.selection_pool), num_bonds, replace=False)
        selected_bonds = self.selection_pool.iloc[idx].reset_index(drop=True)

        n = len(selected_bonds)

        # Extraction des matrices de données
        ytm = selected_bonds['YLD_YTM_MID'].values
        dur = selected_bonds['DUR_ADJ_MID'].values
        mat = selected_bonds['MTY_YEARS'].values

        # mtrice d'expo par region
        region_matrix = np.zeros((n, len(self.regions)))
        for i, r in enumerate(self.regions):
            region_matrix[:, i] = (selected_bonds['Region'] == r).astype(float)
        bench_region_vec = np.array([self.benchmark_region_dist.get(r, 0) for r in self.regions])

        w = cp.Variable(n)  # poids à trouver

        # calcul des spreads par rapport au benchmark
        ytm_diff = w @ ytm - self.benchmark_w_ytm
        dur_diff = w @ dur - self.benchmark_w_dur
        mat_diff = w @ mat - self.benchmark_w_maturity
        region_diff = w @ region_matrix - bench_region_vec

        # Fonction utilité = minimiser somme des écarts au carré
        objective = cp.Minimize(
            lambda_ytm * cp.square(ytm_diff) +
            lambda_dur * cp.square(dur_diff) +
            lambda_maturity * cp.square(mat_diff) +
            lambda_region * cp.sum_squares(region_diff)
        )

        # Contraintes réglementaires et de gestion
        constraints = [
            cp.sum(w) == 1.0,
            w >= min_weight,  # 0 car long only, on ne vend pas a découvert
            w <= max_weight  # diversification par bond
        ]

        prob = cp.Problem(objective, constraints)

        # Résolution
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
        except cp.SolverError:
            prob.solve(verbose=False)

        # vérification si succès
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return {
                'success': False, 'objective_value': np.inf, 'weights': np.zeros(n),
                'w_ytm_spread': 999, 'w_dur_spread': 999, 'w_maturity_spread': 999, 'region_spread': {}
            }

        # cleaning des poids (zéro numérique)
        weights = w.value
        weights[weights < 1e-7] = 0
        weights /= weights.sum()  # renormalisation pour garantir 1

        # calcul des parametres post-optimisation
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
        """
        Cherche le nombre optimal d'obligations (Frontière efficiente nombre/erreur).
        Teste plusieurs tailles de portefeuille pour trouver le meilleur compromis.
        """
        results = []
        best_result = None
        best_objective = float('inf')

        # tqdm = barre de progression pour suivre l'optimisation
        for num_bonds in tqdm(range(min_bonds, max_bonds + 1), desc="Optimisation de la taille du portefeuille"):
            trial_results = []
            # MC : plusieurs essais aléatoires par taille pour éviter les minimums locaux !
            for trial in range(trials_per_count):
                seed = trial * 1000 + num_bonds
                res = self.optimize_portfolio(num_bonds=num_bonds, random_state=seed, **kwargs)
                if res['success']:
                    trial_results.append(res)

            if not trial_results:
                continue

            # On garde le meilleur essai pour cette taille
            best_trial = min(trial_results, key=lambda x: x['objective_value'])

            results.append({
                'num_bonds': num_bonds,
                'objective_value': best_trial['objective_value'],
                'w_ytm_spread': best_trial['w_ytm_spread'],
                'w_dur_spread': best_trial['w_dur_spread'],
                'w_maturity_spread': best_trial['w_maturity_spread'],
                'avg_region_spread': np.mean([abs(v) for v in best_trial['region_spread'].values()])
            })

            if best_trial['objective_value'] < best_objective: # mise à jour du "meilleur global"
                best_objective = best_trial['objective_value']
                best_result = best_trial

        return best_result, pd.DataFrame(results)

    def plot_optimization_results(self, results_df: pd.DataFrame):
        """ Génère les graphiques de convergence de l'erreur """
        if results_df.empty:
            print("Aucun résultat à afficher.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Erreur totale
        axes[0, 0].plot(results_df['num_bonds'], results_df['objective_value'], 'b-', linewidth=2)
        axes[0, 0].set_title('Tracking Error Total vs Taille Portefeuille')
        axes[0, 0].set_xlabel('Nombre de lignes')
        axes[0, 0].set_ylabel('Fonction Objectif (Minimiser)')

        # 2. Écart YTM
        axes[0, 1].plot(results_df['num_bonds'], results_df['w_ytm_spread'], 'g-', linewidth=2)
        axes[0, 1].set_title('Écart de w YTM')
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)

        # 3. Écart duration
        axes[1, 0].plot(results_df['num_bonds'], results_df['w_dur_spread'], 'r-', linewidth=2)
        axes[1, 0].set_title('Écart de w duration')
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)

        # Écart régional
        axes[1, 1].plot(results_df['num_bonds'], results_df['avg_region_spread'], 'm-', linewidth=2)
        axes[1, 1].set_title('Écart moyen allocation régionale')

        plt.tight_layout()
        plt.savefig('optimization_results.png')
        plt.show()
        plt.close()

if __name__ == "__main__":

    BOND_FILE = 'synthetic_bond_data.csv'
    LIQUIDITY_FILE = 'country_liquidity_costs.csv'

    if not os.path.exists(BOND_FILE) or not os.path.exists(LIQUIDITY_FILE):
        print("ERROR: Fichiers sources manquants")
        exit()

    optimizer = BondPortfolioOptimizer(BOND_FILE, LIQUIDITY_FILE)

    # PARAMÈTRES A MODIFIER SELON STRATÉGIE A METTRE EN PLACE
    # Ici on ne veut pas d'exposition à Russie/Israel et on veut un score de liquidité minimum de 3/10
    optimizer.apply_screens(countries_to_exclude=['Russia', 'Israel'], min_liquidity_score=3)

    # On teste des portefeuilles entre x et y bonds
    best_result, results_df = optimizer.find_optimal_bond_count(
        min_bonds=100,
        max_bonds=361,
        trials_per_count=20,  # nombre de simulations par taille (Monte Carlo)
        lambda_ytm=1.0,
        lambda_dur=1.0,
        lambda_maturity=1.0,
        lambda_region=1.0
    )

    if best_result is None:
        print("FAIL : l'optimisation n'a pas trouvé de solution.")
        exit()

    # Création du DataFrame final enrichi
    optimized_portfolio = pd.DataFrame({
        'ISIN': best_result['bonds']['ISIN'].values,
        'Country': best_result['bonds']['Country'].values,
        'Weight': best_result['weights'],
        'YTM': best_result['bonds']['YLD_YTM_MID'].values,
        'Duration': best_result['bonds']['DUR_ADJ_MID'].values,
        'Maturity': best_result['bonds']['MTY_YEARS'].values,
        'Region': best_result['bonds']['Region'].values,
        'Execution_Cost_bps': best_result['bonds']['Execution_Cost_bps'].values,
        'Liquidity_Score': best_result['bonds']['Liquidity_Score'].values,
    # Ponderation des résultats
        'w_ytm': best_result['weights'] * best_result['bonds']['YLD_YTM_MID'].values,
        'w_dur': best_result['weights'] * best_result['bonds']['DUR_ADJ_MID'].values,
        'w_maturity': best_result['weights'] * best_result['bonds']['MTY_YEARS'].values
    })

    # calcul des gains uniquement sur la meilleure liquidité
    port_cost_bps = (optimized_portfolio['Weight'] * optimized_portfolio['Execution_Cost_bps']).sum()
    savings = optimizer.benchmark_cost_bps - port_cost_bps

    # exportation vers un excel unique
    output_file = 'Final_Portfolio_Report.xlsx'

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Sheet 1 : composition du portefeuille
        optimized_portfolio.to_excel(writer, sheet_name='Portfolio_Composition', index=False)

        # Sheet 2 : courbe d'efficience
        results_df.to_excel(writer, sheet_name='Log_Opt', index=False)

        # Sheet 3 : analyse de l'impact de la nouvelle compo
        liq_df = pd.DataFrame({
            'Paramètre': ['Execution Cost (bps)', 'Liquidity Score', 'Yield (YTM)', 'Duration', 'Maturité'],
            'Benchmark': [
                optimizer.benchmark_cost_bps,
                optimizer.benchmark_liq_score,
                optimizer.benchmark_w_ytm,
                optimizer.benchmark_w_dur,
                optimizer.benchmark_w_maturity
            ],
            'Portfolio': [
                port_cost_bps,
                (optimized_portfolio['Weight'] * optimized_portfolio['Liquidity_Score']).sum(),
                best_result['optimized_w_ytm'],
                best_result['optimized_w_dur'],
                best_result['optimized_w_maturity']
            ],
            'Spreads': [
                f"-{savings:.2f}",
                "Plus haut = plus liquide",
                f"{best_result['w_ytm_spread']:.4f}",
                f"{best_result['w_dur_spread']:.4f}",
                f"{best_result['w_maturity_spread']:.4f}"
            ]
        })
        liq_df.to_excel(writer, sheet_name='Liquidity_Analysis', index=False)

    optimizer.plot_optimization_results(results_df)
    print(f" Terminé. Rapport disponible : {output_file}")

    print(f"{'Region':<20} | {'Benchmark':<10} | {'Optimal':<10} | {'Diff':<10}")
    print("*" * 60)

    for region in optimizer.regions:
        bench_val = optimizer.benchmark_region_dist.get(region, 0) * 100
        opt_val = best_result['optimized_region_dist'].get(region, 0) * 100
        spread = best_result['region_spread'].get(region, 0) * 100
        status = " "
        if abs(spread) > 0.02: status = "!!"  # alert si deviation > 0.02%

        print(f"{region:<20} | {bench_val:6.2f}%   | {opt_val:6.2f}%   | {spread:+6.2f}% {status}")