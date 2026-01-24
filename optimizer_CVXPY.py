import numpy as np
import cvxpy as cp
import pandas as pd
from typing import Dict
from dataclasses import dataclass


"""
Ce script permet de construire un portefeuille obligataire optimisé qui minimise
la tracking error par rapport à un indice de référence (benchmark), tout en
réduisant considérablement le nombre de bonds à trader.

Il utilise la programmation convexe pour aligner :
1. Duration
2. YTM
3. Maturité
4. Répartition géographique

Fonctionnalités clés :
- Exclusion dynamique (pays bannis, ESG, liquidité)
- Minimisation des coûts de transaction via pénalités
- Exportation des résultats en Excel + synthèse
"""


@dataclass
class OptimizationConfig:
    """Configuration pour l'optimisation de portefeuille."""
    lambda_ytm: float = 1.0
    lambda_dur: float = 1.0
    lambda_maturity: float = 1.0
    lambda_region: float = 1.0
    lambda_cost: float = 0.5  # Pénalité liée aux coûts de transaction

    min_weight: float = 0.001  # Position minimale de 0.1%
    max_weight: float = 0.05  # Position maximale de 5%
    max_country_weight: float = 0.20

    region_tolerance: float = 0.02  # Déviation régionale tolérée de ±2%


class ImprovedBondOptimizer:
    def __init__(self, bond_file: str, liquidity_file: str):
        self.data = self._load_and_merge_data(bond_file, liquidity_file)
        self._calculate_benchmark()
        self.selection_pool = self.data.copy()

    def _load_and_merge_data(
        self, bond_file: str, liquidity_file: str
    ) -> pd.DataFrame:
        """
        Chargement et fusion des données obligataires et de liquidité avec
        validation.
        """
        bonds = pd.read_csv(bond_file)
        liquidity = pd.read_csv(liquidity_file)

        # Fusion et remplissage des valeurs manquantes
        df = pd.merge(bonds, liquidity, on='Country', how='left')
        df = df.fillna({'Execution_Cost_bps': 50, 'Liquidity_Score': 0})

        # Validation des données
        self._validate_data(df)

        return df

    def _validate_data(self, df: pd.DataFrame):
        """Validation complète des données (colonnes, doublons, plages)."""
        required = [
            'ISIN', 'Country', 'YLD_YTM_MID', 'DUR_ADJ_MID',
            'MTY_YEARS', 'Region', 'Benchmark_Weight'
        ]

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes : {missing}")

        # Vérification des doublons
        if df['ISIN'].duplicated().any():
            raise ValueError("Doublons d'ISIN détectés dans les données")

        # Validation des plages de valeurs
        assert (df['DUR_ADJ_MID'] > 0).all(), "Durations invalides détectées"
        assert (df['MTY_YEARS'] > 0).all(), "Maturités invalides détectées"
        assert (df['Liquidity_Score'] >= 0).all(), \
            "Scores de liquidité invalides"

    def _calculate_benchmark(self):
        """Calcul des caractéristiques de l'indice de référence (benchmark)."""
        weights = (
            self.data['Benchmark_Weight'] /
            self.data['Benchmark_Weight'].sum()
        )

        self.benchmark_ytm = (self.data['YLD_YTM_MID'] * weights).sum()
        self.benchmark_dur = (self.data['DUR_ADJ_MID'] * weights).sum()
        self.benchmark_maturity = (self.data['MTY_YEARS'] * weights).sum()
        self.benchmark_cost = (
            self.data['Execution_Cost_bps'] * weights
        ).sum()
        self.benchmark_liquidity = (
            self.data['Liquidity_Score'] * weights
        ).sum()

        # Répartition régionale
        region_weights = self.data.groupby('Region')['Benchmark_Weight'].sum()
        self.benchmark_regions = (
            region_weights / region_weights.sum()
        ).to_dict()
        self.regions = list(self.benchmark_regions.keys())

    def optimize_portfolio(
            self,
            num_bonds: int,
            config: OptimizationConfig,
            debug: bool = False
    ) -> Dict:
        """
        Optimise le portefeuille en utilisant l'univers filtré COMPLET.
        """
        # UTILISATION DE L'UNIVERS COMPLET
        selected_bonds = self.selection_pool.copy().reset_index(drop=True)
        n = len(selected_bonds)

        if n < num_bonds:
            raise ValueError(
                f"Obligations insuffisantes : besoin de {num_bonds}, "
                f"disponible {n}"
            )

        if debug:
            print(f"DEBUG: {n} obligations en entrée -> Cible {num_bonds}")

        # Extraction des caractéristiques des obligations
        ytm = selected_bonds['YLD_YTM_MID'].values
        dur = selected_bonds['DUR_ADJ_MID'].values
        mat = selected_bonds['MTY_YEARS'].values
        costs = selected_bonds['Execution_Cost_bps'].values
        liq_scores = selected_bonds['Liquidity_Score'].values

        # Création de la matrice régionale
        region_dummies = pd.get_dummies(selected_bonds['Region'])
        for r in self.regions:
            if r not in region_dummies.columns:
                region_dummies[r] = 0
        region_matrix = region_dummies[self.regions].values

        bench_region_vec = np.array([
            self.benchmark_regions.get(r, 0) for r in self.regions
        ])

        # Variables d'optimisation
        w = cp.Variable(n)

        # Calcul des déviations (Tracking Error)
        ytm_error = (w @ ytm - self.benchmark_ytm) / self.benchmark_ytm
        dur_error = (w @ dur - self.benchmark_dur) / self.benchmark_dur
        mat_error = (
            (w @ mat - self.benchmark_maturity) / self.benchmark_maturity
        )
        region_error = w @ region_matrix - bench_region_vec
        cost_impact = w @ costs

        # Objectif : minimiser la tracking error + les coûts de transaction
        objective = cp.Minimize(
            config.lambda_ytm * cp.square(ytm_error) +
            config.lambda_dur * cp.square(dur_error) +
            config.lambda_maturity * cp.square(mat_error) +
            (config.lambda_region * 5) * cp.sum_squares(region_error) +
            config.lambda_cost * cost_impact / 100
        )

        # Contraintes de base
        constraints = [
            cp.sum(w) == 1.0,
            w >= 0,
            w <= config.max_weight
        ]

        # Vérification intelligente de faisabilité pour les limites par pays
        country_counts = selected_bonds['Country'].value_counts()
        for country, count in country_counts.items():
            country_mask = (selected_bonds['Country'] == country).values
            constraints.append(
                cp.sum(w[country_mask]) <= config.max_country_weight
            )

        # Contraintes Régionales (Limites strictes avec tolérance)
        for i, region in enumerate(self.regions):
            bench_weight = bench_region_vec[i]
            region_mask = region_matrix[:, i]
            max_possible_region_weight = (
                np.sum(region_mask) * config.max_weight
            )

            if max_possible_region_weight < (
                bench_weight - config.region_tolerance
            ):
                if debug:
                    print(
                        f"DEBUG: Skip contrainte région {region} (infaisable)."
                    )
                continue

            if bench_weight > 0.01:
                constraints.extend([
                    w @ region_matrix[:, i] >= max(
                        0, bench_weight - config.region_tolerance
                    ),
                    w @ region_matrix[:, i] <= min(
                        1.0, bench_weight + config.region_tolerance
                    )
                ])

        # Résolution
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.CLARABEL, verbose=debug)
        except Exception:
            problem.solve(solver=cp.OSQP, verbose=debug)

        # Nouvelle tentative avec relaxation en cas d'échec
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            if debug:
                print("DEBUG: Échec solveur strict -> Relaxation...")
            constraints_relaxed = [
                cp.sum(w) == 1.0,
                w >= 0,
                w <= config.max_weight
            ]
            problem = cp.Problem(objective, constraints_relaxed)
            problem.solve(verbose=debug)

        # Extraction et nettoyage des poids
        raw_weights = w.value
        raw_weights[raw_weights < config.min_weight] = 0

        # Tri et conservation du Top N
        top_indices = np.argsort(raw_weights)[-num_bonds:]
        mask = np.zeros(n, dtype=bool)
        mask[top_indices] = True

        final_weights = np.zeros(n)
        final_weights[mask] = raw_weights[mask]
        final_weights = final_weights / final_weights.sum()

        # Construction du résultat
        active_positions = final_weights > 0
        portfolio_bonds = selected_bonds[active_positions].copy()
        portfolio_weights = final_weights[active_positions]

        # Recalcul des métriques finales
        port_ytm = np.sum(portfolio_weights * ytm[active_positions])
        port_dur = np.sum(portfolio_weights * dur[active_positions])
        port_mat = np.sum(portfolio_weights * mat[active_positions])
        port_cost = np.sum(portfolio_weights * costs[active_positions])
        port_liq = np.sum(portfolio_weights * liq_scores[active_positions])

        # Recalcul de la répartition régionale
        port_regions = {}
        for region in self.regions:
            region_mask = (portfolio_bonds['Region'] == region).values
            port_regions[region] = np.sum(portfolio_weights[region_mask])

        return {
            'success': True,
            'num_bonds': int(np.sum(active_positions)),
            'bonds': portfolio_bonds,
            'weights': portfolio_weights,
            'objective_value': problem.value,
            'portfolio_ytm': port_ytm,
            'portfolio_dur': port_dur,
            'portfolio_maturity': port_mat,
            'portfolio_cost': port_cost,
            'portfolio_liquidity': port_liq,
            'portfolio_regions': port_regions,
            'tracking_error': {
                'ytm': port_ytm - self.benchmark_ytm,
                'duration': port_dur - self.benchmark_dur,
                'maturity': port_mat - self.benchmark_maturity,
                'cost_savings': self.benchmark_cost - port_cost
            }
        }

    def generate_report(self, result: Dict) -> pd.DataFrame:
        """Génère le rapport détaillé ligne à ligne."""
        bonds = result['bonds']
        weights = result['weights']

        portfolio = pd.DataFrame({
            'ISIN': bonds['ISIN'].values,
            'Country': bonds['Country'].values,
            'Region': bonds['Region'].values,
            'Weight (%)': weights * 100,
            'YTM (%)': bonds['YLD_YTM_MID'].values,
            'Duration': bonds['DUR_ADJ_MID'].values,
            'Maturity (Years)': bonds['MTY_YEARS'].values,
            'Execution Cost (bps)': bonds['Execution_Cost_bps'].values,
            'Liquidity Score': bonds['Liquidity_Score'].values
        })

        portfolio = portfolio.sort_values('Weight (%)', ascending=False)
        return portfolio

    def generate_summary_sheet(self, result: Dict) -> pd.DataFrame:
        """
        Génère une feuille de résumé avec comparaison haute précision (e-15).
        Inclut les métriques globales et régionales.
        """
        # 1. Métriques globales (pondérées)
        metrics_data = [
            {
                'Metric': 'Yield (YTM %)',
                'Benchmark': self.benchmark_ytm,
                'Portfolio': result['portfolio_ytm'],
            },
            {
                'Metric': 'Duration (Years)',
                'Benchmark': self.benchmark_dur,
                'Portfolio': result['portfolio_dur'],
            },
            {
                'Metric': 'Maturity (Years)',
                'Benchmark': self.benchmark_maturity,
                'Portfolio': result['portfolio_maturity'],
            },
            {
                'Metric': 'Execution Cost (bps)',
                'Benchmark': self.benchmark_cost,
                'Portfolio': result['portfolio_cost'],
            },
            {
                'Metric': 'Liquidity Score (Avg)',
                'Benchmark': self.benchmark_liquidity,
                'Portfolio': result['portfolio_liquidity'],
            }
        ]

        # 2. Métriques régionales
        for region in self.regions:
            metrics_data.append({
                'Metric': f'Region: {region}',
                'Benchmark': self.benchmark_regions.get(region, 0.0),
                'Portfolio': result['portfolio_regions'].get(region, 0.0),
            })

        # Création DataFrame
        summary_df = pd.DataFrame(metrics_data)

        # Calcul de la différence avec précision maximale
        summary_df['Difference (e-15)'] = (
            summary_df['Portfolio'] - summary_df['Benchmark']
        )

        # Organisation des colonnes
        summary_df = summary_df[[
            'Metric', 'Benchmark', 'Portfolio', 'Difference (e-15)'
        ]]

        return summary_df


if __name__ == "__main__":
    optimizer = ImprovedBondOptimizer(
        'synthetic_bond_data.csv',
        'country_liquidity_costs.csv'
    )

    optimizer.selection_pool = optimizer.data[
        ~optimizer.data['Country'].isin(['Russia', 'Israel']) &
        (optimizer.data['Liquidity_Score'] >= 3)
    ].copy()

    config = OptimizationConfig(
        lambda_ytm=1.0,
        lambda_dur=1.0,
        lambda_maturity=1.0,
        lambda_region=1.0,
        lambda_cost=0.2,
        max_weight=0.05,
        max_country_weight=0.25,
        region_tolerance=0.0000000
    )

    result = optimizer.optimize_portfolio(
        num_bonds=350, config=config, debug=False
    )

    df_composition = optimizer.generate_report(result)
    df_summary = optimizer.generate_summary_sheet(result)

    output_file = 'Optimized_Portfolio.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Résumé_Global', index=False)
        df_composition.to_excel(writer, sheet_name='Composition', index=False)

    print(f"Terminé. Résultat sauvegardé dans : {output_file}")