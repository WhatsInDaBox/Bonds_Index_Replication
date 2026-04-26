"""
model/optimizer.py
------------------
Responsabilité unique : résoudre le problème d'optimisation quadratique.
Meme modèle qu'au S1:
Reçoit un DataFrame filtré + des métriques benchmark calculées ailleurs.

Algorithme
----------
Minimise la tracking error pondérée (YTM, duration, maturité, régions)
plus un terme de pénalité sur les coûts de transaction, sous contraintes :
  - Somme des poids = 1
  - Poids individuels ∈ [0, max_weight]
  - Poids par pays ≤ max_country_weight
  - Poids par région ∈ [bench ± region_tolerance]

Solveur principal : CLARABEL. Fallback : OSQP.
En cas d'échec, relaxation des contraintes régionales et pays.
"""

from typing import Dict

import cvxpy as cp
import numpy as np
import pandas as pd

from config import OptimizationConfig
from helpers.metrics import compute_portfolio_metrics, compute_tracking_error
from logger import get_logger

logger = get_logger()


class BondOptimizer:
    """Optimiseur de portefeuille obligataire par programmation convexe."""

    def __init__(
        self,
        config: OptimizationConfig,
        benchmark_metrics: Dict,
    ):
        self.config = config
        self.bm = benchmark_metrics  # raccourci interne
        self.regions = benchmark_metrics["region_list"]

    # ------------------------------------------------------------------
    # Méthode principale
    # ------------------------------------------------------------------

    def optimize(self, pool: pd.DataFrame) -> Dict:
        """
        Construit et résout le problème d'optimisation.

        Parameters
        ----------
        pool : univers investissable filtré (BondRepository.pool).

        Returns
        -------
        Dictionnaire de résultat avec bonds, weights, métriques, tracking error.
        """
        cfg = self.config
        selected = pool.reset_index(drop=True)
        n = len(selected)

        if n < cfg.num_bonds:
            raise ValueError(
                f"Obligations insuffisantes : besoin de {cfg.num_bonds}, "
                f"disponibles {n}"
            )

        logger.info(
            f"Lancement optimisation, univers : {n} obligations  "
            f"Cible : {cfg.num_bonds} positions"
        )

        # Vecteurs de caractéristiques
        ytm = selected["YLD_YTM_MID"].values
        dur = selected["DUR_ADJ_MID"].values
        mat = selected["MTY_YEARS"].values
        costs = selected["Execution_Cost_bps"].values

        region_matrix, bench_region_vec = self._build_region_matrix(selected)

        # Variable d'optimisation
        w = cp.Variable(n)

        objective = self._build_objective(
            w, ytm, dur, mat, costs, region_matrix, bench_region_vec
        )
        constraints = self._build_constraints(
            w, selected, region_matrix, bench_region_vec
        )

        problem = cp.Problem(objective, constraints)
        self._solve(problem)

        # Relaxation si échec
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            logger.warning(
                f"Statut '{problem.status}' : relâchement contraintes régionales/pays"
            )
            relaxed = [cp.sum(w) == 1.0, w >= 0, w <= cfg.max_weight]
            problem = cp.Problem(objective, relaxed)
            self._solve(problem)

        return self._build_result(
            problem, w, selected, ytm, dur, mat, costs
        )

    # ------------------------------------------------------------------
    # Construction du problème
    # ------------------------------------------------------------------

    def _build_region_matrix(
        self, bonds: pd.DataFrame
    ):
        """Construit la matrice régionale + vecteur benchmark."""
        region_dummies = pd.get_dummies(bonds["Region"])
        for r in self.regions:
            if r not in region_dummies.columns:
                region_dummies[r] = 0
        region_matrix = region_dummies[self.regions].values.astype(float)
        bench_region_vec = np.array(
            [self.bm["regions"].get(r, 0.0) for r in self.regions]
        )
        return region_matrix, bench_region_vec

    def _build_objective(
        self,
        w: cp.Variable,
        ytm: np.ndarray,
        dur: np.ndarray,
        mat: np.ndarray,
        costs: np.ndarray,
        region_matrix: np.ndarray,
        bench_region_vec: np.ndarray,
    ) -> cp.Minimize:
        """Construit l'objectif quadratique (tracking error + coûts)."""
        cfg = self.config
        bm = self.bm

        ytm_err = (w @ ytm - bm["ytm"]) / bm["ytm"]
        dur_err = (w @ dur - bm["duration"]) / bm["duration"]
        mat_err = (w @ mat - bm["maturity"]) / bm["maturity"]
        region_err = w @ region_matrix - bench_region_vec

        return cp.Minimize(
            cfg.lambda_ytm * cp.square(ytm_err)
            + cfg.lambda_dur * cp.square(dur_err)
            + cfg.lambda_maturity * cp.square(mat_err)
            + (cfg.lambda_region * 5) * cp.sum_squares(region_err)
            + cfg.lambda_cost * (w @ costs) / 100
        )

    def _build_constraints(
        self,
        w: cp.Variable,
        bonds: pd.DataFrame,
        region_matrix: np.ndarray,
        bench_region_vec: np.ndarray,
    ) -> list:
        """Construit les contraintes (somme, max, pays, régions)."""
        cfg = self.config

        constraints = [
            cp.sum(w) == 1.0,
            w >= 0,
            w <= cfg.max_weight,
        ]

        # Contraintes par pays
        for country in bonds["Country"].unique():
            mask = (bonds["Country"] == country).values
            constraints.append(
                cp.sum(w[mask]) <= cfg.max_country_weight
            )

        # Contraintes régionales avec tolérance
        for i, region in enumerate(self.regions):
            bench_w = bench_region_vec[i]
            max_possible = np.sum(region_matrix[:, i]) * cfg.max_weight

            if max_possible < (bench_w - cfg.region_tolerance):
                logger.warning(
                    f"Contrainte région '{region}' ignorée (géométriquement infaisable)"
                )
                continue

            if bench_w > 0.01:
                constraints.extend([
                    w @ region_matrix[:, i] >= max(0.0, bench_w - cfg.region_tolerance),
                    w @ region_matrix[:, i] <= min(1.0, bench_w + cfg.region_tolerance),
                ])

        return constraints

    # ------------------------------------------------------------------
    # Résolution
    # ------------------------------------------------------------------

    def _solve(self, problem: cp.Problem) -> None:
        """Tente CLARABEL, bascule sur OSQP en cas d'échec."""
        try:
            problem.solve(
                solver=cp.CLARABEL,
                verbose=self.config.debug,
            )
            logger.debug("Solveur CLARABEL utilisé")
        except Exception as exc:
            logger.warning(f"CLARABEL échoué ({exc}), bascule sur OSQP")
            problem.solve(solver=cp.OSQP, verbose=self.config.debug)

    # ------------------------------------------------------------------
    # Résultat output
    # ------------------------------------------------------------------

    def _build_result(
        self,
        problem: cp.Problem,
        w: cp.Variable,
        selected: pd.DataFrame,
        ytm: np.ndarray,
        dur: np.ndarray,
        mat: np.ndarray,
        costs: np.ndarray,
    ) -> Dict:
        """Nettoie les poids, sélectionne le Top N, calcule les métriques."""
        cfg = self.config

        raw_weights = w.value.copy()
        raw_weights[raw_weights < cfg.min_weight] = 0.0

        top_idx = np.argsort(raw_weights)[-cfg.num_bonds:]
        mask = np.zeros(len(selected), dtype=bool)
        mask[top_idx] = True

        final_weights = np.zeros(len(selected))
        final_weights[mask] = raw_weights[mask]
        final_weights /= final_weights.sum()

        active = final_weights > 0
        portfolio_bonds = selected[active].copy().reset_index(drop=True)
        portfolio_weights = final_weights[active]

        port_metrics = compute_portfolio_metrics(
            portfolio_bonds, portfolio_weights, self.regions
        )
        te = compute_tracking_error(port_metrics, self.bm)

        n_active = int(np.sum(active))
        logger.info(
            f"Optimisation terminée: {n_active} positions  "
            f"YTM: {port_metrics['ytm']:.4f}  "
            f"Dur: {port_metrics['duration']:.4f}  "
            f"Obj: {problem.value:.6f}"
        )

        return {
            "success": True,
            "num_bonds": n_active,
            "bonds": portfolio_bonds,
            "weights": portfolio_weights,
            "objective_value": problem.value,
            "portfolio_ytm": port_metrics["ytm"],
            "portfolio_dur": port_metrics["duration"],
            "portfolio_maturity": port_metrics["maturity"],
            "portfolio_cost": port_metrics["cost"],
            "portfolio_liquidity": port_metrics["liquidity"],
            "portfolio_regions": port_metrics["regions"],
            "tracking_error": te,
        }
