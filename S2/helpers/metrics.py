"""
helpers/metrics.py
------------------
Fonctions utilitaires de calcul financier.

- Calcul des caractéristiques pondérées du benchmark
- Calcul des métriques d'un portefeuille donné
- Récapitulatif de tracking error

Ces fonctions sont pures et réutilisables indépendamment
de la logique d'optimisation.
"""

from typing import Dict, List

import numpy as np
import pandas as pd


def compute_benchmark_metrics(data: pd.DataFrame) -> Dict:
    """
    Calcule les caractéristiques pondérées de l'indice de référence.

    Parameters:
    data : pd.DataFrame
        Univers complet avec colonne `Benchmark_Weight`.

    Returns:
    dict avec les clés :
        ytm, duration, maturity, cost, liquidity, regions, region_list
    """
    weights = data["Benchmark_Weight"] / data["Benchmark_Weight"].sum()

    region_weights = data.groupby("Region")["Benchmark_Weight"].sum()
    benchmark_regions = (region_weights / region_weights.sum()).to_dict()

    return {
        "ytm": float((data["YLD_YTM_MID"] * weights).sum()),
        "duration": float((data["DUR_ADJ_MID"] * weights).sum()),
        "maturity": float((data["MTY_YEARS"] * weights).sum()),
        "cost": float((data["Execution_Cost_bps"] * weights).sum()),
        "liquidity": float((data["Liquidity_Score"] * weights).sum()),
        "regions": benchmark_regions,
        "region_list": list(benchmark_regions.keys()),
    }


def compute_portfolio_metrics(
    bonds: pd.DataFrame,
    weights: np.ndarray,
    region_list: List[str],
) -> Dict:
    """
    Calcule les métriques pondérées d'un portefeuille.

    Parameters:
    bonds   : DataFrame des obligations sélectionnées (index réinitialisé).
    weights : Tableau numpy des poids (même ordre que bonds).
    region_list : Liste ordonnée des régions du benchmark.

    Returns:
    dict avec ytm, duration, maturity, cost, liquidity, regions.
    """
    ytm = bonds["YLD_YTM_MID"].values
    dur = bonds["DUR_ADJ_MID"].values
    mat = bonds["MTY_YEARS"].values
    costs = bonds["Execution_Cost_bps"].values
    liq = bonds["Liquidity_Score"].values

    port_regions = {
        region: float(np.sum(weights[(bonds["Region"] == region).values]))
        for region in region_list
    }

    return {
        "ytm": float(np.sum(weights * ytm)),
        "duration": float(np.sum(weights * dur)),
        "maturity": float(np.sum(weights * mat)),
        "cost": float(np.sum(weights * costs)),
        "liquidity": float(np.sum(weights * liq)),
        "regions": port_regions,
    }


def compute_tracking_error(
    portfolio_metrics: Dict,
    benchmark_metrics: Dict,
) -> Dict:
    """
    Retourne les écarts absolus entre portefeuille et benchmark.

    Returns:
    dict avec ytm, duration, maturity, cost_savings.
    """
    return {
        "ytm": portfolio_metrics["ytm"] - benchmark_metrics["ytm"],
        "duration": portfolio_metrics["duration"] - benchmark_metrics["duration"],
        "maturity": portfolio_metrics["maturity"] - benchmark_metrics["maturity"],
        "cost_savings": benchmark_metrics["cost"] - portfolio_metrics["cost"],
    }
