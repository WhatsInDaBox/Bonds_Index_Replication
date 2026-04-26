"""
view/report_view.py
-------------------
Responsabilité unique : présentation et export des résultats.

- Génération du rapport détaillé ligne par ligne (composition du portefeuille)
- Génération de la feuille de synthèse (comparaison Benchmark / Portfolio)
- Export Excel multi-feuilles

Aucune logique d'optimisation ici. Reçoit uniquement les résultats calculés
par le modèle et les métriques benchmark du helper.
"""

from typing import Dict

import pandas as pd

from logger import get_logger

logger = get_logger()


class ReportView:
    """Génère et exporte les rapports du portefeuille optimisé."""

    def __init__(self, benchmark_metrics: Dict):
        self.bm = benchmark_metrics

    # ------------------------------------------------------------------
    # Rapport détaillé
    # ------------------------------------------------------------------

    def build_composition(self, result: Dict) -> pd.DataFrame:
        """
        Construit le DataFrame de composition du portefeuille,
        trié par poids décroissant.

        Colonnes : ISIN, Country, Region, Weight (%), YTM (%),
                   Duration, Maturity (Years), Execution Cost (bps),
                   Liquidity Score.
        """
        bonds = result["bonds"]
        weights = result["weights"]

        df = pd.DataFrame({
            "ISIN": bonds["ISIN"].values,
            "Country": bonds["Country"].values,
            "Region": bonds["Region"].values,
            "Weight (%)": weights * 100,
            "YTM (%)": bonds["YLD_YTM_MID"].values,
            "Duration": bonds["DUR_ADJ_MID"].values,
            "Maturity (Years)": bonds["MTY_YEARS"].values,
            "Execution Cost (bps)": bonds["Execution_Cost_bps"].values,
            "Liquidity Score": bonds["Liquidity_Score"].values,
        })

        return df.sort_values("Weight (%)", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Feuille de synthèse
    # ------------------------------------------------------------------

    def build_summary(self, result: Dict) -> pd.DataFrame:
        """
        Construit le DataFrame de synthèse :
        comparaison Benchmark / Portfolio pour chaque métrique,
        avec l'écart absolu.
        """
        bm = self.bm

        rows = [
            {
                "Metric": "Yield (YTM %)",
                "Benchmark": bm["ytm"],
                "Portfolio": result["portfolio_ytm"],
            },
            {
                "Metric": "Duration (years)",
                "Benchmark": bm["duration"],
                "Portfolio": result["portfolio_dur"],
            },
            {
                "Metric": "Maturity (years)",
                "Benchmark": bm["maturity"],
                "Portfolio": result["portfolio_maturity"],
            },
            {
                "Metric": "Execution cost (bps)",
                "Benchmark": bm["cost"],
                "Portfolio": result["portfolio_cost"],
            },
            {
                "Metric": "Liquidity score (avg)",
                "Benchmark": bm["liquidity"],
                "Portfolio": result["portfolio_liquidity"],
            },
        ]

        for region in bm["region_list"]:
            rows.append({
                "Metric": f"Region: {region}",
                "Benchmark": bm["regions"].get(region, 0.0),
                "Portfolio": result["portfolio_regions"].get(region, 0.0),
            })

        df = pd.DataFrame(rows)
        df["Difference"] = df["Portfolio"] - df["Benchmark"]
        return df[["Metric", "Benchmark", "Portfolio", "Difference"]]

    # ------------------------------------------------------------------
    # Export Excel
    # ------------------------------------------------------------------

    def export_excel(
        self,
        result: Dict,
        output_file: str,
    ) -> None:
        """
        Exporte la synthèse et la composition dans un fichier Excel
        deux feuilles : 'Résumé_Global' et 'Composition'.

        Parameters
        ----------
        result      : dict retourné par BondOptimizer.optimize()
        output_file : chemin du fichier .xlsx de sortie
        """
        df_summary = self.build_summary(result)
        df_composition = self.build_composition(result)

        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            df_summary.to_excel(
                writer, sheet_name="Résumé_Global", index=False
            )
            df_composition.to_excel(
                writer, sheet_name="Composition", index=False
            )

        logger.info(f"Résultat exporté : '{output_file}'")

    # ------------------------------------------------------------------
    # Affichage console
    # ------------------------------------------------------------------

    def print_tracking_error(self, result: Dict) -> None:
        """Affiche un résumé de la tracking error dans la console."""
        te = result["tracking_error"]
        print("\n--- Tracking Error ---")
        print(f"  YTM       : {te['ytm']:+.6f}")
        print(f"  Duration  : {te['duration']:+.6f}")
        print(f"  Maturity  : {te['maturity']:+.6f}")
        print(f"  Cost save : {te['cost_savings']:+.4f} bps")
        print(f"  Positions : {result['num_bonds']}")
        print("----------------------\n")
