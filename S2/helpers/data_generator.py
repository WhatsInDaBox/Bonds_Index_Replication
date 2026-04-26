"""
helpers/data_generator.py
--------------------------
Intègre les deux fichiers originaux de génération de données :
  - Investment_Universe_Generator.py  -> génération de l'univers obligataire
  - Liquidity_constrains.py           -> scores et coûts d'exécution par pays

Corrections appliquées vs. versions originales :
  - Probabilités régionales corrigées (dominance NA/Europe, ~Bloomberg Global Agg)
  - ISIN garanti unique (rejet/retirage sur collision)
  - Colonnes `w ytm / w dur / w maturity` supprimées (redondantes avec metrics.py)
  - Les deux générateurs réunis en une seule classe pour simplifier main.py
  - Logger remplace les print()

Reproductibilité : random_state=42 garanti identique à l'original.
"""

import random
import string
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from logger import get_logger

logger = get_logger()

# Probabilités régionales corrigées par rapport au S1 (approximation Bloomberg Global Agg)
# Ordre    : North America, Europe, Asia, Latin America, Middle East, Africa
REGION_PROBS = [0.40, 0.35, 0.12, 0.05, 0.05, 0.03]


class BondDataGenerator:
    """
    Génère un univers obligataire synthétique (simulation Bloomberg Global Agg)
    et le fichier de liquidité associé.

    Produit deux fichiers CSV :
      - synthetic_bond_data.csv
      - country_liquidity_costs.csv
    """

    def __init__(self, num_bonds: int = 1000, random_state: int = 42):
        self.num_bonds = num_bonds

        # Graines identiques à l'original pour reproductibilité
        np.random.seed(random_state)
        random.seed(random_state)

        self.regions = [
            "North America", "Europe", "Asia",
            "Latin America", "Middle East", "Africa",
        ]

        self.countries = {
            "North America": ["United States", "Canada", "Mexico"],
            "Europe": [
                "Germany", "France", "United Kingdom", "Italy", "Spain",
                "Netherlands", "Switzerland", "Turkey", "Poland", "Russia",
            ],
            "Asia": [
                "Japan", "Australia", "Singapore", "South Korea",
                "Hong Kong", "India", "China", "Indonesia",
            ],
            "Latin America": ["Brazil", "Chile", "Colombia", "Peru"],
            "Middle East": ["Saudi Arabia", "UAE", "Qatar", "Kuwait", "Israel"],
            "Africa": ["South Africa", "Egypt", "Nigeria"],
        }

        self.sectors = [
            "Government", "Corporate", "Financial", "Utility",
            "Industrial", "Energy", "Technology", "Communications",
        ]

        self.ratings = [
            "AAA", "AA+", "AA", "AA-", "A+", "A", "A-",
            "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-",
        ]

        self.currencies = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD"]

        self._used_isins: set = set()


    def _generate_isin(self, country: str) -> str:
        """
        Génère un code ISIN fictif unique.
        """
        prefix_map = {
            "United States": "US",
            "United Kingdom": "GB",
            "South Africa": "ZA",
            "Saudi Arabia": "SA",
            "South Korea": "KR",
        }
        prefix = prefix_map.get(country, country[:2].upper())

        for _ in range(100):
            code = "".join(
                random.choices(string.ascii_uppercase + string.digits, k=10)
            )
            isin = prefix + code
            if isin not in self._used_isins:
                self._used_isins.add(isin)
                return isin

        raise RuntimeError(f"Impossible de générer un ISIN unique pour {country}")

    def generate_data(self) -> pd.DataFrame:
        """
        Génère le DataFrame obligataire principal.
        Structure identique à l'original sauf :
          - Probabilités régionales corrigées
          - ISIN unique garanti
          - Colonnes w ytm / w dur / w maturity supprimées
        """
        data = []
        start_date = datetime.today()

        for _ in range(self.num_bonds):
            # Attributs géographiques et qualitatifs
            region = np.random.choice(self.regions, p=REGION_PROBS)
            country = np.random.choice(self.countries[region])
            sector = np.random.choice(self.sectors)
            rating = np.random.choice(self.ratings)
            currency = np.random.choice(
                self.currencies, p=[0.6, 0.2, 0.05, 0.05, 0.05, 0.05]
            )

            # Paramètres temporels — maturité entre 2 et 30 ans
            maturity_years = np.random.uniform(2, 30)
            issue_date = start_date - timedelta(
                days=np.random.randint(0, 365 * 5)
            )
            maturity_date = issue_date + timedelta(
                days=int(maturity_years * 365)
            )

            # Paramètres financiers (C, i, P)
            base_yield = 0.03                                    # Taux sans risque approx
            spread = (self.ratings.index(rating) + 1) * 0.005   # +50 bps/cran
            ytm = base_yield + spread + np.random.normal(0, 0.005)

            coupon = max(0.0, ytm + np.random.normal(0, 0.01))  # Pas de coupon négatif

            duration_proxy = maturity_years * 0.75              # Approximation duration
            price = max(
                50.0,
                100 + (coupon - ytm) * duration_proxy * 100 + np.random.normal(0, 1.0),
            )

            outstanding = np.random.uniform(500, 10_000)        # Millions USD

            # Construction de la ligne
            data.append({
                "ISIN": self._generate_isin(country),
                "Country": country,
                "Region": region,
                "Sector": sector,
                "Rating": rating,
                "Currency": currency,
                "Issue_Date": issue_date.strftime("%Y-%m-%d"),
                "Maturity_Date": maturity_date.strftime("%Y-%m-%d"),
                "MTY_YEARS": round(maturity_years, 4),
                "Coupon": round(coupon * 100, 4),
                "YLD_YTM_MID": round(ytm * 100, 4),
                "DUR_ADJ_MID": round(duration_proxy, 4),
                "Price": round(price, 4),
                "Outstanding_Amount_M": round(outstanding, 2),
                "Market_Value": round(price / 100 * outstanding, 2),
            })

        df = pd.DataFrame(data)

        # Poids benchmark (market-cap weighted)
        df["Benchmark_Weight"] = df["Market_Value"] / df["Market_Value"].sum()

        return df

    # Génération de la liquidité (Liquidity_constrains.py)

    def generate_liquidity_data(self, bond_df: pd.DataFrame) -> pd.DataFrame:
        """
        Génère les scores de liquidité et coûts d'exécution par pays.
        Logique de tiering identique à Liquidity_constrains.py :
          - Tier 1 : très liquide  (US, DE, JP, CH) --> score 10,  1-6 bps
          - Tier 2 : liquide       (G10 + marchés dév.) --> score 8-9, 7-11 bps
          - Tier 3 : émergents liquides --> score 4-7, 12-25 bps
          - Tier 4 : frontier/junk (Russia, Egypt, ...) --> score 1-3, 30-80 bps
        Restreint aux pays présents dans l'univers obligataire généré.
        """
        countries = bond_df["Country"].unique().tolist()

        tier1 = {"United States", "Germany", "Japan", "Switzerland"}

        tier2 = {
            "Canada", "France", "United Kingdom", "Italy", "Spain",
            "Australia", "Singapore", "South Korea", "Hong Kong", "Israel",
        }

        tier3 = {
            "Mexico", "Brazil", "South Africa", "Turkey", "Poland", "India",
            "China", "Indonesia", "Chile", "Colombia", "Peru",
            "Saudi Arabia", "UAE", "Qatar", "Kuwait",
        }

        # Tier 4 par défaut : Russia, Egypt, Nigeria et tout pays non listé

        data = []
        for country in countries:
            if country in tier1:
                score = 10
                cost = round(np.random.uniform(1, 6), 2)

            elif country in tier2:
                score = int(np.random.randint(8, 10))
                cost = round(np.random.uniform(7, 11), 2)

            elif country in tier3:
                score = int(np.random.randint(4, 8))
                cost = round(np.random.uniform(12, 25), 2)

            else:
                score = int(np.random.randint(1, 4))
                cost = round(np.random.uniform(30, 80), 2)

            data.append({
                "Country": country,
                "Liquidity_Score": score,
                "Execution_Cost_bps": cost,
            })

        return (
            pd.DataFrame(data)
            .sort_values("Country")
            .reset_index(drop=True)
        )

    # Export des deux fichiers CSV

    def generate_all(
        self,
        bond_file: str = "synthetic_bond_data.csv",
        liquidity_file: str = "country_liquidity_costs.csv",
    ) -> None:
        """Génère et sauvegarde les deux fichiers CSV requis par l'optimiseur."""
        logger.info(f"Génération de {self.num_bonds} obligations...")
        df_bonds = self.generate_data()
        df_bonds.to_csv(bond_file, index=False)
        logger.info(f"'{bond_file}' généré ({len(df_bonds)} lignes)")

        df_liq = self.generate_liquidity_data(df_bonds)
        df_liq.to_csv(liquidity_file, index=False)
        logger.info(f"'{liquidity_file}' généré ({len(df_liq)} pays)")


if __name__ == "__main__":
    generator = BondDataGenerator(num_bonds=1000, random_state=42)
    generator.generate_all()
