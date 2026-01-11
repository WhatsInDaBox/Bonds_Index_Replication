import random
import string
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


class BondDataGenerator:
    """
    Outil pour générer un fichier Excel contenant un univers d'investissement
    obligataire synthétique (simulant un indice Bloomberg).

    Génère des données aléatoires mais cohérentes financièrement :
    - Prix, Coupons, YTM, Duration, Maturité.
    - Répartition géographique et sectorielle.
    - Calcul des poids de l'indice (Benchmark Weights).
    """

    def __init__(self, num_bonds=1000, random_state=42):
        """
        Initialisation des paramètres de génération.

        Args:
            num_bonds (int): Nombre d'obligations à générer.
            random_state (int): Graine pour la reproductibilité.
        """
        self.num_bonds = num_bonds

        # Fixer les graines aléatoires pour la reproductibilité
        np.random.seed(random_state)
        random.seed(random_state)

        self.regions = [
            'North America', 'Europe', 'Asia',
            'Latin America', 'Middle East', 'Africa'
        ]

        self.countries = {
            'North America': ['United States', 'Canada', 'Mexico'],
            'Europe': [
                'Germany', 'France', 'United Kingdom', 'Italy', 'Spain',
                'Netherlands', 'Switzerland', 'Turkey', 'Poland', 'Russia'
            ],
            'Asia': [
                'Japan', 'Australia', 'Singapore', 'South Korea',
                'Hong Kong', 'India', 'China', 'Indonesia'
            ],
            'Latin America': ['Brazil', 'Chile', 'Colombia', 'Peru'],
            'Middle East': [
                'Saudi Arabia', 'UAE', 'Qatar', 'Kuwait', 'Israel'
            ],
            'Africa': ['South Africa', 'Egypt', 'Nigeria']
        }

        self.sectors = [
            'Government', 'Corporate', 'Financial', 'Utility',
            'Industrial', 'Energy', 'Technology', 'Communications'
        ]

        self.ratings = [
            'AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-',
            'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-'
        ]

        self.currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD']

    def _generate_isin(self, country):
        """Génère un code ISIN fictif basé sur le pays."""
        prefix = country[:2].upper()
        # Fallback pour les codes pays > 2 lettres ou non standard
        if country == 'United States':
            prefix = 'US'
        elif country == 'United Kingdom':
            prefix = 'GB'
        elif country == 'South Africa':
            prefix = 'ZA'
        elif country == 'Saudi Arabia':
            prefix = 'SA'
        elif country == 'South Korea':
            prefix = 'KR'
        elif country == 'New Zealand':
            prefix = 'NZ'

        # 10 caractères alphanumériques aléatoires
        code = ''.join(
            random.choices(string.ascii_uppercase + string.digits, k=10)
        )
        return prefix + code

    def generate_data(self):
        """
        Cœur du générateur : crée le DataFrame avec les données financières.
        Calcul des prix, rendements, durations et pondérations.
        """
        data = []
        start_date = datetime.today()

        for _ in range(self.num_bonds):
            # 1. Attribution des attributs
            region = np.random.choice(self.regions, p=[
                0.15, 0.10, 0.10, 0.10, 0.35, 0.20
            ])
            country = np.random.choice(self.countries[region])
            sector = np.random.choice(self.sectors)
            rating = np.random.choice(self.ratings)
            currency = np.random.choice(self.currencies, p=[
                0.6, 0.2, 0.05, 0.05, 0.05, 0.05
            ])

            # 2. Paramètres temporels (mat)
            # Maturité entre 2 et 30 ans
            maturity_years = np.random.uniform(2, 30)
            issue_date = start_date - timedelta(
                days=np.random.randint(0, 365 * 5)
            )
            maturity_date = issue_date + timedelta(
                days=int(maturity_years * 365)
            )

            # 3. Paramètres (C,i,P)
            # Simulation d'un spread de crédit basique selon le rating
            base_yield = 0.03  # Taux sans risque approx
            spread = (self.ratings.index(rating) + 1) * 0.005  # +50bps/cran
            ytm = base_yield + spread + np.random.normal(0, 0.005)

            # Coupon proche du YTM pour prix ~ 100
            coupon = ytm + np.random.normal(0, 0.01)
            coupon = max(0, coupon)  # Pas de coupon négatif

            # Approximation duration & prix
            duration_proxy = maturity_years * 0.75  # Doigt mouillé
            # Prix sensible au taux + bruit
            price = (
                100 +
                (coupon - ytm) * duration_proxy * 100 +
                np.random.normal(0, 1.0)
            )
            price = max(50, price)  # Floor price

            # Montant en circulation (outstanding) pour market cap
            outstanding = np.random.uniform(500, 10000)  # millions

            # 4. Construction de la ligne de donnée
            bond = {
                'ISIN': self._generate_isin(country),
                'Country': country,
                'Region': region,
                'Sector': sector,
                'Rating': rating,
                'Currency': currency,
                'Issue_Date': issue_date.strftime('%Y-%m-%d'),
                'Maturity_Date': maturity_date.strftime('%Y-%m-%d'),
                'MTY_YEARS': round(maturity_years, 4),
                'Coupon': round(coupon * 100, 4),
                'YLD_YTM_MID': round(ytm * 100, 4),     # En %
                'DUR_ADJ_MID': round(duration_proxy, 4),
                'Price': round(price, 4),
                'Outstanding_Amount_M': round(outstanding, 2),
                'Market_Value': round(price / 100 * outstanding, 2)
            }
            data.append(bond)

        # Création et sauvegarde du df
        df = pd.DataFrame(data)

        # Calcul des poids du benchmark (market cap weighted)
        total_mv = df['Market_Value'].sum()
        df['Benchmark_Weight'] = df['Market_Value'] / total_mv

        # Pré-calcul des contributions pondérées
        # Nécessaire pour l'optimiseur CVXPY
        df['w ytm'] = df['Benchmark_Weight'] * df['YLD_YTM_MID']
        df['w dur'] = df['Benchmark_Weight'] * df['DUR_ADJ_MID']
        df['w maturity'] = df['Benchmark_Weight'] * df['MTY_YEARS']

        output_file = 'synthetic_bond_data.csv'
        df.to_csv(output_file, index=False)
        print(f"'{output_file}' généré avec succès.")

if __name__ == "__main__":

    generator = BondDataGenerator(num_bonds=1000)
    generator.generate_data()