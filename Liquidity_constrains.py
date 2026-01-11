import numpy as np
import pandas as pd

"""
Outil pour générer un fichier CSV avec les paramètres de liquidité
afin de pénaliser (handicaper) les obligations moins liquides dans l'optimiseur.
"""


def generate_liquidity_data():
    """
    Génère un dataset fictif associant à chaque pays :
    1. Un score de liquidité (1-10)
    2. Un coût d'exécution estimé (bid-ask spread en bps)
    """
    # Liste complète des pays (comme dans l'univers d'investissement)
    countries = [
        'United States', 'Canada', 'Mexico', 'Germany', 'France',
        'United Kingdom', 'Italy', 'Spain', 'Netherlands', 'Switzerland',
        'Turkey', 'Poland', 'Russia', 'Japan', 'Australia', 'Singapore',
        'South Korea', 'Hong Kong', 'India', 'China', 'Indonesia',
        'Brazil', 'Chile', 'Colombia', 'Peru', 'Saudi Arabia', 'UAE',
        'Qatar', 'Kuwait', 'Israel', 'South Africa', 'Egypt', 'Nigeria'
    ]

    data = []
    np.random.seed(42)

    for country in countries:
        # Logique : pays développés = score haut, coût bas

        # GROUPE 1 : TRÈS LIQUIDE (Tier 1)
        if country in ['United States', 'Germany', 'Japan', 'Switzerland']:
            score = 10
            cost = np.random.uniform(1, 6)  # 1-6 bps

        # GROUPE 2 : LIQUIDE (Tier 2)
        elif country in [
            'Canada', 'France', 'United Kingdom', 'Italy', 'Spain',
            'Australia', 'Singapore', 'South Korea', 'Hong Kong', 'Israel'
        ]:
            score = np.random.randint(8, 10)  # 8-9
            cost = np.random.uniform(7, 11)   # 7-11 bps

        # GROUPE 3 : ÉMERGENTS LIQUIDES (Tier 3)
        elif country in [
            'Mexico', 'Brazil', 'South Africa', 'Turkey', 'Poland', 'India',
            'China', 'Indonesia', 'Chile', 'Colombia', 'Peru',
            'Saudi Arabia', 'UAE', 'Qatar', 'Kuwait'
        ]:
            score = np.random.randint(4, 8)   # 4-7
            cost = np.random.uniform(12, 25)  # 12-25 bps

        # GROUPE 4 : FRONTIER / JUNK (Tier 4)
        else:
            score = np.random.randint(1, 4)   # 1-3
            cost = np.random.uniform(30, 80)  # 30-80 bps

        data.append({
            'Country': country,
            'Liquidity_Score': score,
            'Execution_Cost_bps': round(cost, 2)
        })

    # Création et sauvegarde du DataFrame
    df = pd.DataFrame(data)
    output_file = 'country_liquidity_costs.csv'
    df.to_csv(output_file, index=False)
    print(f"'{output_file}' généré avec succès.")


if __name__ == "__main__":
    generate_liquidity_data()