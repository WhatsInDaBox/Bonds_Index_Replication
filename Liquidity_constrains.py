import pandas as pd
import numpy as np

"""
    Outil pour generer un fichier excel avec les parametres de liquidites pour handicaper les bonds qui sont moins
    liquides
    """

def generate_liquidity_data():
    # Comme dans le fichier excel (exemple pur)
    countries = ['United States', 'Canada', 'Mexico','Germany', 'France', 'United Kingdom', 'Italy',
                 'Spain', 'Netherlands', 'Switzerland','Turkey', 'Poland', 'Russia','Japan', 'Australia',
                 'Singapore', 'South Korea', 'Hong Kong','India', 'China', 'Indonesia','Brazil', 'Chile',
                 'Colombia', 'Peru','Saudi Arabia', 'UAE', 'Qatar', 'Kuwait', 'Israel','South Africa', 'Egypt',
                 'Nigeria']

    data = []
    np.random.seed(42)

    for country in countries:
        # logique: pays développés = score haut, coût bas
        if country in ['United States', 'Germany', 'Japan', 'Switzerland']:
            score = 10
            cost = np.random.uniform(1, 6)  # 1-6 bps
        elif country in ['Canada','France', 'United Kingdom', 'Italy', 'Spain','Australia', 'Singapore',
                         'South Korea', 'Hong Kong','Israel']:
            score = np.random.randint(8, 10)  # 8-9
            cost = np.random.uniform(7, 11)  # 7-11 bps
        elif country in ['Mexico', 'Brazil', 'South Africa','Turkey', 'Poland','India',
                         'China', 'Indonesia', 'Chile', 'Colombia', 'Peru','Saudi Arabia',
                         'UAE', 'Qatar', 'Kuwait']:
            score = np.random.randint(4, 8)  # 4-7
            cost = np.random.uniform(12, 25)  # 10-25 bps
        else:  # "junk" ou très illiquide
            score = np.random.randint(1, 4)  # 1-3
            cost = np.random.uniform(30, 80)  # 30-80 bps

        data.append({'Country': country, 'Liquidity_Score': score, 'Execution_Cost_bps': round(cost, 2)})

    df = pd.DataFrame(data)
    df.to_csv('country_liquidity_costs.csv', index=False)
    print("Fichier 'country_liquidity_costs.csv' généré.")


if __name__ == "__main__":
    generate_liquidity_data()