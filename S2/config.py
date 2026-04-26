"""
config.py
---------
Point unique de configuration pour le projet d'optimisation obligataire.
Toutes les valeurs modifiables sont ici.
"""

from dataclasses import dataclass, field


@dataclass
class OptimizationConfig:
    """
    Paramètres de l'optimisation de portefeuille.

    Pénalités (lambda_*) : pondèrent chaque terme de la tracking error.
    Plus la valeur est élevée, plus la contrainte correspondante est prioritaire.
    """

    lambda_ytm: float = 1.0
    lambda_dur: float = 1.0
    lambda_maturity: float = 1.0
    lambda_region: float = 1.0
    lambda_cost: float = 0.2      # Pénalité coûts de transaction
    min_weight: float = 0.001     # Position minimale : 0.1 %
    max_weight: float = 0.05      # Position maximale : 5 %
    max_country_weight: float = 0.25

    # Tolérance sur la déviation régionale
    region_tolerance: float = 0.02

    banned_countries: tuple = ("Russia", "Israel")
    min_liquidity_score: int = 3

    num_bonds: int = 350

    # Fichiers de données (chemins relatifs depuis la racine du projet)
    bond_file: str = "synthetic_bond_data.csv"
    liquidity_file: str = "country_liquidity_costs.csv"

    output_file: str = "Optimized_Portfolio.xlsx"

    debug: bool = False
