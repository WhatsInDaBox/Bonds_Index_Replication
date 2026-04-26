"""
repository/bond_repository.py
------------------------------
Responsabilité unique : accès aux données brutes.

- Chargement des CSV obligataires et de liquidité
- Fusion et nettoyage
- Validation (colonnes, doublons, plages de valeurs)
- Filtrage de l'univers investissable (exclusions ESG / liquidité)

Aucune logique métier d'optimisation ici.
"""

import pandas as pd

from config import OptimizationConfig
from logger import get_logger

logger = get_logger()

REQUIRED_COLUMNS = [
    "ISIN",
    "Country",
    "YLD_YTM_MID",
    "DUR_ADJ_MID",
    "MTY_YEARS",
    "Region",
    "Benchmark_Weight",
]


class BondRepository:
    """Charge, valide et filtre l'univers obligataire."""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self._raw: pd.DataFrame = pd.DataFrame()
        self._pool: pd.DataFrame = pd.DataFrame()

    # Chargement

    def load(self) -> None:
        """
        Charge les données depuis les fichiers CSV définis dans la config
        et stocke l'univers complet dans `self._raw`.
        """
        logger.info(
            f"Chargement des données : '{self.config.bond_file}'"
            f" + '{self.config.liquidity_file}'"
        )
        bonds = pd.read_csv(self.config.bond_file)
        liquidity = pd.read_csv(self.config.liquidity_file)

        df = pd.merge(bonds, liquidity, on="Country", how="left")
        df = df.fillna({"Execution_Cost_bps": 50, "Liquidity_Score": 0})

        self._validate(df)
        self._raw = df
        logger.info(f"Univers brut chargé : {len(df)} obligations")

    def _validate(self, df: pd.DataFrame) -> None:
        """Vérifie colonnes, doublons et plages de valeurs."""
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes : {missing}")

        if df["ISIN"].duplicated().any():
            raise ValueError("Doublons d'ISIN détectés dans les données")

        assert (df["DUR_ADJ_MID"] > 0).all(), "Durations invalides détectées"
        assert (df["MTY_YEARS"] > 0).all(), "Maturités invalides détectées"
        assert (df["Liquidity_Score"] >= 0).all(), "Scores de liquidité invalides"

        logger.debug("Validation des données : OK")

    # Filtrage

    def apply_exclusions(self) -> None:
        """
        Construit le pool investissable en excluant :
        - les pays bannis (ESG / sanctions)
        - les obligations sous le seuil de liquidité minimum
        """
        cfg = self.config
        n_before = len(self._raw)

        pool = self._raw[
            ~self._raw["Country"].isin(cfg.banned_countries)
            & (self._raw["Liquidity_Score"] >= cfg.min_liquidity_score)
        ].copy()

        n_after = len(pool)
        logger.info(
            f"Exclusions appliquées : {n_before - n_after} obligations filtrées"
            f" ({n_after} restantes dans le pool investissable)"
        )
        self._pool = pool

    # Accesseurs

    @property
    def raw(self) -> pd.DataFrame:
        """Univers complet (avant exclusions)."""
        if self._raw.empty:
            raise RuntimeError("Appelez load() avant d'accéder à raw.")
        return self._raw

    @property
    def pool(self) -> pd.DataFrame:
        """Univers filtré (après exclusions)."""
        if self._pool.empty:
            raise RuntimeError("Appelez apply_exclusions() avant d'accéder à pool.")
        return self._pool
