import os

from config import OptimizationConfig
from helpers.data_generator import BondDataGenerator
from helpers.metrics import compute_benchmark_metrics
from logger import get_logger
from model.optimizer import BondOptimizer
from repository.bond_repository import BondRepository
from view.report_view import ReportView

logger = get_logger()


def _ensure_data_files(config: OptimizationConfig) -> None:
    """
    Génère les fichiers CSV s'ils n'existent pas encore.
    Permet de lancer le pipeline complet sans préparer les données manuellement.
    """
    bond_missing = not os.path.exists(config.bond_file)
    liq_missing = not os.path.exists(config.liquidity_file)

    if bond_missing or liq_missing:
        logger.info(
            "Fichiers de données absents, génération de l'univers..."
        )
        generator = BondDataGenerator(num_bonds=1000, random_state=42)
        generator.generate_all(
            bond_file=config.bond_file,
            liquidity_file=config.liquidity_file,
        )
    else:
        logger.info("Fichiers de données trouvés, génération ignorée")


def main() -> None:
    logger.info("Démarrage de l'optimisation obligataire")

    # ------------------------------------------------------------------
    # 1. Configuration
    # ------------------------------------------------------------------
    config = OptimizationConfig(
        lambda_ytm=1.0,
        lambda_dur=1.0,
        lambda_maturity=1.0,
        lambda_region=1.0,
        lambda_cost=0.2,
        max_weight=0.05,
        max_country_weight=0.25,
        region_tolerance=0.00,
        banned_countries=("Russia", "Israel"),
        min_liquidity_score=3,
        num_bonds=350,
        bond_file="synthetic_bond_data.csv",
        liquidity_file="country_liquidity_costs.csv",
        output_file="Optimized_Portfolio.xlsx",
        debug=False,
    )

    _ensure_data_files(config)

    repo = BondRepository(config)
    repo.load()
    repo.apply_exclusions()

    benchmark_metrics = compute_benchmark_metrics(repo.raw)
    logger.info(
        f"Benchmark — YTM: {benchmark_metrics['ytm']:.4f}  "
        f"Dur: {benchmark_metrics['duration']:.4f}  "
        f"Mat: {benchmark_metrics['maturity']:.4f}"
    )

    optimizer = BondOptimizer(config, benchmark_metrics)
    result = optimizer.optimize(repo.pool)

    # ------------------------------------------------------------------
    # 5. View — export
    # ------------------------------------------------------------------
    view = ReportView(benchmark_metrics)
    view.print_tracking_error(result)
    view.export_excel(result, config.output_file)

    logger.info("Pipeline terminé")


if __name__ == "__main__":
    main()
