# Bond Portfolio Optimizer, M2 Banque & Finance Paris 1

Projet Python S2 2025-2026 - Optimisation d'un portefeuille obligataire par
minimisation de la tracking error vis-à-vis d'un indice de référence.

## Architecture

```
bond_project/
├── main.py                        # Point d'entrée,  orchestre le pipeline
├── config.py                      # Tous les paramètres configurables
├── logger.py                      # Logger applicatif partagé
├── model/
│   └── optimizer.py               # Optimisation convexe (cvxpy)
├── repository/
│   └── bond_repository.py         # Chargement, validation, filtrage CSV
├── view/
│   └── report_view.py             # Rapports et export Excel
└── helpers/
    └── metrics.py                 # Calculs benchmark / portefeuille / tracking error
```

## Pattern appliqué

| Couche       | Rôle                                                          |
|-------------|---------------------------------------------------------------|
| `main`       | Orchestration du pipeline, aucune logique métier             |
| `repository` | Accès aux données brutes (CSV), validation, filtrage         |
| `helpers`    | Fonctions pures de calcul (métriques, tracking error)        |
| `model`      | Résolution du problème d'optimisation quadratique            |
| `view`       | Présentation des résultats (console + Excel)                 |
| `config`     | Paramètres centralisés, modifiables sans toucher au code     |
| `logger`     | Instance unique, format horodaté, réutilisable partout       |

## Installation

```bash
pip install numpy cvxpy pandas openpyxl
```

## Données requises

Deux fichiers CSV dans le répertoire racine :

**`synthetic_bond_data.csv`**, colonnes requises :
`ISIN`, `Country`, `Region`, `YLD_YTM_MID`, `DUR_ADJ_MID`, `MTY_YEARS`, `Benchmark_Weight`

**`country_liquidity_costs.csv`**, colonnes requises :
`Country`, `Execution_Cost_bps`, `Liquidity_Score`

## Utilisation

```bash
python main.py
```

Les paramètres sont modifiables directement dans le bloc `OptimizationConfig`
de `main.py`, ou en instanciant la config depuis un autre script.

## Sortie

Fichier `Optimized_Portfolio.xlsx` avec deux feuilles :
- **Résumé_Global**: comparaison Benchmark / Portfolio sur toutes les métriques
- **Composition**: détail ligne par ligne des positions, triées par poids décroissant

## Paramètres clés

| Paramètre           | Défaut | Description                                      |
|--------------------|--------|--------------------------------------------------|
| `lambda_ytm`        | 1.0    | Poids de la déviation YTM dans l'objectif        |
| `lambda_dur`        | 1.0    | Poids de la déviation duration                   |
| `lambda_region`     | 1.0    | Poids de la déviation régionale                  |
| `lambda_cost`       | 0.2    | Pénalité sur les coûts de transaction            |
| `region_tolerance`  | 0.02   | Déviation régionale tolérée (±2 %)               |
| `num_bonds`         | 350    | Nombre de positions cibles dans le portefeuille  |
| `banned_countries`  | Russia, Israel | Exclusions ESG / sanctions              |
| `min_liquidity_score` | 3   | Score de liquidité minimum                       |
