<a id="readme-top"></a>

  <h1 align="center">Bond Index replication tool</h3>

  <p align="center">
    Fixed-income portfolio construction and index tracking using convex optimization.
    <br />
    <br />
    <br />
    <a href="https://github.com/WhatsInDaBox/Bonds_Index_Replication/issues/new/choose">Report Bug</a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about">About</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About

This project implements a tool for passive asset management, specifically designed to replicate the performance and risk characteristics of a broad bond index using a smaller subset of securities.

Unlike simple sampling, this engine utilizes **Convex Optimization (CVXPY)** to mathematically minimize the tracking error against a benchmark while strictly adhering to investment constraints. It allows for dynamic exclusion of specific regions or issuers (e.g., ESG filtering or geopolitical sanctions) while automatically rebalancing the remaining portfolio to match the benchmark's key metrics.

**Key Technical Features:**
* **Convex optimization solver:** Minimizes the squared distance between the portfolio and benchmark across critical risk vectors (duration, YTM, maturity).
* **Constraint management:** Enforces long-only positions (no short selling) and full capital allocation ($\sum w_i = 1$).
* **Risk factors matching:** Targets precise exposure to modified Duration and yield-to-maturity to ensure interest rate sensitivity aligns with the index.
* **Geopolitical/ESG filtering:** Exclude specific countries or regions (e.g., 'Asia', 'Russia') and re-optimize the remaining universe to maintain index correlation.
* **Reporting:** Generates Excel-based reports with new allocation and geographic distribution.

### Built With

* [![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)
* [![CVXPY](https://img.shields.io/badge/CVXPY-Optimization-red)](#)
* [![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=fff)](#)
* [![NumPy](https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff)](#)
* [![Matplotlib](https://custom-icon-badges.demolab.com/badge/Matplotlib-71D291?logo=matplotlib&logoColor=fff)](#)

## Getting Started

To run the replication engine locally, follow these steps.

### Prerequisites

Ensure you have Python 3.9+ installed. Install the required dependencies:

```sh
pip install pandas numpy cvxpy matplotlib openpyxl tqdm
```
**Installation**
1. Clone the repo:
```sh
git clone https://github.com/WhatsInDaBox/Bonds_Index_Replication
```
2. Place your source data (e.g., synthetic_bond_data.xlsx) in the root directory.
3. Verify data column headers match the _clean_data method requirements (ISIN, Country, YLD_YTM_MID, DUR_ADJ_MID, MTY_YEARS, Region)

### Usage
Run the main optimization script to generate the portfolio:
```sh
python optimizer_CVXPY.py
```

### Workflow
1. **Ingestion:** The script loads the bond universe from the Excel source.
2. **Benchmark calculation:** Calculates the market-cap-weighted averages of the full universe.
3. **Optimization:** Runs the CVXPY solver to find optimal weights $w$ that minimize:
   $$\text{minimize} \quad || X_{port} - X_{bench} ||^2$$
   subject to:
   $$\sum w_i = 1, \quad w_i \ge 0$$
4. **Reporting:** Outputs the optimized selection to Excel file.

## Roadmap

- [ ] Implement currency conversion matrix for multi-currency indices.
- [ ] Add transaction cost constraints to the solver.
- [ ] Integrate live market data feed (Bloomberg/Reuters API).
- [ ] Add turnover constraints for monthly rebalancing.

## Contact

Nathan Thuille - [nathanthuille](https://www.linkedin.com/in/nathanthuille/) - thuille.nathan@icloud.com <br />

## Acknowledgments

* [Bloomberg methodology for bonds index](https://assets.bbhub.io/professional/sites/10/Bloomberg-Index-Publications-Fixed-Income-Index-Methodology.pdf)
* [_Convex Optimization_, Stephen Boyd & Lieven Vandenberghe (2004)](https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)
* [CVXPY documentation](https://www.cvxpy.org/tutorial/index.html)
* [OSQP solver documentation](https://osqp.org/docs/parsers/cvxpy.html)
