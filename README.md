# Agora-Darknet-Vendor-Regression

Regression analysis on Agora darknet marketplace vendor data to assess whether PGP encryption usage correlates with improved vendor performance. Data sourced from the Kaggle Agora dataset (2014–2015).

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Data Source](#data-source)  
- [Data Schema](#data-schema)  
- [Dependencies](#dependencies)  
- [Usage](#usage)  
- [Analysis Results](#analysis-results)  
- [File Structure](#file-structure)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Project Overview

This project applies nested ordinary least squares (OLS) and logistic regression models to vendor-level data from the Agora darknet marketplace. Key predictors include encryption presence, pricing deviations, category diversification, and average rating. The goal is to determine whether PGP encryption adoption predicts:

- Number of listings per vendor (log-transformed)  
- Odds of top-tier vendor status  
- Odds of high customer ratings  

---

## Data Source

The dataset originates from Kaggle’s “Dark Net Marketplace Data (Agora 2014-2015)” by user philipjames11. It comprises over 100,000 listings parsed from Agora’s HTML archives.

Dataset page:  
https://www.kaggle.com/datasets/philipjames11/dark-net-marketplace-drug-data-agora-20142015

Dataset license: CC0 (Public Domain)

---

## Data Schema

| Column            | Type    | Description                                                         |
|-------------------|---------|---------------------------------------------------------------------|
| `vendor_id`       | integer | Unique identifier for each vendor                                   |
| `total_listings`  | integer | Total number of listings per vendor                                 |
| `num_categories`  | integer | Count of unique categories each vendor offers                       |
| `avg_price_btc`   | float   | Mean listing price in BTC                                           |
| `avg_rating`      | float   | Mean customer rating (scale 0–5)                                    |
| `price_diff_norm` | float   | Z-score of within-category price deviations                         |
| `pgp_present`     | binary  | Flag for PGP encryption mention or requirement in listing details   |

---

## Dependencies

| Package      | Version Requirement |
|--------------|---------------------|
| Python       | 3.8+                |
| pandas       | 1.2+                |
| numpy        | 1.19+               |
| statsmodels  | 0.12+               |
| jupyter      | 1.0+                |

Install dependencies with:

```bash
pip install pandas numpy statsmodels jupyter
