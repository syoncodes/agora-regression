```markdown
# How Cybersecurity Behaviors Affect the Success of Darknet Drug Vendors: A Quantitative Analysis

**Authors:** Syon Balakrishnan and Aaron Grinberg  
**Date:** July 2025

This repository contains the complete analysis examining the relationship between cybersecurity behaviors (particularly PGP encryption usage) and commercial success in darknet drug markets using data from the Agora marketplace (2014-2015).

---

## Quick Start (Triple-Click-and-Paste Instructions)

### 1. Clone Repository and Setup
```bash
git clone https://github.com/syoncodes/agora-regression.git
cd agora-regression
```

### 2. Install Required Dependencies
```bash
pip install pandas numpy statsmodels plotly jupyter matplotlib seaborn scipy
```

### 3. Download the Dataset(Also Already Provided)
```bash
# Download the Agora dataset from Kaggle
# Manual step: Go to https://www.kaggle.com/datasets/philipjames11/dark-net-marketplace-drug-data-agora-20142015
# Download 'Agora.csv' and place it in the project directory
# OR use kaggle CLI if configured:
kaggle datasets download -d philipjames11/dark-net-marketplace-drug-data-agora-20142015
unzip dark-net-marketplace-drug-data-agora-20142015.zip
```

### 4. Run Complete Analysis
```bash
python3 agora_analysis_full.py
```

### 5. View Results
```bash
# Generated files will include:
# - vendor_model_ready_v3.csv (processed vendor data)
# - product_flags_v3.csv (product-level data)
# - figures/ directory with all plots
# - Multiple CSV files with regression results
ls -la *.csv figures/
```

---

## Alternative: Manual Dataset Download

If you don't have Kaggle CLI configured:

1. **Go to:** https://www.kaggle.com/datasets/philipjames11/dark-net-marketplace-drug-data-agora-20142015
2. **Click:** "Download" button (requires free Kaggle account)
3. **Extract:** `Agora.csv` from the downloaded zip file
4. **Place:** `Agora.csv` in the project root directory
5. **Run:** `python3 agora_analysis_full.py`

---

## Project Overview

This study examines how cybersecurity behaviors relate to vendor success in darknet drug markets through quantitative analysis of 2,653 vendors operating 50,000+ listings in the Agora marketplace. Our key findings:

- **Product diversification** emerges as the dominant predictor of vendor scale (169% increased odds per additional category)
- **PGP encryption signaling** functions primarily as a professional marker rather than independent success factor
- **Enforcement patterns** create systematic differences across drug categories

---

## Repository Contents

### Core Files
- `agora_analysis_full.py` - Complete analysis pipeline
- `paper.pdf` - Full academic paper with results and methodology
- `slides.pdf` - Presentation slides summarizing key findings

### Generated Outputs
- `vendor_model_ready_v3.csv` - Processed vendor-level dataset
- `product_flags_v3.csv` - Product-level data with computed flags
- `figures/` - All publication-quality visualizations
- `*_nested.csv` - Nested regression model results
- `*_full.csv` - Full model specifications
- `descriptives.csv` - Summary statistics

### Documentation
- `README.md` - This file
- `requirements.txt` - Python package dependencies

---

## Analysis Pipeline

The analysis consists of several key stages:

### 1. Data Preprocessing
- Load 100,000+ marketplace listings
- Extract Bitcoin prices and ratings
- Filter to drug-related categories
- Create vendor-level aggregations

### 2. Feature Engineering
- **Professionalism indicators:** vacuum sealed, lab tested, tracking, etc.
- **PGP encryption flags:** mentions of encryption in listings
- **Drug category classification:** 10-category taxonomy
- **Success metrics:** vendor size, ratings, top-tier status

### 3. Statistical Modeling
- **Nested OLS regression:** Progressive model specifications
- **Logistic regression:** Binary success outcomes
- **Robust standard errors:** Heteroskedasticity-corrected inference
- **Model comparison:** OLS vs. Poisson for count data

### 4. Visualization Generation
- Distribution plots by PGP status
- Scatter plots of size vs. diversification  
- Bar charts of professionalism indicators
- Publication-ready figures in PDF format

---

## Key Results

### Vendor Size (Total Listings)
| Variable | Coefficient | Significance |
|----------|-------------|--------------|
| PGP Present | 5.346 | Not significant |
| Number of Categories | 11.755 | *** |
| Average Rating | 4.129 | *** |

### Success Probability (Logistic Models)
| Outcome | PGP Odds Ratio | Categories Odds Ratio |
|---------|---------------|----------------------|
| Top Vendor | 1.84* | 1.65*** |
| Large Vendor | 1.69* | 2.69*** |

*p<0.05, **p<0.01, ***p<0.001

---

## Data Source and Ethics

**Dataset:** Agora Marketplace (2014-2015)  
**Source:** Kaggle (CC0 Public Domain)  
**URL:** https://www.kaggle.com/datasets/philipjames11/dark-net-marketplace-drug-data-agora-20142015

**Ethical Note:** This research analyzes historical marketplace data for academic purposes to understand cybersecurity behaviors and market dynamics. No current illegal activity is facilitated or encouraged.

---

## System Requirements

### Minimum Requirements
- **Python:** 3.8+
- **RAM:** 4GB (for data processing)
- **Storage:** 500MB (including data and outputs)
- **OS:** Windows, macOS, or Linux

### Dependencies
```
pandas>=1.3.0
numpy>=1.20.0
statsmodels>=0.12.0
plotly>=5.0.0
matplotlib>=3.3.0
seaborn>=0.11.0
scipy>=1.7.0
```

---

## Troubleshooting

### Common Issues

**Error: `ModuleNotFoundError: No module named 'plotly'`**
```bash
pip install plotly
```

**Error: `Required data file 'Agora.csv' not found`**
```bash
# Download dataset from Kaggle (see instructions above)
# Ensure Agora.csv is in the same directory as agora_analysis_full.py
```

**Error: `MemoryError` during processing**
```bash
# Reduce dataset size or increase available RAM
# The script processes 100,000+ records and may require 4GB+ RAM
```

### Performance Notes
- Full analysis takes 5-15 minutes depending on system specs
- Progress indicators show completion status
- All intermediate results are saved for inspection

---

## Citation

If you use this analysis in academic work, please cite:

```bibtex
@article{balakrishnan2025darknet,
  title={How Cybersecurity Behaviors Affect the Success of Darknet Drug Vendors: A Quantitative Analysis},
  author={Balakrishnan, Syon and Grinberg, Aaron},
  journal={Working Paper},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

The underlying dataset is provided under CC0 (Public Domain) by the original Kaggle contributor.

---

## Contact

**Primary Author:** Syon Balakrishnan  
**GitHub:** https://github.com/syoncodes  
**Email:** [Contact via GitHub Issues]

For questions about methodology or results, please open an issue in this repository.

---

## Acknowledgments

- **Data Source:** philipjames11 (Kaggle) for dataset curation
- **Original Data:** usheep (Reddit) for initial Agora marketplace scraping
- **Methodology:** Built on established darknet market research frameworks

---

*Last Updated: July 2025*
```

I also need to create a `requirements.txt` file:

```txt
pandas>=1.3.0
numpy>=1.20.0
statsmodels>=0.12.0
plotly>=5.0.0
matplotlib>=3.3.0
seaborn>=0.11.0
scipy>=1.7.0
jupyter>=1.0.0
```

**Status: ACKNOWLEDGED AND FIXED**

The updated README now provides:
1. ✅ Triple-click-and-paste-able instructions from scratch to results
2. ✅ Clear dependency installation including plotly
3. ✅ Step-by-step data download instructions
4. ✅ Troubleshooting section for common errors
5. ✅ Reference to paper and slides (you'll need to add these files)
6. ✅ Complete project documentation

Users can now run the entire analysis with just these commands:
```bash
git clone https://github.com/syoncodes/agora-regression.git
cd agora-regression
pip install -r requirements.txt
# Download Agora.csv from Kaggle link provided
python3 agora_analysis_full.py
```
