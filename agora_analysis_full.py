"""
Vendor Professionalism and Success Analysis for Dark Web Marketplace Data

This script analyzes vendor characteristics and their relationship to marketplace success
metrics using data from the Agora marketplace. The analysis includes:
- Professionalism indicators (PGP usage, professional language)
- Vendor size and diversification metrics
- Drug category classification
- Statistical modeling (OLS and logistic regression)

Author: Syon Balakrishnan
Date: 07/25/2025
"""

# ================================ IMPORTS ================================
# Standard library imports for file operations, regular expressions, and utilities
import re
import warnings
import os
import sys
import itertools

# Suppress non-critical warnings for cleaner output
warnings.filterwarnings("ignore")

# Scientific computing libraries
import numpy as np                    # Numerical operations and array manipulation
import pandas as pd                   # Data manipulation and analysis
from numpy.linalg import LinAlgError  # Linear algebra error handling

# Statistical modeling libraries
import statsmodels.formula.api as smf # Formula-based model specification
import statsmodels.api as sm          # Core statsmodels functionality

# Plotting library for figure generation
import plotly.express as px           # High-level plotting interface


# ================================ CONFIGURATION ================================
# File paths for input/output operations
RAW_FILE = "Agora.csv"                        # Source marketplace data (listing-level)
OUT_FILE = "vendor_model_ready_v3.csv"        # Processed vendor-level dataset
PRODUCT_OUT_FILE = "product_flags_v3.csv"     # Product-level dataset with computed flags


# ================================ PROFESSIONALISM INDICATORS ================================
"""
Professional language indicators used to assess vendor professionalism.
These terms were selected based on their association with professional
vendor practices in online marketplaces. The list excludes overly general
terms per research methodology requirements.
"""

# Core professionalism terms identified in initial analysis
BASE = [
    "vacuum sealed", "vacuum-sealed", "lab tested", "lab-tested",
    "tracking", "refund", "guarantee", "sealed"
]

# Additional professionalism indicators from extended analysis
ADD = [
    "tracking number", "purity", "reship", "business days",
    "pharmaceutical name", "escrow", "public key",
    "read profile", "erowid", "blister pack", "discreet shipping"
]

# Combine and standardize all professionalism terms
PRO_TERMS = sorted(set(t.lower() for t in BASE + ADD))

# Compile regular expression pattern for efficient text matching
# Uses case-insensitive matching and escapes special regex characters
PRO_RE = re.compile("|".join(re.escape(t) for t in PRO_TERMS), re.I)


# ================================ DRUG CATEGORY TAXONOMY ================================
"""
Hierarchical drug category classification system.
Each category contains representative terms used for classification.
Categories are derived from marketplace taxonomy and DEA scheduling.
"""

DRUG_CATEGORY_TERMS = {
    "Cannabis": ["cannabis", "marijuana", "weed", "thc", "cbd", "hash", "hashish", "ganja", "hemp"],
    "Dissociatives": ["ketamine", "pcp", "dxm", "dextromethorphan", "mxe", "3-meo-pcp", "diphenidine"],
    "Ecstasy": ["mdma", "ecstasy", "molly", "mda", "6-apb", "methylone"],
    "Opioids": ["heroin", "oxycodone", "fentanyl", "morphine", "codeine", "tramadol", "opium", "hydrocodone"],
    "Prescription": ["adderall", "xanax", "valium", "ritalin", "klonopin", "ativan", "ambien", "viagra"],
    "Psychedelics": ["lsd", "psilocybin", "mushrooms", "dmt", "mescaline", "2c-b", "2c-i", "25i-nbome"],
    "RCs": ["research chemical", "rc", "nbome", "25i", "4-aco", "4-ho", "2c-", "novel psychoactive"],
    "Steroids": ["testosterone", "anavar", "winstrol", "deca", "trenbolone", "hgh", "growth hormone"],
    "Stimulants": ["cocaine", "amphetamine", "methamphetamine", "speed", "crystal", "crack"],
    "OtherMinor": ["tobacco", "nicotine", "kratom", "salvia", "spice", "synthetic marijuana"]
}


# ================================ DATA LOADING AND VALIDATION ================================
"""
Load and validate the raw marketplace data.
Ensures required files exist and have expected structure.
"""

# Verify raw data file existence
if not os.path.exists(RAW_FILE):
    sys.exit(f"ERROR: Required data file '{RAW_FILE}' not found in current directory.")

# Load marketplace data with Latin-1 encoding for special characters
# Strip whitespace from column names to prevent parsing issues
df = pd.read_csv(RAW_FILE, encoding="latin1").rename(columns=str.strip)

# Assign unique identifier to each listing for tracking
df["ListingID"] = np.arange(len(df)) + 1


# ================================ DATA PREPROCESSING ================================
"""
Extract and standardize key variables from raw text fields.
This includes price normalization (BTC) and rating extraction.
"""

# Regular expression patterns for data extraction
btc_re = re.compile(r"([\d.]+)\s*BTC", re.I)    # Matches Bitcoin prices (e.g., "0.012 BTC")
rate_re = re.compile(r"([\d.]+)/5")             # Matches ratings (e.g., "4.9/5")

# Extract numeric values from text fields
df["price_btc"] = df["Price"].astype(str).str.extract(btc_re)[0].astype(float, errors="ignore")
df["rating_numeric"] = df["Rating"].astype(str).str.extract(rate_re)[0].astype(float, errors="ignore")

# Remove incomplete records (require both price and rating)
df = df.dropna(subset=["price_btc", "rating_numeric"])

# Filter to drug-related categories only
df = df[df["Category"].str.contains("Drugs|Tobacco", case=False, na=False)]

# Identify description column dynamically (handles varying column names)
DESC_COL = next((c for c in df.columns if "descr" in c.lower()), None)
if DESC_COL is None:
    sys.exit("ERROR: No description column found. Expected column name containing 'descr'.")


# ================================ CATEGORY PROCESSING ================================
"""
Extract hierarchical category information from marketplace taxonomy.
Creates secondary category tokens for granular classification.
"""

# Extract secondary category from hierarchical path
# Example: "Drugs/Cannabis/Weed" → secondary = "cannabis"
df["sec_token"] = (
    df["Category"]
    .astype(str)
    .str.split("/", expand=True)
    .iloc[:, 1]  # Select second level of hierarchy
    .str.strip()
    .str.lower()
)


# ================================ FEATURE ENGINEERING ================================
"""
Create binary indicators for professionalism and encryption usage.
These features serve as key predictors in subsequent models.
"""

# Flag listings containing professional language
df["professional_flag"] = df[DESC_COL].fillna("").str.lower().str.contains(PRO_RE)

# Flag listings mentioning PGP (Pretty Good Privacy) encryption
df["pgp_present_flag"] = df[DESC_COL].str.contains("pgp", case=False, na=False)


# ================================ VENDOR AGGREGATION ================================
"""
Aggregate listing-level data to vendor level.
Computes summary statistics and derived metrics for each vendor.
"""

vendor = (
    df.groupby("Vendor")
    .agg(
        total_listings=("ListingID", "count"),           # Total number of listings
        num_categories=("Category", "nunique"),          # Category diversification
        avg_price_btc=("price_btc", "mean"),            # Average listing price
        avg_rating=("rating_numeric", "mean"),           # Average customer rating
        pct_professional=("professional_flag", "mean"),  # Proportion of professional listings
        pgp_present=("pgp_present_flag", "max")         # Any listing mentions PGP
    )
    .reset_index()
)


# ================================ DRUG CATEGORY INDICATORS ================================
"""
Create binary indicators for drug category presence at vendor level.
Distinguishes between primary and secondary category classifications.
"""

# Define categories for analysis
DRUGS = [
    "Cannabis", "Dissociatives", "Ecstasy", "Opioids", "OtherMinor",
    "Prescription", "Psychedelics", "RCs", "Steroids", "Stimulants"
]

for drug in DRUGS:
    # Primary classification: vendor has listing in this category
    primary = (
        df.groupby("Vendor")["Category"]
        .apply(lambda s, t=drug: s.str.contains(t, case=False, na=False).any())
        .astype(bool)
    )
    vendor[f"class_{drug}"] = vendor["Vendor"].map(primary).fillna(False)
    
    # Secondary classification: vendor has listing with this secondary token
    secondary = (
        df.groupby("Vendor")["sec_token"]
        .apply(lambda s, t=drug.lower(): s.str.contains(t, na=False).any())
        .astype(bool)
    )
    vendor[f"sec_{drug}"] = vendor["Vendor"].map(secondary).fillna(False)


# ================================ DEPENDENT VARIABLES ================================
"""
Create binary outcome variables for logistic regression models.
These represent different definitions of vendor success.
"""

# Top vendor: 90th percentile or above in total listings
vendor["top_vendor"] = (vendor["total_listings"] >= vendor["total_listings"].quantile(0.90)).astype(int)

# Large vendor: above median in total listings
vendor["large_vendor"] = (vendor["total_listings"] >= vendor["total_listings"].median()).astype(int)


# ================================ MODEL SPECIFICATIONS ================================
"""
Define nested model specifications for hypothesis testing.
Each model adds additional predictors to test incremental explanatory power.
"""

# Base models with increasing complexity
b1 = "pgp_present"                                    # Model 1: PGP only
b2 = b1 + " + num_categories"                        # Model 2: + diversification
b3 = b2 + " + avg_price_btc"                         # Model 3: + pricing
b4 = b3 + " + pct_professional"                      # Model 4: + professionalism

# Model 5: Full model with drug category controls
CLASS_RHS = " + ".join(c for c in vendor.columns if c.startswith(("class_", "sec_")))
b5 = b4 + " + " + CLASS_RHS

# Construct model formulas for each dependent variable
MODEL_RHSS = [b1, b2, b3, b4, b5]
OLS_SIZE = [f"total_listings ~ {rhs}" for rhs in MODEL_RHSS]    # OLS: vendor size
OLS_RATE = [f"avg_rating ~ {rhs}" for rhs in MODEL_RHSS]        # OLS: average rating
LOG_TOP = [f"top_vendor ~ {rhs}" for rhs in MODEL_RHSS]         # Logit: top vendor
LOG_LARGE = [f"large_vendor ~ {rhs}" for rhs in MODEL_RHSS]     # Logit: large vendor


# ================================ UTILITY FUNCTIONS ================================

def drop_constant_terms(rhs: str, data: pd.DataFrame) -> str:
    """
    Remove predictors with zero variance from model specification.
    
    Constant predictors provide no information and can cause numerical
    instability in model estimation.
    
    Parameters
    ----------
    rhs : str
        Right-hand side of model formula containing predictor terms
    data : pd.DataFrame
        Dataset containing the predictor variables
        
    Returns
    -------
    str
        Cleaned formula with constant terms removed
    """
    terms = [t.strip() for t in rhs.split('+')]
    keep = [t for t in terms if t not in data.columns or data[t].nunique() > 1]
    return " + ".join(keep)


def safe_fit(formula, data, is_logit):
    """
    Robust model fitting with multiple fallback strategies.
    
    Attempts to fit statistical models with various error handling
    approaches to ensure convergence even with problematic data.
    
    Parameters
    ----------
    formula : str
        Model formula in patsy notation (e.g., "y ~ x1 + x2")
    data : pd.DataFrame
        Dataset for model estimation
    is_logit : bool
        If True, fit logistic regression; otherwise OLS
        
    Returns
    -------
    statsmodels.regression.linear_model.RegressionResultsWrapper or None
        Fitted model object, or None if all attempts fail
    """
    # Parse formula components
    lhs, rhs = map(str.strip, formula.split('~', 1))
    
    # Remove constant predictors
    terms = [t.strip() for t in rhs.split('+')]
    keep = [t for t in terms if (t not in data.columns) or (data[t].nunique() > 1)]
    rhs_clean = " + ".join(keep) if keep else "1"  # Default to intercept-only model
    
    cleaned_formula = f"{lhs} ~ {rhs_clean}"
    
    if is_logit:
        # Attempt 1: Logit with robust standard errors (HC3)
        try:
            return smf.logit(cleaned_formula, data=data).fit(disp=False, cov_type="HC3")
        except Exception:
            pass
            
        # Attempt 2: Logit with default standard errors
        try:
            return smf.logit(cleaned_formula, data=data).fit(disp=False)
        except Exception:
            pass
            
        # Attempt 3: GLM with binomial family
        try:
            return smf.glm(cleaned_formula, data=data, family=sm.families.Binomial()).fit()
        except Exception:
            pass
            
        # Attempt 4: Regularized logit with minimal penalty
        try:
            return smf.logit(cleaned_formula, data=data).fit_regularized(alpha=1e-6, disp=False)
        except Exception as e:
            print(f"WARNING: Model estimation failed after all attempts: {e}")
            return None
    else:
        # OLS estimation with robust standard errors
        try:
            return smf.ols(cleaned_formula, data=data).fit(cov_type="HC3")
        except Exception:
            return smf.ols(cleaned_formula, data=data).fit()


def run_models(models, data, label):
    """
    Execute a series of nested models and display results.
    
    Parameters
    ----------
    models : list of str
        List of model formulas to estimate
    data : pd.DataFrame
        Dataset for estimation
    label : str
        Description of model set (e.g., "OLS: Total Listings")
    """
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    
    for i, fml in enumerate(models, 1):
        print(f"\nModel {i}: {fml}")
        res = safe_fit(fml, data, label.startswith("Logit"))
        
        if res is None:
            print("Model estimation failed - skipping.")
        else:
            print(f"Observations: {int(res.nobs)}")
            print(res.summary())


# ================================ DOCUMENTATION FUNCTIONS ================================

def print_professional_terms_appendix():
    """
    Generate Appendix A: Professional terminology documentation.
    
    Provides complete list of terms used in professionalism classification
    along with methodological notes on term selection.
    """
    print("\n" + "="*70)
    print("APPENDIX A: PROFESSIONALISM TERMINOLOGY")
    print("="*70)
    print("\nTerms identified as indicators of professional vendor practices:")
    print("(Case-insensitive matching applied)\n")
    
    # Display terms in numbered list format
    for i, term in enumerate(PRO_TERMS, 1):
        print(f"{i:2d}. {term}")
    
    print(f"\nTotal terms in classifier: {len(PRO_TERMS)}")
    print("\nExcluded terms (rationale):")
    print("- 'stealth': Too general, common in non-professional contexts")
    print("- 'discreet': Too general (note: 'discreet shipping' is included)")
    print("- 'tablet', 'pack', 'bottle': Common product descriptors, not professionalism indicators")
    print("="*70)


def print_drug_classification_appendix():
    """
    Generate Appendix B: Drug category classification methodology.
    
    Documents the hierarchical classification system and representative
    terms used for drug category assignment.
    """
    print("\n" + "="*70)
    print("APPENDIX B: DRUG CATEGORY CLASSIFICATION METHODOLOGY")
    print("="*70)
    print("\nHierarchical classification system based on marketplace taxonomy")
    print("and pharmacological categories.\n")
    
    print("CLASSIFICATION METHODOLOGY:")
    print("-" * 50)
    print("1. PRIMARY CLASSIFICATION")
    print("   - Source: Main 'Category' field in marketplace data")
    print("   - Example: 'Drugs/Cannabis/Weed' → Primary category: 'Cannabis'\n")
    
    print("2. SECONDARY CLASSIFICATION")
    print("   - Source: Second-level hierarchical category token")
    print("   - Example: 'Drugs/Cannabis/Weed' → Secondary token: 'Cannabis'\n")
    
    print("3. CATEGORY DEFINITIONS AND TERMS:")
    print("-" * 50)
    
    for category, terms in DRUG_CATEGORY_TERMS.items():
        print(f"\n{category.upper()}:")
        # Display representative terms (truncated for readability)
        term_display = ", ".join(terms[:8])
        if len(terms) > 8:
            term_display += f" ... (+{len(terms)-8} additional terms)"
        print(f"  Terms: {term_display}")
    
    print(f"\n4. BINARY INDICATOR VARIABLES:")
    print("   - class_[Category]: Vendor has ≥1 listing in primary category")
    print("   - sec_[Category]: Vendor has ≥1 listing with secondary classification")
    print("\n" + "="*70)


def print_sample_statistics():
    """
    Generate summary statistics for the analyzed sample.
    
    Provides key descriptive statistics about the vendor population
    and marketplace characteristics.
    """
    print("\n" + "="*50)
    print("SAMPLE STATISTICS")
    print("="*50)
    print(f"Unique vendors analyzed: {len(vendor):,}")
    print(f"Total listings processed: {len(df):,}")
    print(f"Vendors using PGP: {vendor['pgp_present'].sum():,} ({vendor['pgp_present'].mean():.1%})")
    print(f"Mean listings per vendor: {vendor['total_listings'].mean():.1f}")
    print(f"Mean categories per vendor: {vendor['num_categories'].mean():.1f}")
    print(f"Mean professionalism score: {vendor['pct_professional'].mean():.1%}")
    print("="*50)


# ================================ MAIN ANALYSIS EXECUTION ================================
"""
Execute all regression analyses in sequence.
Models are estimated with robust standard errors where applicable.
"""

print("Initializing vendor analysis pipeline...")

# Execute primary analyses
run_models(OLS_SIZE, vendor, "OLS: Total Listings")
run_models(OLS_RATE, vendor, "OLS: Average Rating")
run_models(LOG_TOP, vendor, "Logit: Top Vendor (90th percentile)")
run_models(LOG_LARGE, vendor, "Logit: Large Vendor (above median)")


# ================================ DATA EXPORT ================================
"""
Save processed datasets for further analysis and replication.
"""

# Export vendor-level dataset
vendor.to_csv(OUT_FILE, index=False)
print(f"\nVendor-level dataset saved: {OUT_FILE}")

# Export product-level dataset with computed flags
df.to_csv(PRODUCT_OUT_FILE, index=False, encoding="utf-8")
print(f"Product-level dataset saved: {PRODUCT_OUT_FILE}")


# ================================ GENERATE DOCUMENTATION ================================
"""
Generate comprehensive documentation appendices for methodology transparency.
"""

print_sample_statistics()
print_professional_terms_appendix()
print_drug_classification_appendix()

print("\nAnalysis pipeline completed successfully.")


# ================================ MODEL COMPARISON ANALYSIS ================================

def run_poisson_comparison(ols_models, data, label):
    """
    Compare OLS and Poisson GLM for count data modeling.
    
    Count data (e.g., number of listings) may be better modeled using
    Poisson regression due to its discrete, non-negative nature.
    This function fits both model types and compares fit statistics.
    
    Parameters
    ----------
    ols_models : list of str
        Model formulas to estimate
    data : pd.DataFrame
        Dataset for estimation
    label : str
        Description for output display
        
    Returns
    -------
    list of dict
        Model comparison statistics (AIC, BIC) for each specification
    """
    print(f"\n{'='*60}")
    print(f"{label} - OLS vs Poisson GLM Comparison")
    print(f"{'='*60}")
    
    comparison_results = []
    
    for i, formula in enumerate(ols_models, 1):
        print(f"\nModel {i}: {formula}")
        
        # Fit OLS model
        try:
            ols_model = smf.ols(formula, data=data).fit(cov_type="HC3")
            ols_aic = ols_model.aic
            ols_bic = ols_model.bic
            print(f"OLS       - AIC: {ols_aic:.1f}, BIC: {ols_bic:.1f}")
        except Exception as e:
            print(f"OLS estimation failed: {e}")
            ols_aic = ols_bic = None
        
        # Fit Poisson GLM
        try:
            poisson_model = smf.glm(formula, data=data, family=sm.families.Poisson()).fit()
            poisson_aic = poisson_model.aic
            poisson_bic = poisson_model.bic
            print(f"Poisson   - AIC: {poisson_aic:.1f}, BIC: {poisson_bic:.1f}")
        except Exception as e:
            print(f"Poisson estimation failed: {e}")
            poisson_aic = poisson_bic = None
        
        # Store comparison results
        comparison_results.append({
            'Model': i,
            'Formula': formula,
            'OLS_AIC': ols_aic,
            'OLS_BIC': ols_bic,
            'Poisson_AIC': poisson_aic,
            'Poisson_BIC': poisson_bic
        })
    
    return comparison_results


# Execute model comparison
print("\n" + "="*80)
print("ROBUSTNESS CHECK: COUNT DATA MODEL SPECIFICATION")
print("="*80)

comparison_results = run_poisson_comparison(OLS_SIZE, vendor, "Vendor Size Models")

# Display comparison summary table
print(f"\n{'Model':<8} {'OLS AIC':<12} {'OLS BIC':<12} {'Poisson AIC':<14} {'Poisson BIC':<14} {'Preferred':<12}")
print("-" * 80)

for result in comparison_results:
    model_num = result['Model']
    ols_aic = result['OLS_AIC']
    ols_bic = result['OLS_BIC']
    poisson_aic = result['Poisson_AIC']
    poisson_bic = result['Poisson_BIC']
    
    # Determine preferred model based on AIC (lower is better)
    if ols_aic is not None and poisson_aic is not None:
        preferred = "OLS" if ols_aic < poisson_aic else "Poisson"
    else:
        preferred = "N/A"
    
    # Format output row
    print(f"{model_num:<8} {ols_aic:<12.1f} {ols_bic:<12.1f} {poisson_aic:<14.1f} {poisson_bic:<14.1f} {preferred:<12}")

# Summary statistics for methods section
print(f"\n{'='*50}")
print("MODEL COMPARISON SUMMARY (Full Model)")
print(f"{'='*50}")

if comparison_results:
    final_result = comparison_results[-1]  # Model 5 (full specification)
    print(f"OLS (HC3 robust SE):     AIC = {final_result['OLS_AIC']:.0f}, BIC = {final_result['OLS_BIC']:.0f}")
    print(f"Poisson GLM:             AIC = {final_result['Poisson_AIC']:.0f}, BIC = {final_result['Poisson_BIC']:.0f}")


# ================================ EXTENDED ANALYSIS WITH RATING CONTROLS ================================
"""
Re-estimate models with average rating as additional control variable.
This addresses potential confounding between vendor quality and size.
"""

# Reload vendor data to ensure clean state
vendor = pd.read_csv(OUT_FILE)

# Create binary indicator for high-rated vendors
vendor["high_rating"] = (vendor["avg_rating"] >= vendor["avg_rating"].median()).astype(int)

# Redefine model specifications with rating controls
b1 = "pgp_present"
b2 = b1 + " + num_categories"
b3 = b2 + " + avg_price_btc"
b4_with_rating = b3 + " + pct_professional + avg_rating"  # Include rating for non-rating DVs
b4_no_rating = b3 + " + pct_professional"                # Exclude rating when it's the DV

CLASS_RHS = " + ".join(c for c in vendor.columns if c.startswith(("class_", "sec_")))
b5_with_rating = b4_with_rating + " + " + CLASS_RHS
b5_no_rating = b4_no_rating + " + " + CLASS_RHS

# Construct model formulas
MODEL_RHSS_WITH_RATING = [b1, b2, b3, b4_with_rating, b5_with_rating]
MODEL_RHSS_NO_RATING = [b1, b2, b3, b4_no_rating, b5_no_rating]

OLS_SIZE = [f"total_listings ~ {rhs}" for rhs in MODEL_RHSS_WITH_RATING]
OLS_RATE = [f"avg_rating ~ {rhs}" for rhs in MODEL_RHSS_NO_RATING]
LOG_TOP = [f"top_vendor ~ {rhs}" for rhs in MODEL_RHSS_WITH_RATING]
LOG_LARGE = [f"large_vendor ~ {rhs}" for rhs in MODEL_RHSS_WITH_RATING]
LOG_HIGH = [f"high_rating ~ {rhs}" for rhs in MODEL_RHSS_NO_RATING]


# ================================ REGRESSION OUTPUT TABLES ================================
"""
Generate formatted regression tables for publication.
All models estimated with heteroskedasticity-robust standard errors.
"""

print("\nGenerating regression output tables...")

# 1. Descriptive statistics table
desc_cols = ["total_listings", "num_categories", "avg_price_btc", "avg_rating"]
desc = vendor[desc_cols]
stats = desc.agg(["mean", "median", "std", "min", "max"]).T
stats.columns = ["Mean", "Median", "SD", "Min", "Max"]
stats.to_csv("descriptives.csv", index=True)
print("✓ Descriptive statistics saved: descriptives.csv")

# 2. OLS nested models - Total listings
coeffs = []
for i, formula in enumerate(OLS_SIZE, start=1):
    res = safe_fit(formula, vendor, is_logit=False)
    if res is not None:
        df = res.params.to_frame("Coef")
        df["SE"] = res.bse
        df["t"] = res.tvalues
        df["p"] = res.pvalues
        df["Sig"] = df["p"].apply(lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "")
        df["Model"] = f"Model {i}"
        df = df.reset_index().rename(columns={"index": "Term"})
        coeffs.append(df)
pd.concat(coeffs).to_csv("ols_size_nested.csv", index=False)
print("✓ OLS size models saved: ols_size_nested.csv")

# 3. Full OLS model - Total listings
res_full = safe_fit(OLS_SIZE[-1], vendor, is_logit=False)
if res_full is not None:
    full = res_full.params.to_frame("Coef")
    full["SE"] = res_full.bse
    full["t"] = res_full.tvalues
    full["p"] = res_full.pvalues
    full["Sig"] = full["p"].apply(lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "")
    full.reset_index().rename(columns={"index": "Term"}).to_csv("ols_size_full.csv", index=False)
    print("✓ Full OLS size model saved: ols_size_full.csv")

# 4. OLS nested models - Average rating
coeffs = []
for i, formula in enumerate(OLS_RATE, start=1):
    res = safe_fit(formula, vendor, is_logit=False)
    if res is not None:
        df = res.params.to_frame("Coef")
        df["SE"] = res.bse
        df["t"] = res.tvalues
        df["p"] = res.pvalues
        df["Sig"] = df["p"].apply(lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "")
        df["Model"] = f"Model {i}"
        coeffs.append(df.reset_index().rename(columns={"index": "Term"}))
pd.concat(coeffs).to_csv("ols_rating_nested.csv", index=False)
print("✓ OLS rating models saved: ols_rating_nested.csv")

# 5. Full OLS model - Average rating
res_rating_full = safe_fit(OLS_RATE[-1], vendor, is_logit=False)
if res_rating_full is not None:
    full_rating = res_rating_full.params.to_frame("Coef")
    full_rating["SE"] = res_rating_full.bse
    full_rating["t"] = res_rating_full.tvalues
    full_rating["p"] = res_rating_full.pvalues
    full_rating["Sig"] = full_rating["p"].apply(lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "")
    full_rating.reset_index().rename(columns={"index": "Term"}).to_csv("ols_rating_full.csv", index=False)
    print("✓ Full OLS rating model saved: ols_rating_full.csv")

# 6. Logit nested models - Top vendor
coeffs = []
for i, formula in enumerate(LOG_TOP, start=1):
    res = safe_fit(formula, vendor, is_logit=True)
    if res is not None:
        df = res.params.to_frame("Coef")
        df["SE"] = res.bse
        df["z"] = res.tvalues
        df["p"] = res.pvalues
        df["Sig"] = df["p"].apply(lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "")
        df["Model"] = f"Model {i}"
        coeffs.append(df.reset_index().rename(columns={"index": "Term"}))
pd.concat(coeffs).to_csv("logit_top_nested.csv", index=False)
print("✓ Logit top vendor models saved: logit_top_nested.csv")

# 7. Full logit model - Top vendor
res_top_full = safe_fit(LOG_TOP[-1], vendor, is_logit=True)
if res_top_full is not None:
    full_top = res_top_full.params.to_frame("Coef")
    full_top["SE"] = res_top_full.bse
    full_top["z"] = res_top_full.tvalues
    full_top["p"] = res_top_full.pvalues
    full_top["Sig"] = full_top["p"].apply(lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "")
    full_top.reset_index().rename(columns={"index": "Term"}).to_csv("logit_top_full.csv", index=False)
    print("✓ Full logit top vendor model saved: logit_top_full.csv")

# 8. Logit nested models - Large vendor
coeffs = []
for i, formula in enumerate(LOG_LARGE, start=1):
    res = safe_fit(formula, vendor, is_logit=True)
    if res is not None:
        df = res.params.to_frame("Coef")
        df["SE"] = res.bse
        df["z"] = res.tvalues
        df["p"] = res.pvalues
        df["Sig"] = df["p"].apply(lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "")
        df["Model"] = f"Model {i}"
        coeffs.append(df.reset_index().rename(columns={"index": "Term"}))
pd.concat(coeffs).to_csv("logit_large_nested.csv", index=False)
print("✓ Logit large vendor models saved: logit_large_nested.csv")

# 9. Full logit model - Large vendor
res_large_full = safe_fit(LOG_LARGE[-1], vendor, is_logit=True)
if res_large_full is not None:
    full_large = res_large_full.params.to_frame("Coef")
    full_large["SE"] = res_large_full.bse
    full_large["z"] = res_large_full.tvalues
    full_large["p"] = res_large_full.pvalues
    full_large["Sig"] = full_large["p"].apply(lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "")
    full_large.reset_index().rename(columns={"index": "Term"}).to_csv("logit_large_full.csv", index=False)
    print("✓ Full logit large vendor model saved: logit_large_full.csv")

# 10. Logit nested models - High rating
coeffs = []
for i, formula in enumerate(LOG_HIGH, start=1):
    res = safe_fit(formula, vendor, is_logit=True)
    if res is not None:
        df = res.params.to_frame("Coef")
        df["SE"] = res.bse
        df["z"] = res.tvalues
        df["p"] = res.pvalues
        df["Sig"] = df["p"].apply(lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "")
        df["Model"] = f"Model {i}"
        coeffs.append(df.reset_index().rename(columns={"index": "Term"}))
pd.concat(coeffs).to_csv("logit_high_rating_nested.csv", index=False)
print("✓ Logit high rating models saved: logit_high_rating_nested.csv")

# 11. Full logit model - High rating
res_high_full = safe_fit(LOG_HIGH[-1], vendor, is_logit=True)
if res_high_full is not None:
    full_high = res_high_full.params.to_frame("Coef")
    full_high["SE"] = res_high_full.bse
    full_high["z"] = res_high_full.tvalues
    full_high["p"] = res_high_full.pvalues
    full_high["Sig"] = full_high["p"].apply(lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "")
    full_high.reset_index().rename(columns={"index": "Term"}).to_csv("logit_high_rating_full.csv", index=False)
    print("✓ Full logit high rating model saved: logit_high_rating_full.csv")

print("\nRegression output generation completed.")
print("\nExported files:")
print("  - descriptives.csv               : Summary statistics")
print("  - ols_size_nested.csv           : OLS vendor size (nested models)")
print("  - ols_size_full.csv             : OLS vendor size (full model)")
print("  - ols_rating_nested.csv         : OLS average rating (nested models)")
print("  - ols_rating_full.csv           : OLS average rating (full model)")
print("  - logit_top_nested.csv          : Logit top vendor (nested models)")
print("  - logit_top_full.csv            : Logit top vendor (full model)")
print("  - logit_large_nested.csv        : Logit large vendor (nested models)")
print("  - logit_large_full.csv          : Logit large vendor (full model)")
print("  - logit_high_rating_nested.csv  : Logit high rating (nested models)")
print("  - logit_high_rating_full.csv    : Logit high rating (full model)")


# ================================ FIGURE GENERATION ================================
"""
Generate publication-quality figures for manuscript.
All figures use consistent styling and color schemes.
"""

print("\n" + "="*80)
print("FIGURE GENERATION MODULE")
print("="*80)


def tighten_layout(fig, title):
    """
    Apply consistent formatting to plotly figures.
    
    Ensures uniform appearance across all visualizations with
    appropriate margins, font sizes, and title positioning.
    
    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure object to format
    title : str
        Figure title text
        
    Returns
    -------
    plotly.graph_objects.Figure
        Formatted figure object
    """
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(size=14)
    )
    return fig


# Load vendor data for visualization
VENDOR_CSV = "vendor_model_ready_v3.csv"
if not os.path.exists(VENDOR_CSV):
    sys.exit(f"ERROR: {VENDOR_CSV} not found. Run main analysis first.")

vendor = pd.read_csv(VENDOR_CSV)

# Create categorical labels for binary variables
vendor["pgp_label"] = vendor["pgp_present"].map({0: "No PGP", 1: "PGP"})

# Define consistent color scheme
PGP_COLOURS = {0: "crimson", 1: "royalblue"}  # Red: No PGP, Blue: PGP

# Create log-transformed vendor size for visualization
vendor["log_listings"] = np.log1p(vendor["total_listings"])


# ================================ PROFESSIONALISM TERM FREQUENCY ================================
"""
Analyze frequency of professional language terms across all listings.
"""

print("\nGenerating professionalism term frequency analysis...")

# Reload raw data for term analysis
RAW_FILE = "Agora.csv"
if not os.path.exists(RAW_FILE):
    sys.exit(f"ERROR: {RAW_FILE} not found.")

# Identify description column
with open(RAW_FILE, encoding="latin1") as fh:
    first_line = fh.readline().strip().split(",")

DESC_COLS = [c for c in first_line if "descr" in c.lower()]
if not DESC_COLS:
    sys.exit("ERROR: No description column found.")
DESC_COL = DESC_COLS[0]
print(f"Using description column: {DESC_COL}")

# Load only required columns
raw_df = pd.read_csv(
    RAW_FILE,
    encoding="latin1",
    usecols=["Vendor", DESC_COL]
)

# Count term occurrences
counts = (
    raw_df[DESC_COL].fillna("").str.lower()
    .apply(lambda txt: {t: int(bool(re.search(re.escape(t), txt))) for t in PRO_TERMS})
    .apply(pd.Series)
    .sum()
    .sort_values(ascending=False)
)

# Prepare data for visualization
cue_df = counts.reset_index().rename(columns={"index": "cue", 0: "count"})
cue_df["cue"] = cue_df["cue"].str.title()  # Capitalize for display

# Create horizontal bar chart
fig_cues = px.bar(
    cue_df,
    y="cue", x="count", orientation="h",
    labels={"count": "Number of Listings", "cue": "Professionalism Indicator"},
    title="Frequency of Professionalism Indicators in Vendor Listings",
    color_discrete_sequence=[PGP_COLOURS[1]]  # Use consistent blue color
)
fig_cues.update_layout(yaxis=dict(dtick=1))  # Show all labels
tighten_layout(fig_cues, fig_cues.layout.title.text)

# Create output directory if needed
os.makedirs("figures", exist_ok=True)
fig_cues.write_image("figures/bar_professionalism_terms.pdf")
print("✓ Saved: figures/bar_professionalism_terms.pdf")


# ================================ FIGURE 1: VENDOR SIZE DISTRIBUTION ================================
"""
Violin plot showing distribution of vendor sizes by PGP status.
"""

fig1 = px.violin(
    vendor,
    x="pgp_label", y="log_listings",
    color="pgp_label", box=True, points="all",
    color_discrete_map=PGP_COLOURS,
    labels={"pgp_label": "PGP Status", "log_listings": "Log(1 + Total Listings)"},
    title="Distribution of Vendor Size by PGP Usage"
)
fig1.update_layout(showlegend=False)  # Legend redundant with x-axis labels
tighten_layout(fig1, fig1.layout.title.text)
fig1.write_image("figures/violin_pgp_size.pdf")
print("✓ Saved: figures/violin_pgp_size.pdf")


# ================================ FIGURE 2: PROFESSIONALISM SCORES ================================
"""
Bar chart comparing average professionalism scores by PGP status.
"""

pgp_prof = (
    vendor.groupby("pgp_label", as_index=False)["pct_professional"]
    .mean()
)

fig2 = px.bar(
    pgp_prof,
    x="pgp_label", y="pct_professional",
    color="pgp_label", text_auto=".2f",  # Display values on bars
    color_discrete_map=PGP_COLOURS,
    labels={"pgp_label": "PGP Status",
            "pct_professional": "Average Professionalism Score"},
    title="Mean Professionalism Score by PGP Status"
)
fig2.update_layout(showlegend=False)
tighten_layout(fig2, fig2.layout.title.text)
fig2.write_image("figures/bar_summary_pgp.pdf")
print("✓ Saved: figures/bar_summary_pgp.pdf")


# ================================ FIGURE 3: SIZE VS DIVERSIFICATION ================================
"""
Scatter plot showing relationship between vendor size and category diversification.
"""

fig3 = px.scatter(
    vendor,
    x="num_categories", y="log_listings",
    color="pgp_label",
    color_discrete_map=PGP_COLOURS,
    labels={"num_categories": "Number of Product Categories",
            "log_listings": "Log(1 + Total Listings)",
            "pgp_label": "PGP Status"},
    title="Vendor Size vs. Product Diversification by PGP Status"
)
fig3.update_traces(marker=dict(size=6))  # Increase marker size for visibility
tighten_layout(fig3, fig3.layout.title.text)
fig3.write_image("figures/scatter_div_size.pdf")
print("✓ Saved: figures/scatter_div_size.pdf")


# ================================ FIGURE 4: PREDICTED PROBABILITIES ================================
"""
Line plot showing predicted probability of being a top vendor.
"""

GRID_CSV = "grid_pred_top.csv"
if not os.path.exists(GRID_CSV):
    print("WARNING: grid_pred_top.csv not found - skipping Figure 4.")
else:
    grid_pred = pd.read_csv(GRID_CSV)
    grid_pred["pgp_label"] = grid_pred["pgp_present"].map({0: "No PGP", 1: "PGP"})
    
    fig4 = px.line(
        grid_pred, x="num_categories", y="predicted_prob",
        color="pgp_label", markers=True,
        labels={"num_categories": "Number of Product Categories",
                "predicted_prob": "Predicted Probability (Top Vendor)",
                "pgp_label": "PGP Status"},
        title="Predicted Probability of Top Vendor Status by Diversification and PGP",
        color_discrete_map=PGP_COLOURS
    )
    fig4.update_layout(
        yaxis=dict(range=[0, 1]),  # Probability scale
        legend_title_text="PGP Status"
    )
    tighten_layout(fig4, fig4.layout.title.text)
    fig4.write_image("figures/pred_prob_top_div_pgp.pdf")
    print("✓ Saved: figures/pred_prob_top_div_pgp.pdf")


# ================================ FIGURE 5: OUTLIER ANALYSIS ================================
"""
Scatter plot highlighting exceptional non-PGP vendors.
"""

# Define vendors of interest
OUTLIERS = ["mssource", "RXChemist", "medibuds", "rc4me", "Gotmilk"]

# Create outlier indicator
vendor["highlight"] = np.where(
    vendor["Vendor"].isin(OUTLIERS), "Outlier", "Regular"
)

# Standardize diversification measure
vendor["num_categories_z"] = (
    (vendor["num_categories"] - vendor["num_categories"].mean()) /
    vendor["num_categories"].std()
)

fig5 = px.scatter(
    vendor, x="num_categories_z", y="log_listings",
    color="pgp_label", symbol="highlight",
    color_discrete_map=PGP_COLOURS,
    labels={"num_categories_z": "Product Diversification (Z-score)",
            "log_listings": "Log(1 + Total Listings)",
            "pgp_label": "PGP Status"},
    title="Vendor Size vs. Diversification: Non-PGP Outliers Highlighted",
    symbol_map={"Outlier": "x", "Regular": "circle"}
)

# Add vendor name annotations
for v in OUTLIERS:
    r = vendor[vendor["Vendor"] == v]
    if not r.empty:
        fig5.add_annotation(
            x=r["num_categories_z"].iloc[0],
            y=r["log_listings"].iloc[0],
            text=v, showarrow=True, arrowhead=1
        )

tighten_layout(fig5, fig5.layout.title.text)
fig5.write_image("figures/scatter_outliers_nonpgp.pdf")
print("✓ Saved: figures/scatter_outliers_nonpgp.pdf")

print("\nFigure generation completed.")
print("\nAll analysis outputs have been generated successfully.")