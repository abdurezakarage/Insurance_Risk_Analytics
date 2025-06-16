# Insurance Risk Analytics

A comprehensive toolkit for analyzing insurance risk, performing exploratory data analysis, hypothesis testing, and building predictive models for insurance claims and premiums. This project is designed for data scientists, and analysts working with insurance datasets to gain insights and build robust risk models.

## Features
- **Data Processing:** Clean and preprocess raw insurance data for analysis and modeling.
- **Exploratory Data Analysis:** Visualize and summarize key metrics, missing values, outliers, and trends.
- **Hypothesis Testing:** Statistical tests to compare risk and margin across provinces, zip codes, and demographic groups.
- **Machine Learning Models:** Train and evaluate regression and classification models (Linear Regression, Random Forest, XGBoost, etc.) for claim prediction and risk-based premium calculation.
- **Model Interpretation:** Feature importance and SHAP value analysis for business insights.
- **Jupyter Notebooks:** Example analyses and workflows for reproducibility.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. **Prepare your data:** Place your raw data file (e.g., `MachineLearningRating_v3.txt`) in the `data/` directory.
2. **Data Processing:**
   - Use the scripts in `src/services/process_data.py` to clean and preprocess your data.
3. **Exploratory Analysis:**
   - Use the `InsuranceAnalysis` class in `src/services/insurance_analysis.py` for EDA and visualization.
4. **Hypothesis Testing:**
   - Use the `Test` class in `src/services/hypothesis_test.py` to run statistical tests.
5. **Model Training:**
   - Use the `modelTrain` class in `src/services/modeltrain.py` to train and evaluate predictive models.
6. **Jupyter Notebooks:**
   - Explore the `notebooks/` directory for example workflows and analyses.

## Directory Structure

```
Insurance_Risk_Analytics/
├── data/                # Raw and processed data files
├── notebooks/           # Jupyter notebooks for analysis
├── src/
│   └── services/        # Main modules: data processing, analysis, modeling
├── examples/            # Example scripts and usage
├── scripts/             # Utility scripts
├── tests/               # Unit and integration tests
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Project metadata
└── README.md            # Project documentation
```

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

