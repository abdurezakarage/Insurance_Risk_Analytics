import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import shap
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

class modelTrain:
    def __init__(self, df):
        self.df = df.copy()
        self.df['vehicle_age'] = 2025 - self.df['RegistrationYear']
        self.df['has_claim'] = (self.df['TotalClaims'] > 0).astype(int)
        self.handle_missing_data()
        self.feature_engineering()
        self.preprocess()

    def handle_missing_data(self):
        # Impute numeric columns with median, drop rows with too many missing values
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            self.df[col] = self.df[col].fillna(self.df[col].median())
        # For categorical columns, fill with mode
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

    def feature_engineering(self):
        # Example: Claim frequency per year, premium per claim, etc.
        self.df['claim_frequency'] = self.df['TotalClaims'] / (2025 - self.df['RegistrationYear'] + 1)
        self.df['premium_per_claim'] = self.df['TotalPremium'] / (self.df['TotalClaims'] + 1)
        # Add more as needed

    def preprocess(self):
        # One-hot encode categorical features
        categoricals = ['Gender', 'Province', 'VehicleType', 'CoverType']
        existing_categoricals = [col for col in categoricals if col in self.df.columns]
        if existing_categoricals:
            self.df = pd.get_dummies(self.df, columns=existing_categoricals, drop_first=True)
        # Drop or convert date columns
        date_cols = [col for col in self.df.columns if 'date' in col.lower()]
        for col in date_cols:
            try:
                self.df[col] = pd.to_datetime(self.df[col])
                self.df[col + '_year'] = self.df[col].dt.year
                self.df = self.df.drop(columns=[col])
            except Exception:
                self.df = self.df.drop(columns=[col])
        # Handle any remaining non-numeric columns that might be dates
        non_numeric_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        for col in non_numeric_cols:
            try:
                self.df[col] = pd.to_datetime(self.df[col])
                self.df[col + '_year'] = self.df[col].dt.year
                self.df = self.df.drop(columns=[col])
                # print(f"Converted and extracted year from column: {col}")
            except Exception:
                # print(f"Column '{col}' could not be converted to datetime and will be dropped.")
                self.df = self.df.drop(columns=[col])
        # Print non-numeric columns after preprocessing
        non_numeric_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        # if non_numeric_cols:
        #     print(f"Non-numeric columns after preprocessing: {non_numeric_cols}")
        # else:
        #     print("All columns are numeric after preprocessing.")
        # Print columns with NaNs before filling
        nan_cols = self.df.columns[self.df.isna().any()].tolist()
        # if nan_cols:
        #     print("Columns with NaNs before filling:")
        #     print(self.df[nan_cols].isna().sum())
        # else:
        #     print("No columns have NaNs before filling.")
        # Fill any remaining NaN values in numeric columns with the median
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            self.df[col] = self.df[col].fillna(self.df[col].median())
        # Drop columns that are still all NaN
        still_nan_cols = self.df.columns[self.df.isna().all()].tolist()
        if still_nan_cols:
            # print(f"Dropping columns still all NaN after median fill: {still_nan_cols}")
            self.df = self.df.drop(columns=still_nan_cols)

    def prepare_data(self):
        features = [col for col in self.df.columns if col not in ['TotalClaims', 'TotalPremium', 'CalculatedPremiumPerTerm']]
        self.X = self.df[features]
        self.y_claim_amount = self.df['TotalClaims']
        self.y_has_claim = self.df['has_claim']

        self.X_train_c, self.X_test_c, self.y_train_c, self.y_test_c = train_test_split(
            self.X[self.df['TotalClaims'] > 0], 
            self.y_claim_amount[self.df['TotalClaims'] > 0], 
            test_size=0.2, random_state=42
        )
        self.X_train_p, self.X_test_p, self.y_train_p, self.y_test_p = train_test_split(
            self.X, self.y_has_claim, test_size=0.2, random_state=42
        )

    def train_models(self):
        self.models = {
            'LinearRegression': LinearRegression(),
            'DecisionTree': DecisionTreeRegressor(random_state=42),
            'RandomForest': RandomForestRegressor(random_state=42),
            'XGBoost': XGBRegressor(random_state=42)
        }
        self.results = {}
        for name, model in self.models.items():
            model.fit(self.X_train_c, self.y_train_c)
            preds = model.predict(self.X_test_c)
            rmse = np.sqrt(mean_squared_error(self.y_test_c, preds))
            r2 = r2_score(self.y_test_c, preds)
            self.results[name] = {'RMSE': rmse, 'R2': r2}

    def train_classifiers(self):
        self.classifiers = {
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42)
        }
        self.classification_results = {}
        for name, clf in self.classifiers.items():
            clf.fit(self.X_train_p, self.y_train_p)
            preds = clf.predict(self.X_test_p)
            self.classification_results[name] = {
                'Accuracy': accuracy_score(self.y_test_p, preds),
                'F1': f1_score(self.y_test_p, preds),
                'Precision': precision_score(self.y_test_p, preds),
                'Recall': recall_score(self.y_test_p, preds)
            }

    def predict_premium(self):
        best_class_model = self.classifiers['XGBoost']
        best_reg_model = self.models['XGBoost']

        claim_prob = best_class_model.predict_proba(self.X_test_c)[:, 1]
        expected_severity = best_reg_model.predict(self.X_test_c)

        self.risk_based_premium = claim_prob * expected_severity + 0.10 * expected_severity + 100
        return self.risk_based_premium

    def interpret_model(self):
        best_reg_model = self.models['XGBoost']
        explainer = shap.Explainer(best_reg_model, self.X_test_c)
        shap_values = explainer(self.X_test_c)
        return shap_values

    def print_results(self):
        print("\n--- Regression Model Performance ---")
        for name, metrics in self.results.items():
            print(f"{name} → RMSE: {metrics['RMSE']:.2f}, R²: {metrics['R2']:.3f}")

        print("\n--- Classification Model Performance ---")
        for name, metrics in self.classification_results.items():
            print(f"{name} → Accuracy: {metrics['Accuracy']:.2f}, F1: {metrics['F1']:.2f}, Precision: {metrics['Precision']:.2f}, Recall: {metrics['Recall']:.2f}")

    def model_comparison_report(self):
        print("\n--- Regression Model Comparison ---")
        for name, metrics in self.results.items():
            print(f"{name:15} | RMSE: {metrics['RMSE']:.2f} | R²: {metrics['R2']:.3f}")
        print("\n--- Classification Model Comparison ---")
        for name, metrics in self.classification_results.items():
            print(f"{name:15} | Accuracy: {metrics['Accuracy']:.2f} | F1: {metrics['F1']:.2f} | Precision: {metrics['Precision']:.2f} | Recall: {metrics['Recall']:.2f}")

    def shap_feature_importance(self, top_n=10):
        importances = {}
        best_reg_model = self.models['XGBoost']
        explainer = shap.Explainer(best_reg_model, self.X_test_c)
        shap_values = explainer(self.X_test_c)
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        feature_names = self.X_test_c.columns
        sorted_idx = np.argsort(mean_abs_shap)[::-1][:top_n]
        print(f"\nTop {top_n} Most Influential Features (SHAP):")
        for idx in sorted_idx:
            print(f"{feature_names[idx]}: mean(|SHAP|) = {mean_abs_shap[idx]:.4f}")
        print("\nBusiness Interpretation:")
        for idx in sorted_idx:
            print(f"Feature '{feature_names[idx]}' increases/decreases the predicted claim amount by {mean_abs_shap[idx]:.2f} units on average, holding other factors constant.")
        return [(feature_names[idx], mean_abs_shap[idx]) for idx in sorted_idx]
