import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind

class Test:
    # Load the processed data
    def load_data(self, file_path):
        self.df = pd.read_csv(file_path)

        # Standardize and derive columns
        self.df['has_claim'] = self.df['TotalClaims'].fillna(0).apply(lambda x: 1 if x > 0 else 0)
        self.df['Gender'] = self.df['Gender'].str.lower().str.strip()
        self.df['Province'] = self.df['Province'].str.title().str.strip()
        self.df['zip_code'] = self.df['PostalCode'].astype(str).str.strip()
        self.df['claim_amount'] = self.df['TotalClaims'].fillna(0)
        self.df['premium'] = self.df['TotalPremium'].fillna(0)
        self.df['margin'] = self.df['premium'] - self.df['claim_amount']

    # KPI calculation
    def calculate_claim_frequency(self, group):
        return group['has_claim'].mean()

    def calculate_claim_severity(self, group):
        return group[group['has_claim'] == 1]['claim_amount'].mean()

    def calculate_margin(self, group):
        return group['margin'].mean()

    # Statistical tests
    def chi_squared_test(self, df, feature, target='has_claim'):
        contingency = pd.crosstab(df[feature], df[target])
        chi2, p, dof, expected = chi2_contingency(contingency)
        return p

    def t_test(self, group1, group2, target):
        return ttest_ind(group1[target], group2[target], equal_var=False, nan_policy='omit').pvalue

    # Hypothesis testing for province
    def test_province_risk(self, province_a, province_b, min_samples=5):
        df = self.df
        group_a = df[df['Province'] == province_a]
        group_b = df[df['Province'] == province_b]

        if len(group_a) < min_samples or len(group_b) < min_samples:
            return None

        group_a_claims = group_a[group_a['has_claim'] == 1]
        group_b_claims = group_b[group_b['has_claim'] == 1]

        if len(group_a_claims) < 2 or len(group_b_claims) < 2:
            return None

        if np.isclose(np.std(group_a_claims['claim_amount']), 0) and np.isclose(np.std(group_b_claims['claim_amount']), 0):
            return None

        freq_p = self.chi_squared_test(df[df['Province'].isin([province_a, province_b])], 'Province')
        sev_p = self.t_test(group_a_claims, group_b_claims, 'claim_amount')

        return {
            'hypothesis': 'Risk differences across provinces',
            'feature': 'Province',
            'group_a': province_a,
            'group_b': province_b,
            'claim_frequency_p': freq_p,
            'claim_severity_p': sev_p
        }

    def test_all_province_pairs(self):
        results = []
        provinces = self.df['Province'].dropna().unique()
        for i in range(len(provinces)):
            for j in range(i + 1, len(provinces)):
                result = self.test_province_risk(provinces[i], provinces[j])
                if result is not None:
                    results.append(result)
        return results

    # Hypothesis testing for zip code
    def test_zipcode_risk(self, zip_a, zip_b, min_samples=5):
        df = self.df
        group_a = df[df['zip_code'] == zip_a]
        group_b = df[df['zip_code'] == zip_b]

        if len(group_a) < min_samples or len(group_b) < min_samples:
            return None

        group_a_claims = group_a[group_a['has_claim'] == 1]
        group_b_claims = group_b[group_b['has_claim'] == 1]

        if len(group_a_claims) < 2 or len(group_b_claims) < 2:
            return None

        if np.isclose(np.std(group_a_claims['claim_amount']), 0) and np.isclose(np.std(group_b_claims['claim_amount']), 0):
            return None

        freq_p = self.chi_squared_test(df[df['zip_code'].isin([zip_a, zip_b])], 'zip_code')
        sev_p = self.t_test(group_a_claims, group_b_claims, 'claim_amount')

        return {
            'hypothesis': 'Risk differences across zip codes',
            'feature': 'zip_code',
            'group_a': zip_a,
            'group_b': zip_b,
            'claim_frequency_p': freq_p,
            'claim_severity_p': sev_p
        }

    def test_all_zipcode_pairs(self):
        # Get the top N zip codes by count
        N = 5
        top_zips = self.df['zip_code'].value_counts().head(N).index.tolist()

        # Compare only among these top zip codes
        results = []
        for i in range(len(top_zips)):
            for j in range(i + 1, len(top_zips)):
                result = self.test_zipcode_risk(top_zips[i], top_zips[j])
                if result is not None:
                    results.append(result)

        return results

    # Hypothesis testing for margin differences by zip
    def test_margin_difference(self, zip_a, zip_b, min_samples=5):
        df = self.df
        group_a = df[df['zip_code'] == zip_a]
        group_b = df[df['zip_code'] == zip_b]

        if len(group_a) < min_samples or len(group_b) < min_samples:
            return None

        if np.isclose(np.std(group_a['margin']), 0) and np.isclose(np.std(group_b['margin']), 0):
            return None

        margin_p = self.t_test(group_a, group_b, 'margin')

        return {
            'hypothesis': 'Margin differences across zip codes',
            'feature': 'zip_code',
            'group_a': zip_a,
            'group_b': zip_b,
            'margin_p': margin_p
        }
    def test_all_zipcode_margin_pairs(self, N=5, min_samples=5):
        # Get the top N zip codes by count
        top_zips = self.df['zip_code'].value_counts().head(N).index.tolist()
        results = []
        for i in range(len(top_zips)):
            for j in range(i + 1, len(top_zips)):
                result = self.test_margin_difference(top_zips[i], top_zips[j], min_samples=min_samples)
                if result is not None:
                    results.append(result)
        return results

    # Hypothesis testing by gender
    def test_gender_risk(self, min_samples=5):
        df = self.df
        group_a = df[df['Gender'] == 'male']
        group_b = df[df['Gender'] == 'female']

        if len(group_a) < min_samples or len(group_b) < min_samples:
            return None

        group_a_claims = group_a[group_a['has_claim'] == 1]
        group_b_claims = group_b[group_b['has_claim'] == 1]

        if len(group_a_claims) < 2 or len(group_b_claims) < 2:
            return None

        if np.isclose(np.std(group_a_claims['claim_amount']), 0) and np.isclose(np.std(group_b_claims['claim_amount']), 0):
            return None

        freq_p = self.chi_squared_test(df[df['Gender'].isin(['male', 'female'])], 'Gender')
        sev_p = self.t_test(group_a_claims, group_b_claims, 'claim_amount')

        return {
            'hypothesis': 'Risk differences by gender',
            'feature': 'Gender',
            'group_a': 'male',
            'group_b': 'female',
            'claim_frequency_p': freq_p,
            'claim_severity_p': sev_p
        }

    # Generate report
    def generate_report(self, result):
        print(f"\n📌 Hypothesis: {result['hypothesis']}")
        print(f"Comparing {result['group_a']} vs {result['group_b']}")
        for key, value in result.items():
            if '_p' in key:
                decision = "REJECT" if value < 0.05 else "FAIL TO REJECT"
                print(f"  {key}: {value:.4f} ➤ {decision} the null hypothesis.")

 