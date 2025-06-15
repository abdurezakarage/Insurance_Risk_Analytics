import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from matplotlib.ticker import FuncFormatter

class InsuranceAnalysis:
    def __init__(self, df):
        self.df = df.copy()
        self.original_df = df.copy()  # Keep original data

    def load_data(self, file_path):
        """Load the insurance data from CSV file."""
        return pd.read_csv(file_path)

    def analyze_data_types(self):
        """Analyze and display data types of all columns."""
        print("\n=== Data Types Analysis ===")
        dtype_info = pd.DataFrame({
            'Data Type': self.df.dtypes,
            'Non-Null Count': self.df.count(),
            'Null Count': self.df.isnull().sum(),
            'Null Percentage': (self.df.isnull().sum() / len(self.df) * 100).round(2)
        })
        print(dtype_info)
        return dtype_info

    def analyze_data_structure(self):
        """Analyze data types and structure of the dataset."""
        print("\nData Types Information:")
        print(self.df.dtypes)
        
        print("\nMemory Usage:")
        print(self.df.memory_usage(deep=True).sum() / 1024**2, "MB")
        
        # Convert date columns if they exist
        date_columns = ['TransactionMonth', 'VehicleIntroDate']
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col])
                print(f"\nConverted {col} to datetime")

    def analyze_missing_values(self):
        """Comprehensive analysis of missing values in the dataset."""
        # Calculate missing value statistics
        missing_stats = pd.DataFrame({
            'Missing Count': self.df.isnull().sum(),
            'Missing Percentage': (self.df.isnull().sum() / len(self.df) * 100).round(2),
            'Non-Null Count': self.df.count(),
            'Total Rows': len(self.df)
        })

        # Sort by missing percentage
        missing_stats = missing_stats.sort_values('Missing Percentage', ascending=False)

        print("\n=== Missing Values Analysis ===")
        print("\nMissing Values Summary:")
        print(missing_stats)

        # Plot missing values heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(self.df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.tight_layout()
        plt.show()

        # Plot missing values percentage bar chart
        plt.figure(figsize=(12, 6))
        missing_stats['Missing Percentage'].plot(kind='bar')
        plt.title('Missing Values Percentage by Column')
        plt.xlabel('Columns')
        plt.ylabel('Missing Percentage')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Analyze patterns in missing values
        print("\nMissing Values Patterns:")
        missing_patterns = self.analyze_missing_patterns()
        print(missing_patterns)

        return missing_stats

    def analyze_missing_patterns(self):
        """Analyze patterns in missing values across columns."""
        missing_matrix = self.df.isnull().astype(int)
        missing_corr = missing_matrix.corr()

        missing_patterns = []
        for col1 in missing_corr.columns:
            for col2 in missing_corr.columns:
                if col1 < col2:
                    corr = missing_corr.loc[col1, col2]
                    if corr > 0.5:
                        missing_patterns.append({
                            'Column 1': col1,
                            'Column 2': col2,
                            'Correlation': corr
                        })

        return pd.DataFrame(missing_patterns)

    def analyze_missing_by_category(self, category_col):
        """Analyze missing values distribution across categories."""
        if category_col in self.df.columns:
            print(f"\nMissing Values Analysis by {category_col}:")
            missing_by_cat = self.df.groupby(category_col).apply(
                lambda x: (x.isnull().sum() / len(x) * 100).round(2)
            )

            plt.figure(figsize=(12, 6))
            missing_by_cat.plot(kind='bar')
            plt.title(f'Missing Values Percentage by {category_col}')
            plt.xlabel(category_col)
            plt.ylabel('Missing Percentage')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            return missing_by_cat
        else:
            print(f"Column '{category_col}' not found in the dataset.")
            return None

    def suggest_missing_value_strategies(self):
        """Suggest strategies for handling missing values based on data analysis."""
        missing_stats = self.df.isnull().sum() / len(self.df) * 100

        strategies = []
        for col in self.df.columns:
            missing_pct = missing_stats[col]
            dtype = self.df[col].dtype

            if missing_pct > 0:
                strategy = {
                    'Column': col,
                    'Missing Percentage': round(missing_pct, 2),
                    'Data Type': dtype,
                    'Suggested Strategy': self._get_strategy(col, missing_pct, dtype)
                }
                strategies.append(strategy)

        return pd.DataFrame(strategies)

    def _get_strategy(self, col, missing_pct, dtype):
        """Determine the best strategy for handling missing values."""
        if missing_pct > 50:
            return "Consider dropping the column if not critical"
        elif dtype in ['float64', 'int64']:
            if missing_pct < 5:
                return "Use mean/median imputation"
            else:
                return "Use advanced imputation (KNN, MICE) or model-based imputation"
        elif dtype == 'object':
            if missing_pct < 5:
                return "Use mode imputation"
            else:
                return "Use advanced imputation or create 'Missing' category"
        elif 'date' in str(dtype).lower():
            return "Use forward/backward fill or interpolation"
        else:
            return "Review data and domain knowledge for appropriate strategy"

    def check_total_claim_premium(self):
        """Check missing stats specifically for TotalPremium and TotalClaim."""
        missing_stats = self.analyze_missing_values()
        selected = missing_stats.loc[missing_stats.index.isin(['TotalPremium', 'TotalClaim'])]
        print("\n=== Missing Summary for TotalPremium and TotalClaim ===")
        print(selected)

        pattern_df = self.analyze_missing_patterns()
        print("\n=== Missing Pattern Correlation between TotalPremium and TotalClaim ===")
        print(pattern_df[
            (pattern_df['Column 1'].isin(['TotalPremium', 'TotalClaim'])) &
            (pattern_df['Column 2'].isin(['TotalPremium', 'TotalClaim']))
        ])

        strategy_df = self.suggest_missing_value_strategies()
        print("\n=== Suggested Strategies for TotalPremium and TotalClaim ===")
        print(strategy_df[strategy_df['Column'].isin(['TotalPremium', 'TotalClaim'])])

    def outliers_boxplot(self, columns=['TotalPremium', 'TotalClaims']):
        """
        Detect and remove outliers using the IQR method and visualize with box plots.

        Parameters:
        -----------
        columns : list
            List of numerical columns to analyze for outliers.

        Returns:
        --------
        DataFrame
            DataFrame with outliers removed.
        """
        df_clean = self.df.copy()
        rows_to_drop = set()

        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Identify outliers
            outlier_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            outlier_indices = df_clean[col][outlier_mask].index
            rows_to_drop.update(outlier_indices)

            # Plot boxplot
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=df_clean[col])
            plt.title(f"Boxplot for {col}")
            plt.xlabel(col)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # Print stats
            print(f"\nOutlier Removal for {col}:")
            print(f"Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
            print(f"Number of outliers removed: {len(outlier_indices)}")
            print(f"Percentage of data removed: {(len(outlier_indices)/len(df_clean))*100:.2f}%")

        # Drop the rows
        df_clean = df_clean.drop(index=rows_to_drop)
        print(f"\nTotal rows removed: {len(rows_to_drop)}")
        print(f"Remaining rows: {len(df_clean)}")

        # Update self.df
        self.df = df_clean

        return df_clean

    def analyze_portfolio(self):
        """Analyze overall portfolio metrics and key insights."""
        # Debug information
        print(f"Total Premium sum: {self.df['TotalPremium'].sum():,.2f}")
        print(f"Total Claims sum: {self.df['TotalClaims'].sum():,.2f}")
        print(f"Number of records: {len(self.df)}")
        
        # Check for zero or negative values
        zero_premium = (self.df['TotalPremium'] <= 0).sum()
        zero_claims = (self.df['TotalClaims'] < 0).sum()
        print(f"Records with zero or negative premium: {zero_premium}")
        print(f"Records with negative claims: {zero_claims}")
        
        # Overall Loss Ratio with error handling
        total_premium = self.df['TotalPremium'].sum()
        total_claims = self.df['TotalClaims'].sum()
        
        if total_premium > 0:
            loss_ratio = total_claims / total_premium
            print("\nOverall Portfolio Loss Ratio:")
            print(f"Loss Ratio: {loss_ratio:.4f}")
        else:
            print("\nError: Total Premium is zero or negative, cannot calculate loss ratio")

    def plot_box(self, column):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, y=column)
        plt.title(f'Box Plot of {column}')
        plt.show()

    def analyze_financial_distributions(self):
        """Analyze distributions of key financial variables."""
        # Box plots for TotalClaims and TotalPremium
        self.plot_box('TotalClaims')
        self.plot_box('TotalPremium')
        
        # Distribution of CustomValueEstimate
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['CustomValueEstimate'], bins=50)
        plt.title('Distribution of Custom Value Estimate')
        plt.xlabel('Custom Value Estimate')
        plt.ylabel('Frequency')
        plt.show()

    def analyze_temporal_trends(self):
        """Analyze temporal trends in claims and premiums."""
        if 'TransactionMonth' in self.df.columns:
            monthly_data = self.df.groupby('TransactionMonth').agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum'
            })
            plt.figure(figsize=(12, 6))
            monthly_data.plot()
            plt.title('Monthly Claims and Premiums Trend')
            plt.xlabel('Month')
            plt.ylabel('Amount')
            plt.grid(True)
            plt.show()
            # Monthly Loss Ratio Trend
            monthly_loss_ratio = self.df.groupby('TransactionMonth').apply(
                lambda x: x['TotalClaims'].sum() / x['TotalPremium'].sum()
            )
            plt.figure(figsize=(12, 6))
            monthly_loss_ratio.plot()
            plt.title('Monthly Loss Ratio Trend')
            plt.ylabel('Loss Ratio')
            plt.xlabel('Month')
            plt.grid(True)
            plt.show()

    def analyze_vehicle_metrics(self):
        """Analyze vehicle-related metrics and patterns."""
        # Top 10 Makes by Average Claim Amount
        make_claims = self.df.groupby('Make')['TotalClaims'].mean().sort_values(ascending=False)
        plt.figure(figsize=(12, 6))
        make_claims.head(10).plot(kind='bar')
        plt.title('Top 10 Vehicle Makes by Average Claim Amount')
        plt.xlabel('Make')
        plt.ylabel('Average Claim Amount')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Vehicle Type vs Loss Ratio
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='VehicleType', y='LossRatio', data=self.df)
        plt.title('Loss Ratio Distribution by Vehicle Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def analyze_variability(self):
        """Calculate variability metrics for numerical features."""
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        
        variability_metrics = pd.DataFrame({
            'Mean': self.df[numerical_cols].mean(),
            'Std': self.df[numerical_cols].std(),
            'CV': self.df[numerical_cols].std() / self.df[numerical_cols].mean(),  # Coefficient of Variation
            'Skewness': self.df[numerical_cols].skew(),
            'Kurtosis': self.df[numerical_cols].kurtosis()
        })
        
        print("\nVariability Metrics for Numerical Features:")
        print(variability_metrics)
        return variability_metrics

    def analyze_distributions(self):
        """Plot distributions for numerical and categorical variables."""
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        # Plot numerical distributions
        for col in numerical_cols[:5]:  # Limit to first 5 numerical columns
            plt.figure(figsize=(10, 6))
            sns.histplot(data=self.df, x=col, kde=True)
            plt.title(f'Distribution of {col}')
            plt.show()
        
        # Plot categorical distributions
        for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
            plt.figure(figsize=(10, 6))
            value_counts = self.df[col].value_counts()
            sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.title(f'Distribution of {col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    def analyze_geographic_trends(self):
        """
        Analyze trends across different geographical regions.
        """
        if 'Province' in self.df.columns:
            # Create a figure with multiple subplots
            fig = plt.figure(figsize=(20, 15))
            gs = fig.add_gridspec(3, 2)
            
            # 1. Insurance Cover Type Distribution by Province
            ax1 = fig.add_subplot(gs[0, 0])
            cover_by_province = pd.crosstab(self.df['Province'], self.df['CoverType'])
            cover_by_province.plot(kind='bar', stacked=True, ax=ax1)
            ax1.set_title('Insurance Cover Type Distribution by Province')
            ax1.set_xlabel('Province')
            ax1.set_ylabel('Count')
            plt.xticks(rotation=45)
            
            # 2. Average Premium by Province
            ax2 = fig.add_subplot(gs[0, 1])
            premium_by_province = self.df.groupby('Province')['TotalPremium'].mean().sort_values(ascending=False)
            sns.barplot(x=premium_by_province.index, y=premium_by_province.values, ax=ax2)
            ax2.set_title('Average Premium by Province')
            ax2.set_xlabel('Province')
            ax2.set_ylabel('Average Premium')
            plt.xticks(rotation=45)
            
            # 3. Auto Make Distribution by Province
            ax3 = fig.add_subplot(gs[1, :])
            make_by_province = pd.crosstab(self.df['Province'], self.df['make'])
            make_by_province.plot(kind='bar', stacked=True, ax=ax3)
            ax3.set_title('Auto Make Distribution by Province')
            ax3.set_xlabel('Province')
            ax3.set_ylabel('Count')
            plt.xticks(rotation=45)
            
            # 4. Claims Ratio by Province
            ax4 = fig.add_subplot(gs[2, 0])
            self.df['ClaimsRatio'] = self.df['TotalClaims'] / self.df['TotalPremium']
            claims_by_province = self.df.groupby('Province')['ClaimsRatio'].mean().sort_values(ascending=False)
            sns.barplot(x=claims_by_province.index, y=claims_by_province.values, ax=ax4)
            ax4.set_title('Average Claims Ratio by Province')
            ax4.set_xlabel('Province')
            ax4.set_ylabel('Claims Ratio')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()
            
            # Print summary statistics
            print("\nGeographic Trends Summary:")
            print("\n1. Premium Statistics by Province:")
            print(self.df.groupby('Province')['TotalPremium'].agg(['mean', 'std', 'min', 'max']))
            
            print("\n3. Most Common Cover Types by Province:")
            print(cover_by_province.idxmax(axis=1))
            
            print("\n4. Most Common Auto Makes by Province:")
            print(make_by_province.idxmax(axis=1))
            
            return {
                'premium_stats': self.df.groupby('Province')['TotalPremium'].agg(['mean', 'std', 'min', 'max']),
                'cover_type_dist': cover_by_province,
                'auto_make_dist': make_by_province
            }

    def compare_regional_metrics(self, metric='TotalPremium', group_by='Province'):
        """
        Compare specific metrics across regions with detailed statistics.
        
        Parameters:
        -----------
        metric : str
            The metric to compare (e.g., 'TotalPremium', 'TotalClaims', 'VehicleAge')
        group_by : str
            The geographical grouping variable (e.g., 'Province', 'PostalCode')
        """
        if group_by in self.df.columns and metric in self.df.columns:
            # Calculate statistics
            stats_df = self.df.groupby(group_by)[metric].agg([
                'count', 'mean', 'std', 'min', 'max',
                lambda x: x.quantile(0.25),  # Q1
                lambda x: x.quantile(0.75)   # Q3
            ]).rename(columns={
                '<lambda_0>': 'Q1',
                '<lambda_1>': 'Q3'
            })
            
            # Create visualizations
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Box plot
            sns.boxplot(x=group_by, y=metric, data=self.df, ax=ax1)
            ax1.set_title(f'{metric} Distribution by {group_by}')
            ax1.set_xlabel(group_by)
            ax1.set_ylabel(metric)
            plt.xticks(rotation=45)
            
            # Bar plot with error bars
            sns.barplot(x=group_by, y=metric, data=self.df, ax=ax2, 
                       ci='sd', capsize=0.1)
            ax2.set_title(f'Average {metric} by {group_by} with Standard Deviation')
            ax2.set_xlabel(group_by)
            ax2.set_ylabel(f'Average {metric}')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()
            
            # Print detailed statistics
            print(f"\nDetailed Statistics for {metric} by {group_by}:")
            print(stats_df)
            
            # Calculate and print regional differences
            print("\nRegional Differences:")
            for region1 in stats_df.index:
                for region2 in stats_df.index:
                    if region1 < region2:
                        diff = stats_df.loc[region1, 'mean'] - stats_df.loc[region2, 'mean']
                        pct_diff = (diff / stats_df.loc[region2, 'mean']) * 100
                        print(f"{region1} vs {region2}:")
                        print(f"  Absolute Difference: {diff:.2f}")
                        print(f"  Percentage Difference: {pct_diff:.2f}%")
            
            return stats_df

    def analyze_regional_patterns(self):
        """
        Analyze patterns and relationships between different regional metrics.
        """
        if 'Province' in self.df.columns:
            # Calculate key metrics by province
            regional_metrics = self.df.groupby('Province').agg({
                'TotalPremium': ['mean', 'std'],
                'TotalClaims': ['mean', 'std'],
                'VehicleAge': ['mean', 'std'],
                'CoverType': lambda x: x.mode().iloc[0],
                'AutoMake': lambda x: x.mode().iloc[0]
            })
            
            # Create a correlation heatmap of regional metrics
            plt.figure(figsize=(10, 8))
            sns.heatmap(regional_metrics['TotalPremium'].join(regional_metrics['TotalClaims']).corr(),
                       annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation of Regional Metrics')
            plt.show()
            
            # Print regional patterns
            print("\nRegional Patterns Summary:")
            print("\n1. Premium and Claims Patterns:")
            print(regional_metrics[['TotalPremium', 'TotalClaims']])
            
            print("\n2. Most Common Cover Types by Province:")
            print(regional_metrics['CoverType'])
            
            print("\n3. Most Common Auto Makes by Province:")
            print(regional_metrics['AutoMake'])
            
            return regional_metrics

    def analyze_numerical_features(self):
        """Analyze numerical features with comprehensive statistics."""
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        
        stats_df = pd.DataFrame({
            'Count': self.df[numerical_cols].count(),
            'Mean': self.df[numerical_cols].mean(),
            'Std': self.df[numerical_cols].std(),
            'Min': self.df[numerical_cols].min(),
            '25%': self.df[numerical_cols].quantile(0.25),
            '50%': self.df[numerical_cols].median(),
            '75%': self.df[numerical_cols].quantile(0.75),
            'Max': self.df[numerical_cols].max(),
            'Skewness': self.df[numerical_cols].skew(),
            'Kurtosis': self.df[numerical_cols].kurtosis(),
            'CV': (self.df[numerical_cols].std() / self.df[numerical_cols].mean()).round(3)  # Coefficient of Variation
        })
        
        print("\n=== Numerical Features Analysis ===")
        print(stats_df)
        return stats_df

    def analyze_categorical_features(self):
        """Analyze categorical features with value counts and unique values."""
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        print("\n=== Categorical Features Analysis ===")
        for col in categorical_cols:
            print(f"\n{col}:")
            value_counts = self.df[col].value_counts()
            print(f"Unique values: {len(value_counts)}")
            print(f"Most common values:\n{value_counts.head()}")
            print(f"Least common values:\n{value_counts.tail()}")

    def analyze_date_features(self):
        """Analyze date features if they exist."""
        date_columns = ['TransactionMonth', 'VehicleIntroDate']
        date_cols = [col for col in date_columns if col in self.df.columns]
        
        if date_cols:
            print("\n=== Date Features Analysis ===")
            for col in date_cols:
                print(f"\n{col}:")
                print(f"Date range: {self.df[col].min()} to {self.df[col].max()}")
                print(f"Number of unique dates: {self.df[col].nunique()}")
                
                # Plot date distribution
                plt.figure(figsize=(12, 6))
                self.df[col].value_counts().sort_index().plot(kind='line')
                plt.title(f'Distribution of {col}')
                plt.xlabel('Date')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()

    def plot_numerical_distributions(self):
        """Plot distributions of numerical features."""
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        
        for col in numerical_cols:
            # Create subplot with histogram and boxplot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Histogram with KDE
            sns.histplot(data=self.df, x=col, kde=True, ax=ax1)
            ax1.set_title(f'Distribution of {col}')
            
            # Boxplot
            sns.boxplot(data=self.df, x=col, ax=ax2)
            ax2.set_title(f'Boxplot of {col}')
            
            plt.tight_layout()
            plt.show()

    def analyze_categorical_distributions(self, columns=None, top_n=10):
        """
        Analyze and plot distributions of categorical variables.
        
        Parameters:
        -----------
        columns : list, optional
            List of categorical columns to analyze. If None, analyzes all categorical columns.
        top_n : int, default=10
            Number of top categories to show in plots.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=['object']).columns
        
        for col in columns:
            # Create figure with subplots
            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(2, 2)
            
            # Value counts
            value_counts = self.df[col].value_counts()
            
            # Bar plot of top N categories
            ax1 = fig.add_subplot(gs[0, :])
            value_counts.head(top_n).plot(kind='bar', ax=ax1)
            ax1.set_title(f'Top {top_n} Categories in {col}')
            ax1.set_xlabel(col)
            ax1.set_ylabel('Count')
            plt.xticks(rotation=45)
            
            # Pie chart of top N categories
            ax2 = fig.add_subplot(gs[1, 0])
            value_counts.head(top_n).plot(kind='pie', autopct='%1.1f%%', ax=ax2)
            ax2.set_title(f'Distribution of Top {top_n} Categories in {col}')
            
            # Summary statistics
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.axis('off')
            stats_text = f"""
            Summary Statistics for {col}:
            
            Total Categories: {len(value_counts)}
            Most Common: {value_counts.index[0]} ({value_counts.iloc[0]:,} occurrences)
            Least Common: {value_counts.index[-1]} ({value_counts.iloc[-1]:,} occurrences)
            Null Values: {self.df[col].isnull().sum():,}
            Null Percentage: {(self.df[col].isnull().sum() / len(self.df) * 100):.2f}%
            """
            ax3.text(0.1, 0.5, stats_text, fontsize=10, va='center')
            
            plt.tight_layout()
            plt.show()

    def analyze_monthly_changes_by_zipcode(self):
        """
        Analyze monthly changes in TotalPremium and TotalClaims by ZipCode.
        """
        if 'PostalCode' in self.df.columns and 'TransactionMonth' in self.df.columns:
            # Group by PostalCode and TransactionMonth
            monthly_data = self.df.groupby(['PostalCode', 'TransactionMonth']).agg({
                'TotalPremium': 'sum',
                'TotalClaims': 'sum'
            }).reset_index()
            
            # Calculate monthly changes
            monthly_data['Premium_Change'] = monthly_data.groupby('PostalCode')['TotalPremium'].diff()
            monthly_data['Claims_Change'] = monthly_data.groupby('PostalCode')['TotalClaims'].diff()
            
            # Create visualizations
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Scatter plot of Premium vs Claims
            sns.scatterplot(data=monthly_data, x='TotalPremium', y='TotalClaims', 
                          hue='PostalCode', alpha=0.6, ax=ax1)
            ax1.set_title('Total Premium vs Total Claims by ZipCode')
            
            # Scatter plot of Monthly Changes
            sns.scatterplot(data=monthly_data, x='Premium_Change', y='Claims_Change',
                          hue='PostalCode', alpha=0.6, ax=ax2)
            ax2.set_title('Monthly Changes in Premium vs Claims by ZipCode')
            
            # Line plot of Premium trends
            for zipcode in monthly_data['PostalCode'].unique():
                zip_data = monthly_data[monthly_data['PostalCode'] == zipcode]
                ax3.plot(zip_data['TransactionMonth'], zip_data['TotalPremium'], 
                        label=zipcode, alpha=0.6)
            ax3.set_title('Premium Trends by ZipCode')
            ax3.tick_params(axis='x', rotation=45)
            
            # Line plot of Claims trends
            for zipcode in monthly_data['PostalCode'].unique():
                zip_data = monthly_data[monthly_data['PostalCode'] == zipcode]
                ax4.plot(zip_data['TransactionMonth'], zip_data['TotalClaims'],
                        label=zipcode, alpha=0.6)
            ax4.set_title('Claims Trends by ZipCode')
            ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.show()
            
            # Calculate correlations
            correlation = monthly_data[['TotalPremium', 'TotalClaims', 
                                     'Premium_Change', 'Claims_Change']].corr()
            
            print("\nCorrelation Matrix:")
            print(correlation)
            
            return monthly_data, correlation

    def analyze_correlations(self, columns=None):
        """
        Analyze correlations between numerical variables.
        
        Parameters:
        -----------
        columns : list, optional
            List of columns to analyze. If None, analyzes all numerical columns.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=['float64', 'int64']).columns
        
        # Calculate correlation matrix
        corr_matrix = self.df[columns].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.5:  # Threshold for strong correlation
                    strong_correlations.append({
                        'Variable 1': columns[i],
                        'Variable 2': columns[j],
                        'Correlation': corr
                    })
        
        if strong_correlations:
            print("\nStrong Correlations (|r| > 0.5):")
            print(pd.DataFrame(strong_correlations))
        
        return corr_matrix

    def analyze_pairwise_relationships(self, columns=None):
        """
        Analyze pairwise relationships between numerical variables.
        
        Parameters:
        -----------
        columns : list, optional
            List of columns to analyze. If None, analyzes all numerical columns.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=['float64', 'int64']).columns
        
        # Create pairplot
        sns.pairplot(self.df[columns], diag_kind='kde')
        plt.show()
        
        # Create detailed scatter plots with regression lines
        n_cols = len(columns)
        fig, axes = plt.subplots(n_cols, n_cols, figsize=(15, 15))
        
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i != j:
                    sns.regplot(data=self.df, x=col1, y=col2, ax=axes[i, j], scatter_kws={'alpha':0.5})
                else:
                    sns.histplot(data=self.df, x=col1, ax=axes[i, j], kde=True)
                
                if i == n_cols-1:
                    axes[i, j].set_xlabel(col2)
                if j == 0:
                    axes[i, j].set_ylabel(col1)
        
        plt.tight_layout()
        plt.show()

    def run_analysis(self):
        """Run the complete analysis pipeline."""
        print("=== Data Structure Analysis ===")
        self.analyze_data_types()
        
        print("\n=== Data Quality Assessment ===")
        self.analyze_missing_values()
        
        print("\n=== Missing Values by Category ===")
        self.analyze_missing_by_category('Province')
        self.analyze_missing_by_category('VehicleType')
        
        print("\n=== Missing Value Handling Strategies ===")
        strategies = self.suggest_missing_value_strategies()
        print(strategies)
        
        print("\n=== Numerical Features Analysis ===")
        self.analyze_numerical_features()
        
        print("\n=== Numerical Distributions ===")
        self.analyze_numerical_distributions()
        
        print("\n=== Categorical Distributions ===")
        self.analyze_categorical_distributions()
        
        print("\n=== Date Features Analysis ===")
        self.analyze_date_features()
        
        print("\n=== Monthly Changes by ZipCode ===")
        self.analyze_monthly_changes_by_zipcode()
        
        print("\n=== Correlation Analysis ===")
        self.analyze_correlations()
        
        print("\n=== Pairwise Relationships ===")
        self.analyze_pairwise_relationships()
        
        print("\n=== Outlier Detection (Z-score) ===")
        self.detect_outliers_zscore()
        
        print("\n=== Outlier Removal (Z-score > 3) ===")
        self.drop_outliers_zscore()
        
        print("\n=== Geographic Trends Analysis ===")
        self.analyze_geographic_trends()
        
        print("\n=== Regional Metrics Comparison ===")
        self.compare_regional_metrics('TotalPremium', 'Province')
        self.compare_regional_metrics('TotalClaims', 'Province')
        
        print("\n=== Regional Patterns Analysis ===")
        self.analyze_regional_patterns()
        
        print("\n=== Creating Insightful Visualizations ===")
        self.create_insightful_plots()

