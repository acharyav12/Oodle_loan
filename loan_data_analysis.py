import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)

# Load data
df = pd.read_csv('Case_Study_for_Data_Test_25.csv')

# Data Cleaning
# Remove currency symbols and commas from Loan_amount and Deposit, convert to float
def clean_currency(x):
    if isinstance(x, str):
        return float(x.replace('£', '').replace(',', '').replace('NA', '').strip() or np.nan)
    return x

df['Loan_amount'] = df['Loan_amount'].apply(clean_currency)
df['Deposit'] = df['Deposit'].apply(clean_currency)

# Convert APR to float, handle NA
if 'APR' in df.columns:
    df['APR'] = pd.to_numeric(df['APR'], errors='coerce')

# Remove duplicate rows
initial_shape = df.shape
df = df.drop_duplicates()
print(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows.")

# Basic Info
print("\n--- DATA OVERVIEW ---")
print(df.info())
print(df.head())
print("\nMissing values per column:\n", df.isnull().sum())

# Summary statistics
print("\n--- SUMMARY STATISTICS ---")
print(df.describe(include='all'))

# Approval and Funding Rates
print("\n--- APPLICATION OUTCOMES ---")
if 'Application_outcome' in df.columns:
    print(df['Application_outcome'].value_counts(normalize=True))
    if 'Funded' in df.columns:
        approved = df[df['Application_outcome'] == 'Approved']
        print("\nFunding rate among approved:")
        print(approved['Funded'].value_counts(normalize=True))

# Grouped Analysis
print("\n--- GROUPED ANALYSIS ---")
if 'Car_type' in df.columns:
    print("\nBy Car Type:")
    print(df.groupby('Car_type')['Application_outcome'].value_counts(normalize=True).unstack())
if 'Area' in df.columns:
    print("\nBy Area:")
    print(df.groupby('Area')['Application_outcome'].value_counts(normalize=True).unstack())
if 'Age' in df.columns:
    df['Age_group'] = pd.cut(df['Age'], bins=[17,25,35,45,55,65,100], labels=['18-25','26-35','36-45','46-55','56-65','66+'])
    print("\nBy Age Group:")
    print(df.groupby('Age_group')['Application_outcome'].value_counts(normalize=True).unstack())

# Visualizations
def plot_count(col, title):
    plt.figure(figsize=(8,4))
    sns.countplot(x=col, data=df, order=df[col].value_counts().index)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_box(x, y, title):
    plt.figure(figsize=(8,4))
    sns.boxplot(x=x, y=y, data=df)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Countplots
if 'Car_type' in df.columns:
    plot_count('Car_type', 'Applications by Car Type')
if 'Area' in df.columns:
    plot_count('Area', 'Applications by Area')
if 'Application_outcome' in df.columns:
    plot_count('Application_outcome', 'Application Outcomes')
if 'Funded' in df.columns:
    plot_count('Funded', 'Funded Loans')

# Boxplots
if 'Loan_amount' in df.columns and 'Car_type' in df.columns:
    plot_box('Car_type', 'Loan_amount', 'Loan Amount by Car Type')
if 'Loan_amount' in df.columns and 'Area' in df.columns:
    plot_box('Area', 'Loan_amount', 'Loan Amount by Area')
if 'Loan_amount' in df.columns and 'Application_outcome' in df.columns:
    plot_box('Application_outcome', 'Loan_amount', 'Loan Amount by Application Outcome')
if 'APR' in df.columns and 'Application_outcome' in df.columns:
    plot_box('Application_outcome', 'APR', 'APR by Application Outcome')

# Correlation heatmap
num_cols = df.select_dtypes(include=np.number).columns
if len(num_cols) > 1:
    plt.figure(figsize=(10,8))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

# Save cleaned data for further analysis
cleaned_path = 'Case_Study_for_Data_Test_25_cleaned.csv'
df.to_csv(cleaned_path, index=False)
print(f"\nCleaned data saved to {cleaned_path}")

print("\n--- ANALYSIS COMPLETE ---")
