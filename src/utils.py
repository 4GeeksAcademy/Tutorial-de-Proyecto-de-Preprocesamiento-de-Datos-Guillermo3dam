from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns



# load the .env file variables
load_dotenv()


def db_connect():
    import os
    engine = create_engine(os.getenv('DATABASE_URL'))
    engine.connect()
    return engine


def missing_values_summary(df):

    missing_count = df.isnull().sum().loc[lambda x: x > 0]

    missing_percent = round((missing_count / len(df)) * 100, 2)

    missing_df = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing Percent (%)": missing_percent
    })

    missing_df = missing_df.sort_values(by="Missing Count", ascending=False)

    return missing_df


def remove_duplicates(df, id = None):

    filt_df = df.copy()

    num_duplicates  = filt_df.drop("id", axis = 1).duplicated().sum()
    
    print(f"Number of duplicate rows (excluding '{id}'): {num_duplicates}")

    if num_duplicates:
        filt_df = filt_df.drop_duplicates(subset=filt_df.columns.difference([id]))
        print(f"Duplicates removed. New shape: {filt_df.shape}")
    else:
        print("No duplicates found.")

    return filt_df



def factorize_categorical(df):

    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    print("Categorical cols:", categorical_cols)

    total_data = df.copy()
    transformation_rules = {}

    for col in categorical_cols:
        total_data[f"{col}_n"], uniques = pd.factorize(total_data[col])
        rules_dict = {name: code for code, name in enumerate(uniques)}
        transformation_rules[col] = rules_dict

    with open(f"../data/interim/rules_transformation_{col}.json", "w") as f:
        json.dump(rules_dict, f)

    return total_data


def plot_correlation_heatmap(df):
    corr_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.show()

def replace_outliers(column, df):
  stats = df[column].describe()
  iqr = stats["75%"] - stats["25%"]
  upper_limit = stats["75%"] + 1.5 * iqr
  lower_limit = stats["25%"] - 1.5 * iqr

  if lower_limit < 0: lower_limit = min(df[column])
  df[column] = df[column].clip(lower_limit, upper_limit)

  print(f"\nThe lower_limit of variable {column} is {round(lower_limit, 3)} and the upper_limit is {round(upper_limit, 3)} and the IQR is {round(iqr, 3)}")

  return df, {
    "lower_limit": round(float(lower_limit), 3),
    "upper_limit": round(float(upper_limit), 3)
  }

def fill_missing_values(df, median_cols=None, mode_cols=None, mean_cols=None):
    df_copy = df.copy()

    # Cuando una variable es entera pero numérica/incontable
    if median_cols:
        for col in median_cols:
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
   
    # Cuando una variable es categórica
    if mode_cols:
        for col in mode_cols:
            df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])

    # Cuando una variable es puramente decimal
    if mean_cols:
        for col in mean_cols:
            df_copy[col] = df_copy[col].fillna(df_copy[col].mean())

    return df_copy



def check_missing_values(df):

    return df.isna().sum().loc[lambda x: x > 0].sort_values(ascending=False)


def merge_columns(df_with_outliers, df_without_outliers, merge, first_element, second_element):

    # In case we want to merge two or more variables and simplify

    df_with_outliers[merge] = df_with_outliers[first_element] + df_with_outliers[second_element]
    df_without_outliers[merge] = df_without_outliers[first_element] + df_without_outliers[second_element]
    
    return df_with_outliers, df_without_outliers