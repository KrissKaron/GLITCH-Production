import sys
import pandas as pd
from textblob import TextBlob
from __path__ import *

class CSVExcavator:
    def __init__(self, path_root):
        self.PATH_GENERAL_NEWS      = PATH_GENERAL_NEWS
        self.PATH_INTEREST_RATE     = PATH_INTEREST_RATE
        self.PATH_REGULATORY_NEWS   = PATH_REGULATORY_NEWS
        self.PATH_WHALES_100K       = PATH_WHALES_100K
        self.PATH_WHALES_1M         = PATH_WHALES_1M

    def _read_csv(self, file_path, columns):
        """
        Internal method to read a CSV file and extract specified columns.
        
        Args:
        - file_path (str): Path to the CSV file.
        - columns (list): List of columns to extract.

        Returns:
        - pd.DataFrame or None: DataFrame with selected columns, or None if an error occurs.
        """
        try:
            df = pd.read_csv(file_path, usecols=columns)
            return df
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
        except ValueError as e:
            print(f"Error in reading columns from {file_path}: {e}")
            return None

    def extract_general_news(self):
        """Extract 'title', 'description', 'content', and 'publishedAt' from general news CSV."""
        return self._read_csv(self.PATH_GENERAL_NEWS, ['title', 'description', 'content', 'publishedAt'])

    def extract_interest_rate_news(self):
        """Extract 'title', 'description', and 'published_at' from interest rate CSV."""
        return self._read_csv(self.PATH_INTEREST_RATE, ['title', 'description', 'published_at'])

    def extract_regulatory_news(self):
        """Extract 'title', 'description', and 'published_at' from regulatory news CSV."""
        return self._read_csv(self.PATH_REGULATORY_NEWS, ['title', 'description', 'published_at'])

    def extract_whale_transactions(self):
        """
        Extract 'datetime' and 'value' from both whale transaction CSVs and combine them.

        Returns:
        - pd.DataFrame: Combined DataFrame with selected columns.
        """
        df_100k = self._read_csv(self.PATH_WHALES_100K, ['datetime', 'value'])
        df_1m = self._read_csv(self.PATH_WHALES_1M, ['datetime', 'value'])

        if df_100k is not None and df_1m is not None:
            df_combined = pd.concat([df_100k, df_1m], ignore_index=True)
            df_combined.sort_values(by='datetime', inplace=True)
            return df_combined
        elif df_100k is not None:
            return df_100k
        elif df_1m is not None:
            return df_1m
        else:
            print("No whale transaction data found.")
            return None

class NewsScorer:
    def __init__(self):
        pass

    def _calculate_importance_score_text(self, text):
        """
        Calculate an importance score using TextBlob sentiment polarity for textual data.

        Args:
        - text (str): The input text to analyze.

        Returns:
        - float: Importance score scaled from 0 to 100.
        """
        if pd.isna(text) or text.strip() == "":
            return 50.0  # Neutral importance if no content

        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity  # Polarity ranges from -1 to 1
        importance_score = (sentiment_score + 1) * 50  # Scale to 0-100
        return round(importance_score, 2)

    def _calculate_importance_score_numeric(self, df, column_name):
        """
        Normalize a numeric column and scale values to 0-100.

        Args:
        - df (pd.DataFrame): The DataFrame containing numeric data.
        - column_name (str): The column to normalize.

        Returns:
        - pd.Series: Importance scores scaled from 0 to 100.
        """
        if df.empty:
            print("Empty DataFrame provided.")
            return pd.Series([])

        if column_name not in df.columns:
            raise KeyError(f"Column '{column_name}' not found in the DataFrame.")

        min_value = df[column_name].min()
        max_value = df[column_name].max()

        if min_value == max_value:
            return pd.Series([50.0] * len(df))  # Assign neutral score if values are identical

        importance_scores = ((df[column_name] - min_value) / (max_value - min_value)) * 100
        return importance_scores.round(2)

    def score_general_news(self, df):
        """
        Assign importance scores to the general news DataFrame based on 'title', 'description', and 'content'.

        Args:
        - df (pd.DataFrame): DataFrame containing 'title', 'description', 'content', and 'publishedAt'.

        Returns:
        - pd.DataFrame: Updated DataFrame with an 'importance_score' column.
        """
        df['importance_score'] = df[['title', 'description', 'content']].astype(str).agg(' '.join, axis=1).apply(self._calculate_importance_score_text)
        return df

    def score_interest_rate_news(self, df):
        """
        Assign importance scores to the interest rate news DataFrame based on 'title' and 'description'.

        Args:
        - df (pd.DataFrame): DataFrame containing 'title', 'description', and 'published_at'.

        Returns:
        - pd.DataFrame: Updated DataFrame with an 'importance_score' column.
        """
        df['importance_score'] = df[['title', 'description']].astype(str).agg(' '.join, axis=1).apply(self._calculate_importance_score_text)
        return df

    def score_regulatory_news(self, df):
        """
        Assign importance scores to the regulatory news DataFrame based on 'title' and 'description'.

        Args:
        - df (pd.DataFrame): DataFrame containing 'title', 'description', and 'published_at'.

        Returns:
        - pd.DataFrame: Updated DataFrame with an 'importance_score' column.
        """
        df['importance_score'] = df[['title', 'description']].astype(str).agg(' '.join, axis=1).apply(self._calculate_importance_score_text)
        return df

    def score_whale_transactions(self, df):
        """
        Assign importance scores to the whale transactions DataFrame based on 'value'.

        Args:
        - df (pd.DataFrame): DataFrame containing 'datetime' and 'value'.

        Returns:
        - pd.DataFrame: Updated DataFrame with an 'importance_score' column.
        """
        df['importance_score'] = self._calculate_importance_score_numeric(df, 'value')
        return df
    
class CSVCompacter:
    def __init__(self, general_news_df, interest_rate_df, regulatory_news_df, whale_transactions_df):
        """
        Initialize the CSVCompacter with required DataFrames.

        Args:
        - general_news_df (pd.DataFrame): DataFrame containing general news data.
        - interest_rate_df (pd.DataFrame): DataFrame containing interest rate news data.
        - regulatory_news_df (pd.DataFrame): DataFrame containing regulatory news data.
        - whale_transactions_df (pd.DataFrame): DataFrame containing whale transaction data.
        """
        self.general_news_df        = general_news_df
        self.interest_rate_df       = interest_rate_df
        self.regulatory_news_df     = regulatory_news_df
        self.whale_transactions_df  = whale_transactions_df
        self.output_path            = PATH_NEWS

    def standardize_columns(self):
        """
        Standardize column names across all DataFrames to allow concatenation.
        """

        # Rename columns to common structure
        self.general_news_df.rename(columns={'publishedAt': 'published_at'}, inplace=True)
        self.interest_rate_df.rename(columns={'published_at': 'published_at'}, inplace=True)
        self.regulatory_news_df.rename(columns={'published_at': 'published_at'}, inplace=True)
        self.whale_transactions_df.rename(columns={'datetime': 'published_at'}, inplace=True)

        # Check if 'value' column exists before processing
        if 'value' in self.whale_transactions_df.columns:
            # Process whale transactions: move 'value' to 'title' and adjust importance_score
            self.whale_transactions_df['title'] = self.whale_transactions_df['value'].astype(str)

            # Determine if value should adjust importance_score (100k or 1m transactions)
            self.whale_transactions_df['importance_score'] = self.whale_transactions_df.apply(
                lambda row: row['importance_score'] / 10 if len(str(int(row['value']))) == 4 else row['importance_score'],
                axis=1
            )
            self.whale_transactions_df.drop(columns=['value'], inplace=True)
        else:
            print("Warning: 'value' column not found in whale_transactions_df. Skipping value-related operations.")
        standard_columns = ['published_at', 'title', 'description', 'content', 'importance_score']
        for df in [self.general_news_df, self.interest_rate_df, self.regulatory_news_df, self.whale_transactions_df]:
            for col in standard_columns:
                if col not in df.columns:
                    df[col] = None  # Fill missing columns with None

    def compact_dataframes(self):
        """
        Concatenate all DataFrames into a single DataFrame after standardization.

        Returns:
        - pd.DataFrame: Concatenated DataFrame with all data.
        """
        self.standardize_columns()
        dfs = [self.general_news_df, self.interest_rate_df, self.regulatory_news_df, self.whale_transactions_df]
        compacted_df = pd.concat(dfs, ignore_index=True, sort=False)

        # Convert published_at to datetime format
        compacted_df['published_at'] = pd.to_datetime(compacted_df['published_at'], errors='coerce')
        compacted_df.sort_values(by='published_at', inplace=True)
        compacted_df.reset_index(drop=True, inplace=True)
        return compacted_df

    def save_to_csv(self):
        """
        Save the compacted DataFrame to a CSV file at the specified path.
        """
        compacted_df = self.compact_dataframes()
        compacted_df.to_csv(self.output_path, index=False)
        print(f"Compacted CSV saved successfully to {self.output_path}")