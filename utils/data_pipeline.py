import os
import re
import csv
import logging
import numpy as np
import pandas as pd
from wordsegment import load, segment

from utils.regex_data_type_detector import RegexDataTypeDetector
from utils.statistical_features_extractor import StatisticalFeatureExtractor


class DataPipeline:
    def __init__(self, source_path, destination_path, logger, embedder=None):
        self.source_path = source_path
        self.destination_path = destination_path
        self.features = pd.DataFrame()
        self.embedder = embedder
        self.data_type_detector = RegexDataTypeDetector()
        load()
        self.data = None
        self.stat_feature_extractor = StatisticalFeatureExtractor()
        self.logger = logger

    def generate_embeddings(self, column_name):
        """Generate embeddings for column names"""
        max_words_after_segment = 8
        # remove special characters and extra white spaces from column name
        cleaned_column_name = re.sub(r'[^a-zA-Z\s]', ' ', column_name)
        cleaned_column_name = re.sub(r'\s+', ' ', cleaned_column_name)
        cleaned_column_name = cleaned_column_name.lower()

        # segment column name
        if ' ' in cleaned_column_name:
            segmented_column_name = cleaned_column_name.split(' ')
        else:
            segmented_column_name = segment(cleaned_column_name)

        # remove empty strings
        segmented_column_name = [w for w in segmented_column_name if w]

        # get embeddings
        embeddings = [self.embedder.get_embedding(word, top_k=5, verbose=True)
                      for word in reversed(segmented_column_name[-max_words_after_segment:])]

        # add padding
        rows_to_add = max_words_after_segment - len(embeddings)
        padded_embeddings = np.pad(embeddings,
                                   pad_width=((0, rows_to_add), (0, 0)),
                                   mode='constant',
                                   constant_values=0)

        return segmented_column_name, padded_embeddings

    def extract(self):
        """
        Extract features from the CSV file
        """
        self.logger.info(f"Extracting data from: {self.source_path}")
        try:
            # Read data as strings for pattern matching
            self.data = pd.read_csv(self.source_path, header=0, dtype=str)
            self.pd_data = pd.read_csv(self.source_path, header=0)

            self.logger.info(f"Loaded {len(self.data)} rows and {len(self.data.columns)} columns")
            # self.logger.info(f"First few rows:\n{self.data.head()}")

            self.logger.info("Data extracted successfully.")
        except FileNotFoundError:
            self.logger.error(f"Source file not found: {self.source_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error during data extraction: {e}")
            raise

    def transform(self):
        """
        Transform data and extract features
        """
        self.logger.info("Transforming data.")
        if self.data is None or self.data.empty:
            self.logger.warning("No data to transform. Run extract() first.")
            return

        all_rows = []
        for column in self.data.columns:
            # Auto-detect data type
            label = self.data_type_detector.detect_column_type(self.data[column])
            self.logger.info(f"Auto-detected type for column '{column}': {label}")

            if label == 'no_data':
                continue

            col_features_dict = {
                'ColumnName': column,
                'Filepath': self.source_path,
                'DataType': label,
                'PandasDataType': self.pd_data[column].dtype
            }

            self.logger.info(f"Extracting statistical features from column: {column} of type {label}")
            statistical_features = self.stat_feature_extractor.extract_statistical_features_from_column(
                self.data[column], label, generate_plot=False)
            col_features_dict.update(statistical_features)

            # Optionally generate embeddings if embedder is provided
            if self.embedder is not None:
                self.logger.info(f"Generating embeddings for column name: {column}")
                segmented_column_name, column_name_embeddings = self.generate_embeddings(column)
                col_features_dict['column_name_segments'] = " ".join(segmented_column_name)
                col_features_dict['num_of_segments'] = len(segmented_column_name)

                n_words, n_dims = column_name_embeddings.shape
                flattened = column_name_embeddings.flatten()
                prefix = 'emb'

                # Create column names: emb_w1_d1, emb_w1_d2, ..., emb_w8_d100
                column_names = [
                    f"{prefix}_w{word_idx + 1}_d{dim_idx + 1}"
                    for word_idx in range(n_words)
                    for dim_idx in range(n_dims)
                ]

                # Add columns to DataFrame
                for col_name, value in zip(column_names, flattened):
                    col_features_dict[col_name] = value

            col_features_df = pd.DataFrame([col_features_dict])
            all_rows.append(col_features_df)  # ← collect

        if all_rows:
            self.features = pd.concat(all_rows, ignore_index=True).drop_duplicates()

        self.logger.info(f"Data transformed successfully. Extracted features for {len(self.features)} columns.")

    def load(self):
        """
        Save extracted features to a CSV file
        """
        self.logger.info(f"Loading data to: {self.destination_path}")
        if self.features is None or self.features.empty:
            self.logger.warning("No data to load. Run transform() first.")
            return

        try:
            if not os.path.exists(self.destination_path):
                # If the file doesn't exist, write the DataFrame with headers
                self.features.to_csv(self.destination_path, index=False, header=True)
                self.logger.info(f"Created new file: {self.destination_path}")
            else:
                # If the file exists, append the DataFrame without headers
                self.features.to_csv(self.destination_path, mode='a', index=False, header=False)
                self.logger.info(f"Appended to existing file: {self.destination_path}")
        except Exception as e:
            self.logger.error(f"Error during data loading: {e}")
            raise

    def run_pipeline(self):
        """
        Executes the entire data pipeline.
        """
        self.logger.info("Starting data pipeline execution.")
        try:
            self.extract()
            self.transform()
            self.load()
            self.logger.info("Data pipeline executed successfully.")
        except Exception as e:
            self.logger.critical(f"Data pipeline failed: {e}")
            raise
