# DTProfiler

A machine learning pipeline for classifying semantic data types of CSV columns using statistical features and FastText word embeddings.

---

## Building the Classifier from Scratch

### Step 1 — Download Datasets

Refer to [`DATA_SOURCES.md`](DATA_SOURCES.md) for the full list of data sources and [`DATA.md`](DATA.md) for download instructions.

---

### Step 2 — Collect CSV File Paths

Use the helper script to collect all CSV file paths from your dataset directory. The resulting file list is consumed by the data pipeline in Step 3.

```bash
python utils/get_data_filepaths.py --source_dir ../path/to/your/data
```

This writes all discovered file paths to a text file (one path per line) for use in the next step.

---

### Step 3 — Run the Data Pipeline

**Prerequisites:** Download `crawl-300d-2M-subword.vec` from [FastText](https://fasttext.cc/docs/en/english-vectors.html) and place it in the `model/` directory.

Run the pipeline to extract features from all collected CSV files:

```bash
python run_data_pipeline.py data/data_source_files.txt
```

> **Note:** The pipeline currently only supports CSV files that use a comma (`,`) as the delimiter.

---

### Step 4 — Review and Correct Data Type Labels

Open the extracted features file and manually review the assigned data types. Update any labels that are semantically incorrect based on the column content. This step ensures label quality before model training.

---

### Step 5 — Exploratory Data Analysis

Open and run [`data_analysis.ipynb`](data_analysis.ipynb). This end-to-end notebook covers:

1. Load & merge feature data
2. Null value handling
3. Data distributions (labels, characters, lengths, entropy)
4. Correlation matrix
5. PCA on word embeddings
6. Embedding model (FastText OOV handling)
7. Visualizations (neighbors, clustering, score distributions)
8. Data source distribution (VizNet / Kaggle / GitHub / data.gov breakdown)

---

### Step 6 — Prepare Dataset for Training

Open and run [`data_preparation.ipynb`](data_preparation.ipynb). This notebook covers:

1. Load data
2. Null value handling
3. Drop correlated features
4. Apply PCA on word embeddings
5. Split dataset (train / val / test)

---

### Step 7 — Train and Evaluate Models

Open and run [`model_pipeline.ipynb`](model_pipeline.ipynb). This notebook covers:

1. Load libraries
2. Set up training pipelines for Tree and Neural Network models
3. Load data
4. Initialize models
5. Train models
6. Compare results

---

## Project Structure

```
├── logs/                        # Pipeline run logs
├── model/                       # Model definitions and FastText vectors
│   ├── col_name_neural_network.py
│   ├── modular_neural_network_model.py
│   ├── neural_network_model.py
│   ├── stats_neural_network.py
│   └── tree_model.py
├── utils/                       # Pipeline utilities
│   ├── backoff_embedding_generator.py
│   ├── data_pipeline.py
│   ├── get_data_filepaths.py
│   ├── regex_data_type_detector.py
│   └── statistical_features_extractor.py
├── data_analysis.ipynb          # Step 5: EDA
├── data_preparation.ipynb       # Step 6: Dataset preparation
├── model_pipeline.ipynb         # Step 7: Model training & evaluation
├── run_data_pipeline.py         # Step 3: Feature extraction
├── DATA_SOURCES.md              # Dataset sources and licenses
└── DATA.md                      # Dataset reproduction instructions
```

## Dataset License

[![CC BY 4.0][cc-by-shield]][cc-by]

The dataset in the `data/` directory is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
