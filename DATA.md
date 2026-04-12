# DATA.md

This file provides instructions for reproducing the dataset used in this work.
We do not redistribute raw data. All datasets must be downloaded directly from
their original sources in accordance with their respective licenses.

---

## Requirements

For Kaggle datasets, you will need:
- A free Kaggle account: https://www.kaggle.com
- The Kaggle API installed: `pip install kaggle`
- Your API credentials configured at `~/.kaggle/kaggle.json`
  (download from https://www.kaggle.com/settings > API > Create New Token)

---

## 1. Kaggle Datasets

Run the following commands to download all Kaggle datasets:

```bash
# Create a directory for raw data
mkdir -p data/raw && cd data/raw

# data_source_1 — Transactions Fraud Datasets (Apache 2.0)
kaggle datasets download -d computingvictor/transactions-fraud-datasets

# data_source_2 — Lending Club Loan Data (DbCL 1.0)
kaggle datasets download -d adarshsng/lending-club-loan-data-csv

# data_source_3 — Finance Loan Approval Prediction (CC0 1.0)
kaggle datasets download -d krishnaraj30/finance-loan-approval-prediction-data

# data_source_4 — Lending Club (CC0 1.0)
kaggle datasets download -d wordsforthewise/lending-club

# data_source_5 — Loan Data (DbCL 1.0)
kaggle datasets download -d itssuru/loan-data

# data_source_6 — Loan Data Sampled (CC0 1.0)
kaggle datasets download -d savasy/loan-data-sampled

# data_source_7 — Bank Dataset (DbCL 1.0)
kaggle datasets download -d kirankarri/kiran101995-bank

# data_source_8 — Loan Status (CC0 1.0)
kaggle datasets download -d vikrantthakur14/loan-status

# data_source_9 — Bank Loan (CC0 1.0)
kaggle datasets download -d kylangentri/bankloan

# data_source_10 — Airbnb Open Data (ODbL 1.0)
kaggle datasets download -d arianazmoudeh/airbnbopendata

# data_source_11 — Online Shop Business (CC0 1.0)
kaggle datasets download -d gabrielramos87/an-online-shop-business

# data_source_12 — Bank Transaction Dataset for Fraud Detection (Apache 2.0)
kaggle datasets download -d valakhorasani/bank-transaction-dataset-for-fraud-detection

# data_source_13 — Fraud Transaction Detection (DbCL 1.0)
kaggle datasets download -d sanskar457/fraud-transaction-detection

# data_source_14 — Transaction Data (CC0 1.0)
kaggle datasets download -d vipin20/transaction-data

# data_source_15 — Transaction (MIT)
kaggle datasets download -d smeyra/transaction

# data_source_16 — Bank Transaction Data (CC0 1.0)
kaggle datasets download -d apoorvwatsky/bank-transaction-data

# data_source_17 — Transaction Dashboard (CC BY-SA 4.0)
kaggle datasets download -d saidaminsaidaxmadov/transaction-dashboard

# data_source_18 — Loan Application and Transaction Fraud Detection (CC0 1.0)
kaggle datasets download -d prajwaldongre/loan-application-and-transaction-fraud-detection

# data_source_19 — Synthetic Transaction Monitoring Dataset AML (CC BY-NC-SA 4.0)
kaggle datasets download -d berkanoztas/synthetic-transaction-monitoring-dataset-aml

# data_source_20 — Online Payments Fraud Detection (CC BY-NC-SA 4.0)
kaggle datasets download -d rupakroy/online-payments-fraud-detection-dataset

# data_source_21 — Credit Card Spendings (Apache 2.0)
kaggle datasets download -d ayushchandramaurya/credit-card-spendings

# data_source_22 — Transaction Data for Banking Operations (CC0 1.0)
kaggle datasets download -d ziya07/transaction-data-for-banking-operations

# data_source_23 — Fake Bills (CC0 1.0)
kaggle datasets download -d alexandrepetit881234/fake-bills

# data_source_24 — Auto Market Dataset (CC BY-NC 4.0)
kaggle datasets download -d qubdidata/auto-market-dataset

# data_source_25 — Users vs Bots Classification (MIT)
kaggle datasets download -d juice0lover/users-vs-bots-classification

# data_source_26 — Airbnb Price Determinants in Europe (CC0 1.0)
kaggle datasets download -d thedevastator/airbnb-price-determinants-in-europe

# Unzip all downloaded files
for f in *.zip; do unzip -q "$f" -d "${f%.zip}"; done
```

---

## 2. data.gov Datasets (Public Domain / CC BY)

Download directly from the links below. Each dataset has a CSV download
button on its catalog page, or use the direct API links provided.

| ID             | Dataset | License | URL |
|----------------|---------|---------|-----|
| data_source_27 | Credit Unions | CC BY | https://catalog.data.gov/dataset/credit-unions |
| data_source_28 | Iowa Credit Union Liabilities, Shares & Equity | CC BY | https://catalog.data.gov/dataset/iowa-credit-union-liabilities-shares-equity |
| data_source_29 | Iowa Credit Union Assets | CC BY | https://catalog.data.gov/dataset/iowa-credit-union-assets |
| data_source_30 | County Expenditures | CC BY | https://catalog.data.gov/dataset/county-expenditures-1c0b6 |
| data_source_31 | Campaign Finance State Filer Data | PDDL 1.0 | https://catalog.data.gov/dataset/campaign-finance-state-filer-data |
| data_source_32 | County Revenues | CC BY | https://catalog.data.gov/dataset/county-revenues-581ae |
| data_source_33 | Purchase Card Transactions | CC BY | https://catalog.data.gov/dataset/purchase-card-transactions |

To download programmatically:

```bash
mkdir -p data/raw/datagov

# Example using curl — repeat for each dataset
# Find the direct CSV download URL on each catalog page and substitute below
curl -L "https://data.iowa.gov/api/views/vpuv-c5ha/rows.csv?accessType=DOWNLOAD" \
     -o data/raw/datagov/iowa_credit_union_income_expenses.csv
```

---

## 3. World Bank Dataset

| ID             | Dataset | License | URL |
|----------------|---------|---------|-----|
| data_source_34 | IBRD Statement of Loans and Guarantees | CC BY 4.0 | https://financesone.worldbank.org/ibrd-statement-of-loans-and-guarantees-latest-available-snapshot/DS00047 |

Download the CSV directly from the World Bank Finance One portal at the
link above. Attribution required — cite as: World Bank, IBRD Statement
of Loans and Guarantees, CC BY 4.0.

---

## 4. GitTables

GitTables data was collected using the official extraction pipeline.

**License:** CC BY 4.0 (corpus); individual tables inherit their source
repository license. Only tables from permissive-licensed repositories
(MIT, Apache 2.0, CC0, BSD, CC BY) were retained.

```bash
# Install dependencies
pip install gittables pyarrow fasttext

# Clone the official repository
git clone https://github.com/madelonhulsebos/gittables
cd gittables

# Install the package
pip install .
pip install -r requirements.txt
```
Add your GitHub credentials to `settings.toml`
```
   github_username = "your_username"
   github_token    = "your_token"
```

Modify the `file_extraction.py` script by setting the custom_topics argument of the set_topics method to 

```
["account,amount,bank,created_at,customer,datetime,debit,due_date,member,timestamp,updated_at"]
```

> **Note:** GitHub Search API results are non-deterministic and will vary
> between runs. An exact reproduction of our scrape is not guaranteed.
> We provide our keyword list and filter script to enable best-effort
> reproduction.

---

## 5. VizNet

VizNet data was accessed via the MatchBench dataset hosted on HuggingFace.
This data is used strictly for research purposes and is not redistributed.

```bash
# Requires git-lfs
git lfs install

# Clone the MatchBench dataset
git clone https://huggingface.co/datasets/RUC-DataLab/MatchBench-Dataset

# Extract VizNet
cd MatchBench-Dataset
tar -xzf Viznet.tar.gz
```

If you use VizNet data, please cite the original paper:

```bibtex
@inproceedings{2019-viznet,
  title     = {{VizNet: Towards A Large-Scale Visualization Learning
                and Benchmarking Repository}},
  author    = {Kevin Hu and Snehalkumar {``Neil''} S. Gaikwad and
               Madelon Hulsebos and Michiel Bakker and Emanuel Zgraggen
               and C{\'e}sar Hidalgo and Tim Kraska and Guoliang Li and
               Arvind Satyanarayan and {\c{C}}a{\u{g}}atay Demiralp},
  booktitle = {ACM Human Factors in Computing Systems (CHI)},
  year      = {2019},
  doi       = {10.1145/3290605.3300892}
}
```
---

## License Notes

Datasets carry different licenses. The following sources have restrictions
that prevent redistribution of derived data — use for research only:

| Source         | License | Restriction |
|----------------|---------|-------------|
| data_source_17 | CC BY-SA 4.0 | Derived works must carry same license |
| data_source_19 | CC BY-NC-SA 4.0 | Non-commercial use only |
| data_source_20 | CC BY-NC-SA 4.0 | Non-commercial use only |
| data_source_24 | CC BY-NC 4.0 | Non-commercial use only |
