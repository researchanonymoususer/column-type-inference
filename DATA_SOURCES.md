# Data Sources

This document lists all datasets used in this work, grouped by source. Raw data is not redistributed — all datasets must be downloaded directly from their original sources in accordance with their respective licenses.

---

## Kaggle

Requires a free [Kaggle account](https://www.kaggle.com), the [Kaggle API](https://pypi.org/project/kaggle/), and credentials configured at `~/.kaggle/kaggle.json`.

| ID | Dataset | License | Download Command |
|----|---------|---------|-----------------|
| data_source_1 | Transactions Fraud Datasets | Apache 2.0 | `kaggle datasets download -d computingvictor/transactions-fraud-datasets` |
| data_source_2 | Lending Club Loan Data | DbCL 1.0 | `kaggle datasets download -d adarshsng/lending-club-loan-data-csv` |
| data_source_3 | Finance Loan Approval Prediction | CC0 1.0 | `kaggle datasets download -d krishnaraj30/finance-loan-approval-prediction-data` |
| data_source_4 | Lending Club | CC0 1.0 | `kaggle datasets download -d wordsforthewise/lending-club` |
| data_source_5 | Loan Data | DbCL 1.0 | `kaggle datasets download -d itssuru/loan-data` |
| data_source_6 | Loan Data Sampled | CC0 1.0 | `kaggle datasets download -d savasy/loan-data-sampled` |
| data_source_7 | Bank Dataset | DbCL 1.0 | `kaggle datasets download -d kirankarri/kiran101995-bank` |
| data_source_8 | Loan Status | CC0 1.0 | `kaggle datasets download -d vikrantthakur14/loan-status` |
| data_source_9 | Bank Loan | CC0 1.0 | `kaggle datasets download -d kylangentri/bankloan` |
| data_source_10 | Airbnb Open Data | ODbL 1.0 | `kaggle datasets download -d arianazmoudeh/airbnbopendata` |
| data_source_11 | Online Shop Business | CC0 1.0 | `kaggle datasets download -d gabrielramos87/an-online-shop-business` |
| data_source_12 | Bank Transaction Dataset for Fraud Detection | Apache 2.0 | `kaggle datasets download -d valakhorasani/bank-transaction-dataset-for-fraud-detection` |
| data_source_13 | Fraud Transaction Detection | DbCL 1.0 | `kaggle datasets download -d sanskar457/fraud-transaction-detection` |
| data_source_14 | Transaction Data | CC0 1.0 | `kaggle datasets download -d vipin20/transaction-data` |
| data_source_15 | Transaction | MIT | `kaggle datasets download -d smeyra/transaction` |
| data_source_16 | Bank Transaction Data | CC0 1.0 | `kaggle datasets download -d apoorvwatsky/bank-transaction-data` |
| data_source_17 | Transaction Dashboard | CC BY-SA 4.0 | `kaggle datasets download -d saidaminsaidaxmadov/transaction-dashboard` |
| data_source_18 | Loan Application and Transaction Fraud Detection | CC0 1.0 | `kaggle datasets download -d prajwaldongre/loan-application-and-transaction-fraud-detection` |
| data_source_19 | Synthetic Transaction Monitoring Dataset AML | CC BY-NC-SA 4.0 | `kaggle datasets download -d berkanoztas/synthetic-transaction-monitoring-dataset-aml` |
| data_source_20 | Online Payments Fraud Detection | CC BY-NC-SA 4.0 | `kaggle datasets download -d rupakroy/online-payments-fraud-detection-dataset` |
| data_source_21 | Credit Card Spendings | Apache 2.0 | `kaggle datasets download -d ayushchandramaurya/credit-card-spendings` |
| data_source_22 | Transaction Data for Banking Operations | CC0 1.0 | `kaggle datasets download -d ziya07/transaction-data-for-banking-operations` |
| data_source_23 | Fake Bills | CC0 1.0 | `kaggle datasets download -d alexandrepetit881234/fake-bills` |
| data_source_24 | Auto Market Dataset | CC BY-NC 4.0 | `kaggle datasets download -d qubdidata/auto-market-dataset` |
| data_source_25 | Users vs Bots Classification | MIT | `kaggle datasets download -d juice0lover/users-vs-bots-classification` |
| data_source_26 | Airbnb Price Determinants in Europe | CC0 1.0 | `kaggle datasets download -d thedevastator/airbnb-price-determinants-in-europe` |

---

## data.gov

| ID | Dataset | License | URL |
|----|---------|---------|-----|
| data_source_27 | Credit Unions | CC BY | https://catalog.data.gov/dataset/credit-unions |
| data_source_28 | Iowa Credit Union Liabilities, Shares & Equity | CC BY | https://catalog.data.gov/dataset/iowa-credit-union-liabilities-shares-equity |
| data_source_29 | Iowa Credit Union Assets | CC BY | https://catalog.data.gov/dataset/iowa-credit-union-assets |
| data_source_30 | County Expenditures | CC BY | https://catalog.data.gov/dataset/county-expenditures-1c0b6 |
| data_source_31 | Campaign Finance State Filer Data | PDDL 1.0 | https://catalog.data.gov/dataset/campaign-finance-state-filer-data |
| data_source_32 | County Revenues | CC BY | https://catalog.data.gov/dataset/county-revenues-581ae |
| data_source_33 | Purchase Card Transactions | CC BY | https://catalog.data.gov/dataset/purchase-card-transactions |

---

## World Bank

| ID | Dataset | License | URL |
|----|---------|---------|-----|
| data_source_34 | IBRD Statement of Loans and Guarantees | CC BY 4.0 | https://financesone.worldbank.org/ibrd-statement-of-loans-and-guarantees-latest-available-snapshot/DS00047 |

Attribution required — cite as: World Bank, IBRD Statement of Loans and Guarantees, CC BY 4.0.

---

## GitTables

**License:** CC BY 4.0 (corpus); individual tables inherit their source repository license. Only tables from repositories with the following licenses were retained: MIT, Apache 2.0, GPL 2.0, GPL 3.0, LGPL 2.1, LGPL 3.0, BSD 2-Clause, BSD 3-Clause, CC BY 4.0, CC BY-SA 4.0, CC0 1.0, ISC, MPL 2.0, EUPL 1.2, AGPL 3.0.

Source: [https://github.com/madelonhulsebos/gittables](https://github.com/madelonhulsebos/gittables)

> **Note:** GitHub Search API results are non-deterministic and will vary between runs. An exact reproduction of the original scrape is not guaranteed.

---

## VizNet

Accessed via the [MatchBench dataset](https://huggingface.co/datasets/RUC-DataLab/MatchBench-Dataset) hosted on HuggingFace. Used strictly for research purposes and not redistributed.

Please cite the original paper if you use VizNet data:

```bibtex
@inproceedings{2019-viznet,
  title     = {{VizNet: Towards A Large-Scale Visualization Learning and Benchmarking Repository}},
  author    = {Kevin Hu and Snehalkumar {``Neil''} S. Gaikwad and Madelon Hulsebos and Michiel Bakker and
               Emanuel Zgraggen and C{\'e}sar Hidalgo and Tim Kraska and Guoliang Li and
               Arvind Satyanarayan and {\c{C}}a{\u{g}}atay Demiralp},
  booktitle = {ACM Human Factors in Computing Systems (CHI)},
  year      = {2019},
  doi       = {10.1145/3290605.3300892}
}
```

---
