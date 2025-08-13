# DSA-2040_Practical_Exam_Hana_555
This repository contains my DSA 2040 (US 2025) practical exam project, demonstrating data warehousing and data mining using Python and SQL, including star schema design, ETL, OLAP queries, and machine learning classification.

---

## Overview
This repository contains the complete submission for the DSA 2040 end-semester practical exam. It covers two main sections:  

1. **Data Warehousing (50 marks)**  
   - Design of a star schema for a retail company  
   - ETL process implementation using Python and SQLite  
   - OLAP queries and analysis with visualizations  

2. **Data Mining (50 marks)**  
   - Data preprocessing and exploration of the Iris dataset  
   - Clustering using K-Means  
   - Classification (Decision Tree & KNN) and Association Rule Mining  

---

## Datasets
- **Retail Dataset:** Generated synthetic data (~1000 rows) resembling the UCI Online Retail dataset.  
- **Iris Dataset:** Loaded from scikit-learn.  
- **Synthetic Transactional Data:** Created for association rule mining.  

---

## Folder Structure
DSA_2040_Practical_Exam_Hana_555
│
├─ etl_retail.py # ETL implementation for retail DW
├─ retail_dw.db # SQLite database (fact & dimension tables)
├─ preprocessing_iris.py # Iris preprocessing, exploration
├─ clustering_iris.py # K-Means clustering implementation
├─ classification_iris.py # Classification and rule mining
├─ sql_queries.sql # OLAP and other SQL queries
├─ images/ # Visualizations and schema diagrams
│ ├─ schema.png
│ ├─ sales_by_country.png
│ ├─ pairplot_iris.png
│ └─ clusters.png
└─ README.md


## Overview

- **Data Warehousing (50 marks):** Star schema design, ETL, OLAP queries, visualizations.  
- **Data Mining (50 marks):** Preprocessing, clustering, classification, and association rule mining using Iris dataset and synthetic transactional data.  

## How to Run

1. Ensure Python 3.x is installed with required libraries: `pandas`, `numpy`, `scikit-learn`, `sqlite3`, `matplotlib`, `seaborn`, `mlxtend`.  
2. Run ETL: `python etl_retail.py`.  
3. Run Iris preprocessing: `python preprocessing_iris.py`.  
4. Run clustering: `python clustering_iris.py`.  
5. Run classification & association rules: `python classification_iris.py`.  
6. Execute SQL queries via SQLite client: `sql_queries.sql`.  

## Self-Assessment
- **Data Warehousing:** Schema design, ETL, and OLAP queries fully implemented; visualizations included.  
- **Data Mining:** Preprocessing, clustering, classification, and association rules implemented; analysis included.  
- **Notes:** Synthetic data used for retail and transactional datasets; may affect realism but maintains structure and scale.  
