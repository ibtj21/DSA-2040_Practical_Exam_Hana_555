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

DSA_2040_Practical_Exam_Hana_555/
│
├─ `etl_retail.py`          - ETL implementation for retail DW  
├─ `retail_dw.db`           - SQLite database (fact & dimension tables)  
├─ `preprocessing_iris.py`  - Iris preprocessing, exploration  
├─ `clustering_iris.py`     - K-Means clustering implementation  
├─ `classification_iris.py` - Classification and rule mining  
├─ `sql_queries.sql`        - OLAP and other SQL queries  
├─ `images/`                - Visualizations and schema diagrams  
│   ├─ `schema.png`  
│   ├─ `sales_by_country.png`  
│   ├─ `pairplot_iris.png`  
│   └─ `clusters.png`  
└─ `README.md`


# Data Warehousing

## 1.1 Design of a Star Schema for a Retail Company

**Scenario:** Designing a data warehouse for a retail company that sells products across categories. The company tracks sales, customers, products, and time.  

**Requirements:**  
- Support queries like total sales by product category per quarter, customer demographics analysis, and inventory trends.

### 1.1.1 Star Schema Design

**Fact Table:** `Sales`  
- Columns: `SalesKey` (PRIMARY KEY), `Quantity`, `TotalSales`  
- Foreign Keys: `CustomerKey`, `ProductKey`, `TimeKey`  

**Dimension Tables:**  
- `TimeDim`: Columns: `TimeKey`, `InvoiceDate`, `Year`, `Quarter`, `Month`  
- `CustomerDim`: Columns: `CustomerKey`, `CustomerID`, `CustomerName`, `Country`  
- `ProductDim`: Columns: `ProductKey`, `StockCode`, `Description`, `Category`, `UnitPrice`  

**Schema Diagram:**  
![Star Schema Diagram](path/to/your/image.png) <!-- Replace with actual image path -->

### 1.1.2 Explanation for Choosing Star Schema Over Snowflake

The star schema was chosen because it simplifies queries and improves query performance by denormalizing dimension tables. It is easier for business analysts to understand and use for reporting, as all relevant dimension attributes are in single tables.  

### 1.1.3 SQL CREATE TABLE Statements

The SQL `CREATE TABLE` statements for the fact and dimension tables (assuming SQLite syntax) can be found in:  
`path/to/your/sql_script.sql` <!-- Replace with actual file path -->

   
   
   
