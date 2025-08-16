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
![Star Schema Diagram](Section_1/Task_1_Data_Warehouse_Design/Schema_diagram.png)

### 1.1.2 Explanation for Choosing Star Schema Over Snowflake

The star schema was chosen because it simplifies queries and improves query performance by denormalizing dimension tables. It is easier for business analysts to understand and use for reporting, as all relevant dimension attributes are in single tables.  

### 1.1.3 SQL CREATE TABLE Statements

The SQL `CREATE TABLE` statements for the fact and dimension tables (assuming SQLite syntax) can be found in:  
[Schema_retail.sql](Section_1/Task_1_Data_Warehouse_Design/Schema_retail.sql)


## 1.2 ETL Process Implementation

**Dataset:** Synthetic data that mimics the structure and scale of the described dataset, with similar columns like `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, and `Country`. The dataset contains around 500–1000 rows for practicality, generated using `pandas`, `faker`, and random values:  

- `Quantity`: 1–50  
- `UnitPrice`: 1–100  
- `InvoiceDate`: dates over 2 years  
- 100 unique customers  
- 5–10 countries  

Additionally, missing values, column categories, and outliers in `Quantity` and `UnitPrice` were generated for practicality. A seed was used for reproducibility.  

**Generation code for the synthetic data can be found at:** `(put placeholder for path)`

### 1.2.1 Extract
- Python was used to generate the data in a DataFrame.  
- Handled missing values in columns like `Description` and `Country`.  
- Corrected data types, e.g., `InvoiceDate`.  

### 1.2.2 Transform
- Calculated a new column: `TotalSales = Quantity * UnitPrice`.  
- Filtered data for sales in the last year (assuming current date as August 12, 2025).  
- Handled outliers: removed rows where `Quantity < 0` or `UnitPrice <= 0`.  

### 1.2.3 Load
- Used `sqlite3` in Python to create a database file (path placeholder).  
- Loaded the transformed data into a **fact table** (`SalesFact`) and **three dimension tables** (`ProductDim`, `CustomerDim`, `TimeDim`).  

A function to perform the full ETL and log the number of rows processed at each stage was implemented:

```python
# ===== ETL Process: Export Only .db, Transformed Data, Synthetic Data =====
def run_etl_export_only():
    try:
        logging.info("Starting ETL process...")

        # === Extract (Synthetic Data) ===
        df_synthetic = generate_base_data(NUM_ROWS)
        df_synthetic = inject_missing_values(df_synthetic)
        df_synthetic = inject_outliers(df_synthetic)
        customer_names = generate_customer_names(df_synthetic['CustomerID'].nunique())
        df_synthetic = assign_customer_names(df_synthetic, customer_names)

        # Handle missing values & convert data types
        df_synthetic['Description'] = df_synthetic['Description'].fillna('Unknown Product')
        df_synthetic['Country'] = df_synthetic['Country'].fillna('Unknown Country')
        df_synthetic['InvoiceDate'] = pd.to_datetime(df_synthetic['InvoiceDate'], errors='coerce')

        # Save synthetic data
        df_synthetic.to_csv("synthetic_retail_dataset.csv", index=False)
        logging.info("Synthetic dataset exported as 'synthetic_retail_dataset.csv'")
        logging.info(f"Rows after extraction: {len(df_synthetic)}")

        # === Transform (Cleaned and Filtered Data) ===
        df_transformed = clean_and_convert(df_synthetic.copy())
        df_transformed['TotalSales'] = df_transformed['Quantity'] * df_transformed['UnitPrice']
        df_transformed = df_transformed[df_transformed['Quantity'] > 0]
        df_transformed = df_transformed[df_transformed['UnitPrice'] > 0]

        # Filter last 12 months
        current_date = pd.Timestamp("2025-08-12")
        one_year_ago = current_date - pd.DateOffset(years=1)
        df_transformed = df_transformed[(df_transformed['InvoiceDate'] >= one_year_ago) & 
                                        (df_transformed['InvoiceDate'] <= current_date)]

        # Save transformed data
        df_transformed.to_csv("transformed_retail_dataset.csv", index=False)
        logging.info("Transformed dataset exported as 'transformed_retail_dataset.csv'")
        logging.info(f"Rows after transformation: {len(df_transformed)}")   
        
        # === Load ===
        logging.info("Loading data into SQLite database using external schema...")

        conn = sqlite3.connect("retail.db")
        cur = conn.cursor()

        # Read schema from external SQL file
        with open("Schema2.sql", "r") as f:
            schema_sql = f.read()
        cur.executescript(schema_sql)

        # Insert into dimension tables
        customer_dim = df_transformed[['CustomerID', 'CustomerName', 'Country']].drop_duplicates()
        customer_dim.to_sql('CustomerDim', conn, if_exists='append', index=False)

        product_dim = df_transformed[['StockCode', 'Description', 'Category']].drop_duplicates()
        product_dim.to_sql('ProductDim', conn, if_exists='append', index=False)

        # Merge keys for fact table
        cust_keys = pd.read_sql("SELECT CustomerKey, CustomerID FROM CustomerDim", conn)
        prod_keys = pd.read_sql("SELECT ProductKey, StockCode FROM ProductDim", conn)
        fact_df = df_transformed.merge(cust_keys, on='CustomerID').merge(prod_keys, on='StockCode')
        fact_df = fact_df[['InvoiceNo', 'InvoiceDate', 'Quantity', 'UnitPrice', 'CustomerKey', 'ProductKey']]
        fact_df.to_sql('SalesFact', conn, if_exists='append', index=False)

        conn.commit()
        conn.close()
        logging.info("Data loaded successfully into SQLite database.")
        logging.info("ETL process completed: only synthetic, transformed, and .db exported.")

    except Exception as e:
        logging.error(f"ETL process failed: {e}")

# ===== Run ETL =====
if __name__ == "__main__":
    run_etl_export_only()

## Key Features

- Modular function applicable to any dataset.
- Performs full ETL by calling a single function: `run_etl_export_only()`.
- Logs the number of rows processed at each stage.
- Includes error handling.

### Exports

- **Synthetic dataset:** (put placeholder for path)  
- **Transformed dataset:** (put placeholder for path)  
- **SQLite database:** (put placeholder for path)  

### Sample Output

2025-08-14 23:46:38,674 - INFO - Starting ETL process...
2025-08-14 23:46:38,730 - INFO - Synthetic dataset exported as 'synthetic_retail_dataset.csv'
2025-08-14 23:46:38,731 - INFO - Rows after extraction: 1000
2025-08-14 23:46:38,748 - INFO - Transformed dataset exported as 'transformed_retail_dataset.csv'
2025-08-14 23:46:38,750 - INFO - Rows after transformation: 493
2025-08-14 23:46:38,751 - INFO - Loading data into SQLite database using external schema...
2025-08-14 23:46:38,994 - INFO - Data loaded successfully into SQLite database.
2025-08-14 23:46:38,996 - INFO - ETL process completed: only synthetic, transformed, and .db exported.

### Post-load

- **Fact and dimension tables** are available at: (put placeholder for path)  
- **To explore the full ETL process**, visit: (put placeholder for path)


