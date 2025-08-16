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


# 1. Data Warehousing

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

**Dataset:** Synthetic data designed to mimic the structure and scale of the target dataset with similar columns:  

| Column       | Description                                      |
|--------------|--------------------------------------------------|
| InvoiceNo    | Unique invoice identifier                        |
| StockCode    | Product code                                     |
| Description  | Product description                              |
| Quantity     | Number of items purchased                        |
| InvoiceDate  | Date of purchase                                 |
| UnitPrice    | Price per item                                   |
| CustomerID   | Unique customer identifier                        |
| Country      | Customer's country                               |

**Dataset Features:**  
- Row count: ~500–1000 (practicality)  
- Quantities: 1–50, Prices: 1–100  
- Dates span 2 years  
- 100 unique customers  
- 5–10 countries  
- Includes missing values, categorical columns, and outliers for Quantity and UnitPrice  
- Seeded for reproducibility  

**Generation code:** [etl_retail.ipynb](Section_1/Task_2_ETL_Process%20_Implementation/etl_retail.ipynb)

---

### 1.2.1 Extract
- Python (pandas & Faker) was used to generate the synthetic dataset as a DataFrame.  
- Missing values handled for `Description` and `Country`.  
- Data types corrected, e.g., `InvoiceDate` converted to datetime.  


```python
# Handle missing values & convert data types
df_synthetic['Description'] = df_synthetic['Description'].fillna('Unknown Product')
df_synthetic['Country'] = df_synthetic['Country'].fillna('Unknown Country')
df_synthetic['InvoiceDate'] = pd.to_datetime(df_synthetic['InvoiceDate'], errors='coerce')

# Save synthetic data
df_synthetic.to_csv("synthetic_retail_dataset.csv", index=False)
logging.info("Synthetic dataset exported as 'synthetic_retail_dataset.csv'")
```

### 1.2.2 Transform

**Transformations Applied:**  
- Added a new column: `TotalSales = Quantity * UnitPrice`  
- Filtered data for sales in the last year (assuming current date = 2025-08-12)  
- Handled outliers by removing rows where `Quantity <= 0` or `UnitPrice <= 0`  


```python
# Calculate total sales
df_transformed['TotalSales'] = df_transformed['Quantity'] * df_transformed['UnitPrice']

# Remove outliers
df_transformed = df_transformed[df_transformed['Quantity'] > 0]
df_transformed = df_transformed[df_transformed['UnitPrice'] > 0]

# Filter for the last 12 months
current_date = pd.Timestamp("2025-08-12")
one_year_ago = current_date - pd.DateOffset(years=1)
df_transformed = df_transformed[
    (df_transformed['InvoiceDate'] >= one_year_ago) & 
    (df_transformed['InvoiceDate'] <= current_date)
]

# Export transformed dataset
df_transformed.to_csv("transformed_retail_dataset.csv", index=False)
logging.info("Transformed dataset exported as 'transformed_retail_dataset.csv'")
```


### 1.2.3 Load

**Loading Process:**  
- Used `sqlite3` in Python to create a database  
- Loaded data into:

  * 1 Fact Table: `SalesFact`  
  * 3 Dimension Tables: `ProductDim`, `CustomerDim`, `TimeDim`  


```python
import sqlite3
import pandas as pd

# Connect to SQLite database
conn = sqlite3.connect("retail.db")
cur = conn.cursor()

# Execute external schema
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

# Load fact table
fact_df.to_sql('SalesFact', conn, if_exists='append', index=False)

# Commit and close connection
conn.commit()
conn.close()
logging.info("Data loaded successfully into SQLite database.")
```



### 1.2.4 Full ETL Function

**Overview:**  
- Modular ETL function that can be applied to any dataset  
- Performs **full ETL** by calling `run_etl_export_only()`  
- Logs the number of rows processed at each stage  
- Handles errors gracefully  
- Exports:

  **Synthetic dataset** → [synthetic_retail_dataset.csv](Section_1/Task_2_ETL_Process%20_Implementation/Datasets/synthetic_retail_dataset.csv)
  
  **Transformed dataset** → [transformed_retail_dataset.csv](Section_1/Task_2_ETL_Process%20_Implementation/Datasets/transformed_retail_dataset.csv)
  
  **SQLite database** → [retail_db](Section_1/Task_2_ETL_Process%20_Implementation/Datasets/retail_db)

**ETL Log Output**

```
2025-08-14 23:46:38,674 - INFO - Starting ETL process...
2025-08-14 23:46:38,730 - INFO - Synthetic dataset exported as 'synthetic_retail_dataset.csv'
2025-08-14 23:46:38,731 - INFO - Rows after extraction: 1000
2025-08-14 23:46:38,748 - INFO - Transformed dataset exported as 'transformed_retail_dataset.csv'
2025-08-14 23:46:38,750 - INFO - Rows after transformation: 493
2025-08-14 23:46:38,751 - INFO - Loading data into SQLite database using external schema...
2025-08-14 23:46:38,994 - INFO - Data loaded successfully into SQLite database.
2025-08-14 23:46:38,996 - INFO - ETL process completed: only synthetic, transformed, and .db exported.
```




**Post-load Data:**  

- Fact and dimension tables can be found at:

 * SalesFact → [SalesFact.csv](https://github.com/ibtj21/DSA-2040_Practical_Exam_Hana_555/blob/main/Section_1/Task_2_ETL_Process%20_Implementation/PostLoad_Fact_and_Dimention_tables/SalesFact.csv)
   
 * CustomerDim → [CustomerDim.csv](https://github.com/ibtj21/DSA-2040_Practical_Exam_Hana_555/blob/main/Section_1/Task_2_ETL_Process%20_Implementation/PostLoad_Fact_and_Dimention_tables/CustomerDim.csv)
 
 * ProductDim → [ProductDim.csv](https://github.com/ibtj21/DSA-2040_Practical_Exam_Hana_555/blob/main/Section_1/Task_2_ETL_Process%20_Implementation/PostLoad_Fact_and_Dimention_tables/ProductDim.csv)

 * TimeDim → [TimeDim.csv](https://github.com/ibtj21/DSA-2040_Practical_Exam_Hana_555/blob/main/Section_1/Task_2_ETL_Process%20_Implementation/PostLoad_Fact_and_Dimention_tables/TimeDim.csv)


**For a deep dive into the ETL process:** [etl_retail.ipynb](Section_1/Task_2_ETL_Process%20_Implementation/etl_retail.ipynb)



````markdown
## 1.3 OLAP Queries and Analysis

Using the Data Warehouse from Task 2:

### 1.3.1 OLAP-style SQL Queries

<details>
<summary><strong>i. Roll-up: Total sales by country and quarter</strong></summary>

Aggregates sales at a higher level (quarterly) per country.

```sql
SELECT 
    c.country,            -- Country name
    d.quarter,            -- Quarter of the year
    SUM(f.total_sales) AS total_sales  -- Total sales summed
FROM fact_sales f
JOIN dim_customer c ON f.customer_id = c.customer_id
JOIN dim_date d ON f.date_id = d.date_id
GROUP BY c.country, d.quarter
ORDER BY c.country, d.quarter;
````

**Output CSV:** `path/to/rollup_output.csv`

</details>

<details>
<summary><strong>ii. Drill-down: Sales details for a specific country (UK) by month</strong></summary>

Shows detailed monthly sales per product for a given country.

```sql
SELECT 
    d.year,               -- Year of sale
    d.month,              -- Month of sale
    p.name AS product_name, -- Product name
    f.quantity,           -- Quantity sold
    f.total_sales         -- Sales amount
FROM fact_sales f
JOIN dim_customer c ON f.customer_id = c.customer_id
JOIN dim_date d ON f.date_id = d.date_id
JOIN dim_product p ON f.product_id = p.product_id
WHERE c.country = 'UK'   -- Filter for UK
ORDER BY d.year, d.month, p.name;
```

**Output CSV:** `path/to/drilldown_output.csv`

</details>

<details>
<summary><strong>iii. Slice: Total sales for Electronics category</strong></summary>

Filters the fact table to only the Electronics category.

```sql
SELECT 
    SUM(f.total_sales) AS total_sales
FROM fact_sales f
JOIN dim_product p ON f.product_id = p.product_id
WHERE p.category = 'Electronics';  -- Only Electronics products
```

**Output CSV:** `path/to/slice_output.csv`

</details>

### 1.3.2 Visualization

A bar chart of sales by country using Matplotlib was created.

| Visualization                | File Path                      |
| ---------------------------- | ------------------------------ |
| Sales by Country (Bar Chart) | `path/to/sales_by_country.png` |

### 1.3.3 Analysis of Results

#### Analysis of Total Sales Using OLAP Queries

This report summarizes the results of OLAP queries performed on the sales data warehouse, including roll-up, drill-down, and slice analyses.

**i. Roll-up Analysis: Total Sales by Country and Quarter**
The roll-up query aggregates sales by country and quarter, revealing top-performing regions. **Germany, Norway, France, and Italy** consistently generated high total sales across quarters, while **Netherlands, Portugal, Spain, and the United Kingdom** showed moderate performance. The **Unknown Country** category had minimal sales, likely due to missing or incomplete customer data. This analysis helps identify strong markets and seasonal trends.

**ii. Drill-down Analysis: Monthly Sales for a Specific Country**
Focusing on monthly sales for a specific country (e.g., the UK), sales fluctuate throughout the year. Peak months, such as January and September, contrast with lower months like February and August, indicating seasonal demand variations. Drill-down analysis allows decision-makers to examine performance at a finer granularity, supporting tactical planning such as promotions or stock allocation.

**iii. Slice Analysis: Sales of Electronics Category**
The slice query isolates total sales for the **Electronics category**, which amounted to **144,339.15**. This highlights product-specific performance, aiding decisions related to inventory, marketing campaigns, and product development strategies.

**iv. Insights and Decision-Making Support**
The data warehouse consolidates sales, customer, and product data, enabling fast OLAP queries for roll-up, drill-down, and slice analyses. These insights support strategic decisions, including identifying top-selling countries, seasonal trends, and high-performing product categories. Marketing, inventory, and operational strategies can be tailored accordingly.

**v. Effect of Synthetic Data**
Since synthetic data was used, some patterns or sales volumes may not perfectly reflect real-world markets. However, synthetic data allows testing OLAP queries, visualizations, and decision-support processes without exposing sensitive information.

```

This format:

- Makes **SQL queries collapsible** to keep the README tidy.  
- Includes **CSV and image placeholders** in a **table format** for clarity.  
- Keeps your original structure, flow, and content intact.  

I can also **add badges or sections for “Top-Selling Countries” and “Seasonal Trends” charts** to make it more visually appealing. Do you want me to do that next?
```
