-- Dimension: Date
DROP TABLE IF EXISTS fact_sales;
DROP TABLE IF EXISTS dim_customer;
DROP TABLE IF EXISTS dim_product;
DROP TABLE IF EXISTS dim_date;


CREATE TABLE dim_date (
    date_id INTEGER PRIMARY KEY,       
    full_date TEXT,
    day INTEGER,
    month INTEGER,
    quarter INTEGER,
    year INTEGER
);

-- Dimension: Product
CREATE TABLE dim_product (
    product_id INTEGER PRIMARY KEY,    
    stock_code TEXT,
    name TEXT,
    category TEXT
);


-- Dimension: Customer
CREATE TABLE dim_customer (
    customer_id INTEGER PRIMARY KEY,   
    customer_code TEXT,                
    country TEXT
);

-- Fact: Sales
CREATE TABLE fact_sales (
    sale_id INTEGER PRIMARY KEY,       
    invoice_no TEXT,
    date_id INTEGER,   
    product_id INTEGER,
    customer_id INTEGER,
    quantity INTEGER,
    unit_price REAL,
    total_sales REAL,
    FOREIGN KEY (date_id) REFERENCES dim_date(date_id),
    FOREIGN KEY (product_id) REFERENCES dim_product(product_id),
    FOREIGN KEY (customer_id) REFERENCES dim_customer(customer_id)
);
