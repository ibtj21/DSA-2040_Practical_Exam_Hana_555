-- =========================================
-- OLAP Queries on Data Warehouse
-- =========================================

-- 1 Roll-up: Total sales by country and quarter
-- Aggregates sales at a higher level (quarterly) per country
SELECT 
    c.country,            -- Country name
    d.quarter,            -- Quarter of the year
    SUM(f.total_sales) AS total_sales  -- Total sales summed
FROM fact_sales f
JOIN dim_customer c ON f.customer_id = c.customer_id
JOIN dim_date d ON f.date_id = d.date_id
GROUP BY c.country, d.quarter
ORDER BY c.country, d.quarter;

-- 2 Drill-down: Sales details for a specific country (UK) by month
-- Shows detailed monthly sales per product for a given country
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

-- 3 Slice: Total sales for Electronics category
-- Filters the fact table to only the Electronics category
SELECT 
    SUM(f.total_sales) AS total_sales
FROM fact_sales f
JOIN dim_product p ON f.product_id = p.product_id
WHERE p.category = 'Electronics';  -- Only Electronics products
