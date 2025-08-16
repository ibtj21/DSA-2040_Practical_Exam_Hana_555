-- Create ProductDim table if it does not exist
-- This table stores product-related information
CREATE TABLE IF NOT EXISTS ProductDim ( 
    ProductKey INTEGER PRIMARY KEY AUTOINCREMENT,  -- Unique identifier for each product
    StockCode TEXT,                                -- Product stock code
    Description TEXT,                              -- Product description
    Category TEXT,                                 -- Product category
    UnitPrice REAL                                 -- Unit price of the product
);

-- Create CustomerDim table if it does not exist
-- This table stores customer-related information
CREATE TABLE IF NOT EXISTS CustomerDim (
    CustomerKey INTEGER PRIMARY KEY AUTOINCREMENT, -- Unique identifier for each customer
    CustomerID TEXT,                               -- Customer ID (external system)
    CustomerName TEXT,                             -- Name of the customer
    Country TEXT                                   -- Country of the customer
);

-- Create TimeDim table if it does not exist
-- This table stores date and time-related information
CREATE TABLE IF NOT EXISTS TimeDim (
    TimeKey INTEGER PRIMARY KEY AUTOINCREMENT,  -- Unique identifier for each time record
    InvoiceDate TEXT,                           -- Date of the invoice
    Year INTEGER,                               -- Year part of the invoice date
    Month INTEGER,                              -- Month part of the invoice date
    Quarter INTEGER                             -- Quarter of the year
);

-- Create SalesFact table if it does not exist
-- This table stores the fact data (sales transactions) linking dimensions
CREATE TABLE IF NOT EXISTS SalesFact (
    SalesKey INTEGER PRIMARY KEY AUTOINCREMENT,     -- Unique identifier for each sale
    CustomerKey INTEGER,                            -- Foreign key referencing CustomerDim
    ProductKey INTEGER,                             -- Foreign key referencing ProductDim
    TimeKey INTEGER,                                -- Foreign key referencing TimeDim
    Quantity INTEGER,                               -- Number of products sold
    TotalSales REAL,                                -- Total sales amount
    FOREIGN KEY (CustomerKey) REFERENCES CustomerDim(CustomerKey),  -- Define relationship with CustomerDim
    FOREIGN KEY (ProductKey) REFERENCES ProductDim(ProductKey),    -- Define relationship with ProductDim
    FOREIGN KEY (TimeKey) REFERENCES TimeDim(TimeKey)               -- Define relationship with TimeDim
);
