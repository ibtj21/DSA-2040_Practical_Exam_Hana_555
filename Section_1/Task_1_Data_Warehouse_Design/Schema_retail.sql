CREATE TABLE IF NOT EXISTS ProductDim (
    ProductKey INTEGER PRIMARY KEY AUTOINCREMENT,
    StockCode TEXT,
    Description TEXT,
    Category TEXT,
    UnitPrice REAL
);

CREATE TABLE IF NOT EXISTS CustomerDim (
    CustomerKey INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerID TEXT,
    CustomerName TEXT,
    Country TEXT
);

CREATE TABLE IF NOT EXISTS TimeDim (
    TimeKey INTEGER PRIMARY KEY AUTOINCREMENT,
    InvoiceDate TEXT,
    Year INTEGER,
    Month INTEGER,
    Quarter INTEGER
);

CREATE TABLE IF NOT EXISTS SalesFact (
    SalesKey INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerKey INTEGER,
    ProductKey INTEGER,
    TimeKey INTEGER,
    Quantity INTEGER,
    TotalSales REAL,
    FOREIGN KEY (CustomerKey) REFERENCES CustomerDim(CustomerKey),
    FOREIGN KEY (ProductKey) REFERENCES ProductDim(ProductKey),
    FOREIGN KEY (TimeKey) REFERENCES TimeDim(TimeKey)
);
