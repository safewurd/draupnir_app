PRAGMA foreign_keys=OFF;
BEGIN;
CREATE TABLE AfterTaxProjections (
        atp_id INTEGER PRIMARY KEY AUTOINCREMENT,
        projection_id INTEGER,
        month INTEGER,
        gross_income REAL,
        taxes_paid REAL,
        net_income REAL,
        currency TEXT, portfolio_income_pre DECIMAL(15,2), portfolio_income_after DECIMAL(15,2), employment_income_pre DECIMAL(15,2), employment_income_after DECIMAL(15,2), tax_portfolio DECIMAL(15,2), tax_employment DECIMAL(15,2), total_income_pre DECIMAL(15,2), total_income_after DECIMAL(15,2),
        FOREIGN KEY (projection_id) REFERENCES Projections(projection_id)
    )
CREATE TABLE Assets (
        trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
        portfolio_id INTEGER NOT NULL,
        ticker VARCHAR(10) NOT NULL,
        quantity DECIMAL(10,2) NOT NULL,
        currency VARCHAR(3) NOT NULL,
        book_price DECIMAL(10,2),
        market_price DECIMAL(10,2),
        book_value DECIMAL(15,2),
        market_value DECIMAL(15,2),
        book_value_cad DECIMAL(15,2),
        market_value_cad DECIMAL(15,2) NOT NULL, account_number VARCHAR(20), book_value_usd DECIMAL(15,2), market_value_usd DECIMAL(15,2),
        FOREIGN KEY (portfolio_id) REFERENCES "Portfolios_old"(portfolio_id)
    )
CREATE TABLE "EmploymentIncome" (
"year" INTEGER,
  "annual_income" INTEGER,
  "currency" TEXT,
  "created_at" TEXT
)
CREATE TABLE FXRates (
        year INTEGER PRIMARY KEY,
        fx_rate DECIMAL(10,4) NOT NULL
    )
CREATE TABLE GlobalSettings (
        setting_name TEXT PRIMARY KEY,
        setting_value TEXT NOT NULL,
        description TEXT
    )
CREATE TABLE IncomeSources (
        source_id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_name TEXT NOT NULL,
        source_type TEXT NOT NULL,        -- e.g., Portfolio, Employment, Pension
        portfolio_id INTEGER,
        tax_treatment TEXT NOT NULL,      -- TFSA, RRSP, RRIF, Taxable, DBPension, DCPension, Employment
        FOREIGN KEY (portfolio_id) REFERENCES "Portfolios_old"(portfolio_id)
    )
CREATE TABLE Institutions (
            name TEXT PRIMARY KEY
        )
CREATE TABLE "MacroForecast" (
"year" INTEGER,
  "equities_growth" REAL,
  "fixedincome_growth" REAL,
  "moneymarket_growth" REAL,
  "fx_rate" REAL,
  "inflation_rate" REAL
)
CREATE TABLE Markets (
        market_id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker VARCHAR(10) NOT NULL,
        exchange VARCHAR(10),
        market_price DECIMAL(15,4) NOT NULL,
        currency VARCHAR(3),
        market_date DATE NOT NULL,
        provider TEXT DEFAULT 'Yahoo Finance',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
CREATE TABLE ProjectionInputs (
        input_id INTEGER PRIMARY KEY AUTOINCREMENT,
        portfolio_id INTEGER NOT NULL,
        annual_growth DECIMAL(5,4) NOT NULL,
        monthly_contrib DECIMAL(15,2) DEFAULT 0,
        monthly_withdrawal DECIMAL(15,2) DEFAULT 0,
        years INTEGER NOT NULL DEFAULT 40,
        scenario_name VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, account_number VARCHAR(20), annual_inflation DECIMAL(5,4) DEFAULT 0, base_currency VARCHAR(3) DEFAULT 'CAD',
        FOREIGN KEY (portfolio_id) REFERENCES "Portfolios_old"(portfolio_id)
    )
CREATE TABLE Projections (
        projection_id INTEGER PRIMARY KEY AUTOINCREMENT,
        input_id INTEGER NOT NULL,
        month INTEGER NOT NULL,
        portfolio_value DECIMAL(15,2) NOT NULL,
        income DECIMAL(15,2) NOT NULL, real_portfolio_value DECIMAL(15,2), real_income DECIMAL(15,2), currency VARCHAR(3), contribution DECIMAL(15,2) DEFAULT 0,
        FOREIGN KEY (input_id) REFERENCES ProjectionInputs(input_id)
    )
CREATE TABLE TaxRules (
        rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
        year INTEGER NOT NULL,
        jurisdiction TEXT NOT NULL,
        income_type TEXT NOT NULL,
        bracket_min REAL NOT NULL,
        bracket_max REAL,
        rate REAL NOT NULL,
        credit REAL DEFAULT 0
    )
CREATE TABLE TaxTreatments (
            name TEXT PRIMARY KEY
        )
CREATE TABLE "Trades_old" (
        trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
        account_number VARCHAR(20) NOT NULL,
        ticker VARCHAR(10) NOT NULL,
        trade_date DATE NOT NULL,
        trade_type TEXT NOT NULL CHECK(trade_type IN ('BUY','SELL')),
        quantity DECIMAL(10,2) NOT NULL,
        price DECIMAL(10,4) NOT NULL,
        currency VARCHAR(3) NOT NULL,
        fees DECIMAL(10,2) DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
CREATE TABLE forecast_results_annual (
            run_id INTEGER,
            year INTEGER,
            portfolio_id INTEGER,
            portfolio_name TEXT,
            tax_treatment TEXT,
            after_tax_income REAL,
            real_after_tax_income REAL,
            taxes_paid REAL,
            PRIMARY KEY (run_id, year, portfolio_id)
        )
CREATE TABLE forecast_results_monthly (
            run_id INTEGER,
            period TEXT,
            portfolio_id INTEGER,
            portfolio_name TEXT,
            tax_treatment TEXT,
            nominal_value REAL,
            real_value REAL,
            contributions REAL,
            withdrawals REAL,
            PRIMARY KEY (run_id, period, portfolio_id)
        )
CREATE TABLE forecast_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            params_json TEXT,
            settings_json TEXT
        )
CREATE TABLE global_settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
CREATE TABLE portfolio_flows (
            flow_id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER NOT NULL,
            kind TEXT NOT NULL,                  -- 'CONTRIBUTION' or 'WITHDRAWAL'
            amount REAL NOT NULL,                -- amount in portfolio currency
            frequency TEXT NOT NULL,             -- 'monthly' or 'annual'
            start_date TEXT NOT NULL,            -- 'YYYY-MM-01'
            end_date TEXT,                       -- nullable; if NULL, open-ended
            index_with_inflation INTEGER NOT NULL DEFAULT 1, -- 1/0
            notes TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
CREATE TABLE "portfolios" ("portfolio_id" INTEGER, "account_number" VARCHAR(20) NOT NULL, "portfolio_name" VARCHAR(50) NOT NULL, "portfolio_owner" VARCHAR(100) NOT NULL, "institution" VARCHAR(50), "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "tax_treatment" TEXT, PRIMARY KEY ("portfolio_id"))
CREATE TABLE settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
CREATE TABLE tax_treatments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
CREATE TABLE "trades" (
    trade_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date     TEXT,
    portfolio_name TEXT,
    portfolio_id   INTEGER,
    account_number TEXT,
    ticker         TEXT,
    currency       TEXT,
    action         TEXT,
    quantity       REAL,
    price          REAL,
    commission     REAL,
    fees           REAL,
    notes          TEXT,
    exchange       TEXT,
    yahoo_symbol   TEXT,
    created_at     TEXT
)
COMMIT;
