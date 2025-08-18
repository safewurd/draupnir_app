-- Draupnir schema export

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
    );

CREATE TABLE "EmploymentIncome" (
"year" INTEGER,
  "amount" INTEGER,
  "employer" TEXT,
  "note" TEXT
);

CREATE TABLE "MacroForecast" (
"year" INTEGER,
  "inflation" REAL,
  "growth" REAL,
  "fx" REAL,
  "note" TEXT
);

CREATE TABLE TaxRules (
        rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
        year INTEGER NOT NULL,
        jurisdiction TEXT NOT NULL,
        income_type TEXT NOT NULL,
        bracket_min REAL NOT NULL,
        bracket_max REAL,
        rate REAL NOT NULL,
        credit REAL DEFAULT 0
    );

CREATE TABLE "forecast_results_annual" (
            run_id INTEGER,
            year INTEGER,
            portfolio_id INTEGER,
            portfolio_name TEXT,
            tax_treatment TEXT,
            nominal_pretax_income REAL,
            nominal_taxes_paid REAL,
            nominal_after_tax_income REAL,
            nominal_effective_tax_rate REAL,
            real_pretax_income REAL,
            real_taxes_paid REAL,
            real_after_tax_income REAL,
            real_effective_tax_rate REAL,
            contributions REAL,
            withdrawals REAL, dividend_income_base REAL, interest_income_base REAL, dividends_reinvested_base REAL, interest_reinvested_base REAL,
            PRIMARY KEY (run_id, year, portfolio_id)
        );

CREATE TABLE forecast_results_monthly (
            run_id INTEGER,
            period TEXT,
            portfolio_id INTEGER,
            portfolio_name TEXT,
            tax_treatment TEXT,
            nominal_value REAL,
            real_value REAL,
            contributions REAL,
            withdrawals REAL, is_t0 INTEGER DEFAULT 0, cash_flow_from_distributions_base REAL, dividend_income_base REAL, interest_income_base REAL, dividends_reinvested_base REAL, interest_reinvested_base REAL,
            PRIMARY KEY (run_id, period, portfolio_id)
        );

CREATE TABLE forecast_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            params_json TEXT,
            settings_json TEXT
        );

CREATE TABLE global_settings (
            key TEXT PRIMARY KEY,
            value TEXT
        );

CREATE TABLE institutions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        );

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
        );

CREATE TABLE "portfolios" ("portfolio_id" INTEGER, "account_number" VARCHAR(20) NOT NULL, "portfolio_name" VARCHAR(50) NOT NULL, "portfolio_owner" VARCHAR(100) NOT NULL, "institution" VARCHAR(50), "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "tax_treatment" TEXT, interest_yield REAL, div_eligible_yield REAL, div_noneligible_yield REAL, reinvest_interest INTEGER DEFAULT 1, reinvest_dividends INTEGER DEFAULT 1, PRIMARY KEY ("portfolio_id"));

CREATE TABLE settings (
            key TEXT PRIMARY KEY,
            value TEXT
        );

CREATE TABLE sqlite_sequence(name,seq);

CREATE TABLE tax_distribution_assumptions(
        scope TEXT PRIMARY KEY,
        interest_yield REAL NOT NULL,
        eligible_div_yield REAL NOT NULL,
        noneligible_div_yield REAL NOT NULL,
        last_updated TEXT
    );

CREATE TABLE tax_policy(
        year INTEGER NOT NULL,
        province TEXT NOT NULL,
        cap_gains_inclusion REAL NOT NULL,
        eligible_div_gross_up REAL NOT NULL,
        noneligible_div_gross_up REAL NOT NULL,
        fed_eligible_div_credit_rate REAL NOT NULL,
        fed_noneligible_div_credit_rate REAL NOT NULL,
        prov_eligible_div_credit_rate REAL NOT NULL,
        prov_noneligible_div_credit_rate REAL NOT NULL,
        PRIMARY KEY (year, province)
    );

CREATE TABLE tax_profile(
        year INTEGER PRIMARY KEY,
        province TEXT NOT NULL
    );

CREATE TABLE tax_treatments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        );

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
);

CREATE TRIGGER trg_trades_action_check_insert
BEFORE INSERT ON trades
BEGIN
    SELECT CASE
        WHEN NEW.action IS NULL OR TRIM(NEW.action) = '' THEN
            RAISE(ABORT, 'Invalid action: action cannot be NULL or empty')
        WHEN UPPER(NEW.action) NOT IN ('BUY','SELL') THEN
            RAISE(ABORT, 'Invalid action: must be BUY or SELL')
    END;
END;

CREATE TRIGGER trg_trades_action_check_update
BEFORE UPDATE OF action ON trades
BEGIN
    SELECT CASE
        WHEN NEW.action IS NULL OR TRIM(NEW.action) = '' THEN
            RAISE(ABORT, 'Invalid action (update): action cannot be NULL or empty')
        WHEN UPPER(NEW.action) NOT IN ('BUY','SELL') THEN
            RAISE(ABORT, 'Invalid action (update): must be BUY or SELL')
    END;
END;

