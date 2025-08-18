# Draupnir App

A financial modeling and portfolio management application.

## 📂 Project Structure

draupnir_app/
├─ venv/                         # your virtual env (optional but recommended)
├─ data/
│  └─ draupnir.db               # SQLite file lives here
├─ exports/                      # Excel exports land here (holdings, trades, forecasts)
├─ draupnir_core/                # Python package with your modules
│  ├─ __init__.py
│  ├─ settings.py
│  ├─ portfolio.py
│  ├─ trades.py
│  ├─ summary.py
│  ├─ forecast.py
│  ├─ forecast_engine.py
│  ├─ draupnir.py
│  ├─ tax_engine.py
│  └─ utils.py                   # (ok to keep for legacy helpers)
├─ app.py                        # Streamlit entrypoint
├─ requirements.txt
└─ README.md


---

## 🚀 Setup Instructions

These steps work on **Windows** using the Python launcher (`py`).  
If you’re on another OS, replace `py -3.13` with `python3` or `python` as appropriate.

### 1. Clone the Repository

Pick a local directory **outside Dropbox**. Examples:
- `C:\Users\ryann\Code` (main machine)
- `N:\Code` (other machine)

```powershell
cd C:\Users\ryann\Code       # or N:\Code
git clone https://github.com/safewurd/draupnir_app.git
cd draupnir_app


2. Create and Activate a Virtual Environment
py -3.13 -m venv .venv
.\.venv\Scripts\activate


If you need Python 3.12 instead (for package compatibility):

py -3.12 -m venv .venv
.\.venv\Scripts\activate

3. Install Dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

4. Run the App
# Example for Streamlit
streamlit run app.py


🔄 Daily Workflow

When switching between machines:

Pull latest changes

git pull

Hard git pull:

git reset --hard HEAD      # throw away local changes
git pull                   # now bring in latest from remote


Activate venv

.\.venv\Scripts\activate


Run app / make changes

Commit & push

git add -A
git commit -m "Describe your change"
git push

🗄 Database Files

The data/draupnir.db file is ignored by Git and not synced.
To run locally, either:

Copy data/draupnir.db from another machine, or

Run the project’s database initialization/migration script.

⚙ VS Code Setup

Interpreter: Ctrl+Shift+P → Python: Select Interpreter → choose .venv\Scripts\python.exe.

Auto-activate venv in terminal: Settings → search for Terminal: Activate Environment → enable.

🛠 Useful Commands
# Check Python version
python --version

# Export dependencies to requirements.txt
python -m pip freeze > requirements.txt

# Upgrade pip
python -m pip install --upgrade pip

📌 Notes

Always activate .venv before running python or pip.

Always git pull before starting work, and git push when finished.

Keep large data files, secrets, and local DBs out of Git (check .gitignore).

