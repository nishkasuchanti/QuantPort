# QUANTPORT Portfolio Optimiser

Deployable IB Computer Science IA application implementing constrained Monte Carlo portfolio generation, return/covariance calculations, custom sorting, linear/binary target search, efficient-frontier construction, rankings, CSV upload, and runtime benchmarking.

## Local

    python -m venv .venv
    .venv/Scripts/activate
    pip install -r requirements.txt
    python app.py

Open http://localhost:5000. CSV format: Date followed by 2-20 unique asset columns, with positive prices and at least three data rows.

## Render

Push this folder to GitHub. In Render choose New > Blueprint and select the repository. Render reads render.yaml automatically. No database or environment variables are needed.
