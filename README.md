# Kernel Conditional Moment Test for Predictability

Applies a kernel‑based conditional moment test to detect non‑linear predictability from macro variables (VIX, DXY, yields). The test statistic measures how much macro variables predict ETF returns beyond linear models. Higher score → greater predictability.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Removes linear dependence via regression
- Gaussian kernel on macro variables
- Test statistic T = (1/n²) Σ K(macro_i, macro_j) * resid_i * resid_j
- Score = T (higher = more predictable)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-kernel-conditional-moment-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (O(n²) but fast for typical window sizes)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High score → ETF returns are non‑linearly predictable from macro variables.
- Low score → linear model captures all predictability.

## Requirements

See `requirements.txt`.
