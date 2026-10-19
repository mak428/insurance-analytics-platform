# insurance-analytics-platform
This is an application that is built upon the knowledge that I gained from my Risk Theory (PSTAT 173) college course. I have various ideas about how the frontend visualizations could look, but the mathematical framework for the backend has been meticulously crafted to be fully applicable to real world risk scenarios.

# Insurance Risk Analytics Platform (backend)

FastAPI backend for a Risk Theory course project. Each course topic is a pure-Python service in
`app/services/` (testable without a web server) with a thin router in `app/routers/`.

## Run
```bash
pip install -r requirements.txt
# copy the 4 data files into ./data :
#   liability_claims.csv, insurance_dataset.csv, data_synthetic.csv, Dataset_of_health_insurance_portfolio.xlsx
uvicorn app.main:app --reload      # then open http://127.0.0.1:8000/docs
```

## Topic -> module -> endpoint
| Course topic | Service | Endpoint |
|---|---|---|
| Continuous distributions + MLE (severity) | severity_service.py | POST /api/severity/fit |
| MLE (frequency), Compound Poisson | frequency_service.py | POST /api/frequency/fit, /api/frequency/compound-poisson |
| Monte Carlo | monte_carlo_service.py | POST /api/monte-carlo/simulate |
| Numerical integration (stop-loss, Panjer) | stop_loss_service.py | POST /api/stop-loss/premium |
| Quadrature | quadrature_service.py | POST /api/quadrature/limited-expected-value, /convergence |
| Credibility (Buhlmann-Straub) | credibility_service.py | POST /api/credibility/buhlmann-straub, /validate |
| Time series | timeseries_service.py | GET /api/timeseries/monthly, POST /api/timeseries/forecast |
| Machine learning | ml_service.py | GET /api/ml/evaluate, POST /api/ml/predict-severity |

## Known caveats (read before presenting results)
- **Not yet run end-to-end:** the FastAPI routers and `timeseries_service.py` (needs statsmodels) were
  written in an offline sandbox and never executed. All other services were tested on the real data.
- **liability_claims.csv has no zero-claim policies** (every row has ClaimNb >= 1), so it is
  zero-truncated. The Poisson rate (~1.65 claims per exposure-year) is therefore overstated and
  the Poisson-vs-NegBin comparison is not reliable. Use it to demonstrate the method, not to
  price a real portfolio. ClaimAmount is also a policy-level total, not a per-claim amount.
- Pareto here means Lomax (Pareto II); scipy's `pareto` (Type I) degenerates with loc=0.
- Credibility uses `ID` (policy_insured) as the risk key; `ID_insured` alone repeats across policies.
- Time series: no per-claim date exists, so monthly volume = sum(Claim History) by Policy Renewal
  Date month (24 monthly points) - a proxy.


# Risk Theory Workbench (frontend)

React + Vite + Recharts. Each panel calls one backend endpoint and charts the result.

```bash
# terminal 1: backend (from insurance-analytics-backend/)
uvicorn app.main:app --reload
# terminal 2: this folder
npm install
npm run dev      # http://localhost:5173  (Vite proxies /api to port 8000)
```

Not yet built or run: written without network access, so expect small fixes on first launch.
Not included yet: stop-loss panel, compound Poisson / frequency panel, ML prediction form, data upload.