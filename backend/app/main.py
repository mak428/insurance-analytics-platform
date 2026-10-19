from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import credibility, frequency, ml, monte_carlo, quadrature, severity, stop_loss, timeseries

app = FastAPI(title="Insurance Risk Analytics Platform",
              description="Risk Theory course project: compound Poisson, severity MLE, credibility, "
                          "Monte Carlo, stop-loss, quadrature, time series, ML.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://localhost:5173"],
                   allow_methods=["*"], allow_headers=["*"])
for r in (severity, frequency, monte_carlo, stop_loss, quadrature, credibility, timeseries, ml):
    app.include_router(r.router)


@app.get("/health")
def health():
    return {"status": "ok"}
