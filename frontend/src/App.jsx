import { useState } from "react";
import {
  ResponsiveContainer, LineChart, Line, BarChart, Bar, Cell, XAxis, YAxis, Tooltip, Legend, CartesianGrid,
} from "recharts";

/* ---------- helpers ---------- */
async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${(await res.text()).slice(0, 160)}`);
  return res.json();
}
const money = (x) => (x >= 1e6 ? `${(x / 1e6).toFixed(2)}M` : x >= 1e3 ? `${(x / 1e3).toFixed(1)}k` : x.toFixed(0));
const num = (x, d = 3) => (Math.abs(x) < 0.001 && x !== 0 ? x.toExponential(1) : Number(x).toFixed(d));
const COLORS = { empirical: "var(--ink)", pareto: "var(--accent)", lognormal: "var(--c2)", weibull: "var(--c3)", gamma: "var(--c4)" };

function Panel({ title, sub, onRun, label = "Run", wide, children }) {
  const [state, setState] = useState({ loading: false, error: null, data: null });
  const run = async () => {
    setState((s) => ({ ...s, loading: true, error: null }));
    try { setState({ loading: false, error: null, data: await onRun() }); }
    catch (e) { setState((s) => ({ ...s, loading: false, error: e.message })); }
  };
  return (
    <section className={wide ? "wide" : ""}>
      <h2>{title}</h2>
      <p className="sub">{sub}</p>
      <div className="bar">
        <button onClick={run} disabled={state.loading}>{state.loading ? "Running…" : label}</button>
        {state.error && <span className="err">Request failed: {state.error}. Is the backend running on port 8000?</span>}
      </div>
      {state.data ? children(state.data) : !state.loading && <p className="hint">Run this analysis to see results.</p>}
    </section>
  );
}
const Chart = ({ children, h = 300 }) => <div style={{ height: h }}><ResponsiveContainer>{children}</ResponsiveContainer></div>;
const Stats = ({ items }) => (
  <div className="stats">{items.map(([k, v]) => <div key={k}><span>{k}</span><b>{v}</b></div>)}</div>
);

/* ---------- panels ---------- */
function Severity({ onBest }) {
  return (
    <Panel title="Severity distributions" sub="Maximum-likelihood fits of four families to claim amounts; the tail chart shows who tracks the extremes." label="Fit severity models"
      onRun={async () => { const r = await api("/api/severity/fit", { source: "liability" }); onBest(r.fits[0]); return r; }}>
      {(r) => {
        const t = r.tail_comparison;
        const rows = t.quantiles.map((q, i) => ({ q: `${q * 100}%`, empirical: t.empirical[i], ...Object.fromEntries(r.fits.map((f) => [f.distribution, t[f.distribution][i]])) }));
        return (<>
          <Chart>
            <LineChart data={rows}>
              <CartesianGrid stroke="var(--rule)" /><XAxis dataKey="q" /><YAxis tickFormatter={money} />
              <Tooltip formatter={money} /><Legend />
              {["empirical", ...r.fits.map((f) => f.distribution)].map((k) => <Line key={k} dataKey={k} stroke={COLORS[k]} strokeWidth={k === "empirical" ? 3 : 2} dot={false} />)}
            </LineChart>
          </Chart>
          <table><thead><tr><th>Family</th><th>AIC</th><th>BIC</th><th>KS statistic</th></tr></thead>
            <tbody>{r.fits.map((f, i) => <tr key={f.distribution} className={i === 0 ? "best" : ""}><td>{f.distribution}</td><td>{f.aic.toFixed(0)}</td><td>{f.bic.toFixed(0)}</td><td>{num(f.ks_statistic)}</td></tr>)}</tbody></table>
        </>);
      }}
    </Panel>
  );
}

function MonteCarlo() {
  return (
    <Panel title="Aggregate annual loss" sub="Monte Carlo simulation of a compound Poisson portfolio. Darker bars mark the worst 5% of simulated years." label="Simulate 5,000 years"
      onRun={() => api("/api/monte-carlo/simulate", { n_trials: 5000 })}>
      {(r) => {
        const { counts, bin_edges: e } = r.histogram;
        const rows = counts.map((c, i) => ({ x: (e[i] + e[i + 1]) / 2, count: c, tail: e[i] >= r.var_95 }));
        return (<>
          <Stats items={[["Mean", money(r.mean)], ["VaR 95%", money(r.var_95)], ["TVaR 95%", money(r.tvar_95)], ["VaR 99%", money(r.var_99)], ["TVaR 99%", money(r.tvar_99)]]} />
          <Chart>
            <BarChart data={rows} barCategoryGap={0}>
              <CartesianGrid stroke="var(--rule)" vertical={false} /><XAxis dataKey="x" tickFormatter={money} minTickGap={40} /><YAxis />
              <Tooltip labelFormatter={money} />
              <Bar dataKey="count">{rows.map((d, i) => <Cell key={i} fill={d.tail ? "var(--tail)" : "var(--accent)"} />)}</Bar>
            </BarChart>
          </Chart>
        </>);
      }}
    </Panel>
  );
}

function Quadrature({ best }) {
  const fit = best || { distribution: "pareto", scipy_params: [3.8, 0, 4665] };
  return (
    <Panel title="Quadrature vs. Monte Carlo" sub={`Error when computing E[min(X, 10,000)] for the ${fit.distribution} severity fit, on log scales.`} label="Run convergence study"
      onRun={() => api("/api/quadrature/convergence", { distribution: fit.distribution, params: fit.scipy_params, limit: 10000 })}>
      {(r) => {
        const q = r.quadrature.node_counts.map((n, i) => ({ n, err: Math.max(r.quadrature.abs_errors[i], 1e-14) }));
        const m = r.monte_carlo.sample_sizes.map((n, i) => ({ n, err: Math.max(r.monte_carlo.abs_errors[i], 1e-14) }));
        return (<>
          <Stats items={[["Reference value", r.reference_value.toFixed(2)]]} />
          <Chart>
            <LineChart>
              <CartesianGrid stroke="var(--rule)" />
              <XAxis dataKey="n" type="number" scale="log" domain={["auto", "auto"]} name="evaluations" />
              <YAxis scale="log" domain={[1e-14, "auto"]} allowDataOverflow tickFormatter={(v) => v.toExponential(0)} />
              <Tooltip /><Legend />
              <Line data={q} dataKey="err" name="Gauss–Legendre (nodes)" stroke="var(--accent)" strokeWidth={2} />
              <Line data={m} dataKey="err" name="Monte Carlo (samples)" stroke="var(--tail)" strokeWidth={2} />
            </LineChart>
          </Chart>
        </>);
      }}
    </Panel>
  );
}

function Credibility() {
  return (
    <Panel title="Credibility on held-out years" sub="Each insured's latest period is hidden, then predicted three ways. Lower error is better." label="Validate credibility"
      onRun={() => api("/api/credibility/validate", {})}>
      {(r) => {
        const rows = [["Credibility blend", r.credibility_mse], ["Individual only", r.individual_only_mse], ["Portfolio only", r.portfolio_only_mse]].map(([name, mse]) => ({ name, mse }));
        return (<>
          <Stats items={[["Insureds tested", r.n_holdout_entities.toLocaleString()]]} />
          <Chart h={220}>
            <BarChart data={rows} layout="vertical" margin={{ left: 30 }}>
              <CartesianGrid stroke="var(--rule)" horizontal={false} /><XAxis type="number" tickFormatter={money} /><YAxis type="category" dataKey="name" width={130} />
              <Tooltip formatter={money} />
              <Bar dataKey="mse" name="Mean squared error">{rows.map((d, i) => <Cell key={i} fill={i === 0 ? "var(--accent)" : "var(--muted)"} />)}</Bar>
            </BarChart>
          </Chart>
        </>);
      }}
    </Panel>
  );
}

function TimeSeries() {
  return (
    <Panel title="Monthly claim volume forecast" sub="Holt-Winters forecast with a 95% band. Volume is a proxy built from policy renewal months." label="Forecast 6 months"
      onRun={async () => ({ hist: await api("/api/timeseries/monthly"), fc: await api("/api/timeseries/forecast", { horizon: 6, holdout: 6 }) })}>
      {({ hist, fc }) => {
        const rows = hist.dates.map((d, i) => ({ d, actual: hist.values[i] }))
          .concat(fc.forecast_dates.map((d, i) => ({ d, forecast: fc.forecast_values[i], lo: fc.ci_lower_95[i], hi: fc.ci_upper_95[i] })));
        return (<>
          <Stats items={[["Backtest MAPE", fc.backtest_mape_pct == null ? "n/a" : `${fc.backtest_mape_pct.toFixed(1)}%`], ["Stationary (ADF, 5%)", hist.stationarity.is_stationary_at_5pct ? "yes" : "no"]]} />
          <Chart>
            <LineChart data={rows}>
              <CartesianGrid stroke="var(--rule)" /><XAxis dataKey="d" tickFormatter={(d) => d.slice(0, 7)} minTickGap={30} /><YAxis domain={["auto", "auto"]} />
              <Tooltip /><Legend />
              <Line dataKey="actual" stroke="var(--ink)" strokeWidth={2} dot={false} />
              <Line dataKey="forecast" stroke="var(--accent)" strokeWidth={2} dot />
              <Line dataKey="hi" name="95% upper" stroke="var(--accent)" strokeDasharray="4 4" dot={false} />
              <Line dataKey="lo" name="95% lower" stroke="var(--accent)" strokeDasharray="4 4" dot={false} />
            </LineChart>
          </Chart>
        </>);
      }}
    </Panel>
  );
}

function Ml() {
  return (
    <Panel title="Severity prediction models" sub="Gamma GLM against gradient boosting. The closer a Lorenz curve bows below the diagonal, the better it ranks risk." label="Train and evaluate"
      onRun={() => api("/api/ml/evaluate")}>
      {(r) => {
        const g = r.lorenz_curve_gbm, l = r.lorenz_curve_glm;
        const rows = g.cumulative_share_of_policies.map((x, i) => ({ x, gbm: g.cumulative_share_of_losses[i], glm: l.cumulative_share_of_losses[i], even: x }));
        return (<>
          <Chart h={260}>
            <LineChart data={rows}>
              <CartesianGrid stroke="var(--rule)" /><XAxis dataKey="x" type="number" domain={[0, 1]} /><YAxis domain={[0, 1]} />
              <Tooltip formatter={(v) => v.toFixed(3)} /><Legend />
              <Line dataKey="even" name="No skill" stroke="var(--muted)" strokeDasharray="4 4" dot={false} />
              <Line dataKey="gbm" name="Gradient boosting" stroke="var(--accent)" strokeWidth={2} dot={false} />
              <Line dataKey="glm" name="Gamma GLM" stroke="var(--c2)" strokeWidth={2} dot={false} />
            </LineChart>
          </Chart>
          <table><thead><tr><th>Model</th><th>Gini</th><th>RMSE</th><th>MAE</th></tr></thead>
            <tbody>{[["Gamma GLM", r.glm_gamma], ["Gradient boosting", r.gbm]].map(([n, m]) => <tr key={n}><td>{n}</td><td>{m.gini.toFixed(3)}</td><td>{m.rmse.toFixed(0)}</td><td>{m.mae.toFixed(0)}</td></tr>)}</tbody></table>
        </>);
      }}
    </Panel>
  );
}

export default function App() {
  const [best, setBest] = useState(null);
  return (
    <>
      <header>
        <h1>Risk Theory Workbench</h1>
        <p>Fit loss models, simulate a year of portfolio losses, and test how well each method predicts. Run severity fitting first and the quadrature panel uses the winning distribution.</p>
      </header>
      <main>
        <Severity onBest={setBest} />
        <MonteCarlo />
        <Quadrature best={best} />
        <Credibility />
        <TimeSeries />
        <Ml />
      </main>
    </>
  );
}
