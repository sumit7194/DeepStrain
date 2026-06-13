"""Live progress dashboard for all BlackHole sub-project runs.

Serves a self-refreshing page over the heartbeat files that echolib / rdlib /
pbh.progress already write to <sub>/results/progress/*.json. Stdlib only.

Run:  python3 dashboard.py [--port 8765]   then open http://localhost:8765
"""

from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def scan() -> list[dict]:
    now = time.time()
    runs = []
    for sub in sorted(ROOT.iterdir()):
        prog = sub / "results" / "progress"
        if not prog.is_dir():
            continue
        for f in sorted(prog.glob("*.json")):
            if f.name == "index.json":
                continue
            try:
                d = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue  # mid-write or torn file; next poll catches it
            d["project"] = sub.name
            d["age"] = now - d.get("updated", 0)
            runs.append(d)
    return runs


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>BlackHole — Run Active Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-color: #05030f;
    --panel-bg: rgba(20, 16, 43, 0.4);
    --border-color: rgba(255, 255, 255, 0.08);
    --text-primary: #f3f4f6;
    --text-muted: #9ca3af;
    --accent-blue: #6366f1;
    --accent-green: #10b981;
    --accent-amber: #f59e0b;
  }
  * { box-sizing: border-box; }
  body {
    background: radial-gradient(circle at 50% 30%, #120d2a 0%, #070512 60%, #020105 100%);
    color: var(--text-primary);
    font-family: 'Inter', -apple-system, sans-serif;
    margin: 0;
    padding: 32px;
    min-height: 100vh;
  }
  .font-mono {
    font-family: 'JetBrains Mono', monospace;
  }
  /* Header */
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 40px;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 24px;
  }
  .logo-container {
    display: flex;
    align-items: center;
    gap: 16px;
  }
  h1 {
    font-size: 24px;
    font-weight: 700;
    letter-spacing: 0.12em;
    margin: 0;
    background: linear-gradient(135deg, #fff 0%, #a5b4fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .subtitle {
    font-size: 12px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
  }
  .status-badge {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--border-color);
    padding: 8px 16px;
    border-radius: 9999px;
    font-size: 13px;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  
  /* Black hole animation */
  .bh-logo {
    animation: pulse-glow 4s ease-in-out infinite;
  }
  .bh-ring-1 {
    animation: spin 8s linear infinite;
    transform-origin: 50px 50px;
  }
  .bh-ring-2 {
    animation: spin 12s linear infinite reverse;
    transform-origin: 50px 50px;
  }
  @keyframes spin {
    100% { transform: rotate(360deg); }
  }
  @keyframes pulse-glow {
    0%, 100% { filter: drop-shadow(0 0 5px rgba(99, 102, 241, 0.4)); }
    50% { filter: drop-shadow(0 0 15px rgba(16, 185, 129, 0.7)); }
  }

  /* Sections */
  h2 {
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin: 40px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  h2::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border-color);
  }

  /* Grid layout */
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
    gap: 20px;
  }

  /* Active card */
  .card {
    background: var(--panel-bg);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .card:hover {
    transform: translateY(-4px);
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.15);
  }
  .card.active-card {
    border-color: rgba(16, 185, 129, 0.25);
    box-shadow: 0 8px 32px 0 rgba(16, 185, 129, 0.05);
  }
  .card.active-card:hover {
    border-color: var(--accent-green);
    box-shadow: 0 12px 40px 0 rgba(16, 185, 129, 0.2);
  }

  /* Card Header */
  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
  }
  .run-info {
    display: flex;
    flex-direction: column;
  }
  .run-name {
    font-size: 18px;
    font-weight: 600;
    color: #fff;
    letter-spacing: -0.01em;
  }
  .project-tag {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
  }
  .status-indicator {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
  }
  .dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }
  .dot.green {
    background: var(--accent-green);
    box-shadow: 0 0 8px var(--accent-green);
    animation: pulse-green 2s infinite;
  }
  .dot.yellow {
    background: var(--accent-amber);
    box-shadow: 0 0 8px var(--accent-amber);
  }
  .dot.gray {
    background: var(--text-muted);
  }
  @keyframes pulse-green {
    0% { transform: scale(0.95); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 1; box-shadow: 0 0 12px var(--accent-green); }
    100% { transform: scale(0.95); opacity: 0.5; }
  }

  /* Progress Section */
  .progress-container {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .progress-labels {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
  }
  .progress-pct {
    font-size: 28px;
    font-weight: 700;
    background: linear-gradient(90deg, #fff 0%, #cbd5e1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .progress-steps {
    font-size: 13px;
    color: var(--text-muted);
  }
  .progress-bar-bg {
    height: 6px;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 9999px;
    overflow: hidden;
  }
  .progress-bar-fill {
    height: 100%;
    border-radius: 9999px;
    background: linear-gradient(90deg, var(--accent-blue) 0%, var(--accent-green) 100%);
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 0 8px rgba(16, 185, 129, 0.5);
  }

  /* Meta Section */
  .meta-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    background: rgba(0, 0, 0, 0.15);
    border-radius: 8px;
    padding: 12px;
    border: 1px solid rgba(255, 255, 255, 0.02);
  }
  .meta-item {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .meta-label {
    font-size: 9px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .meta-value {
    font-size: 13px;
    font-weight: 500;
    color: #fff;
  }

  /* Loss / Sparkline wrapper */
  .spark-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    padding-top: 16px;
  }
  .spark-wrapper {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  /* Chips / Badges */
  .chips-container {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .chip {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 11px;
    color: var(--text-muted);
  }
  .chip-highlight {
    color: #a5b4fc;
    border-color: rgba(99, 102, 241, 0.2);
    background: rgba(99, 102, 241, 0.05);
  }

  /* Idle grid */
  .idle-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 16px;
  }
  .idle-card {
    padding: 18px;
    gap: 12px;
    background: rgba(20, 16, 43, 0.2);
  }
  .idle-card .run-name {
    font-size: 15px;
  }
  .idle-card:hover {
    border-color: rgba(255, 255, 255, 0.15);
    box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.2);
  }
  .idle-progress {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
  }
  .idle-progress .pct {
    font-weight: 600;
    color: #fff;
  }
  .idle-progress .steps {
    color: var(--text-muted);
  }
  .idle-footer {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: var(--text-muted);
    border-top: 1px solid rgba(255, 255, 255, 0.03);
    padding-top: 8px;
  }

  .empty {
    color: var(--text-muted);
    font-style: italic;
    font-size: 13px;
    grid-column: 1 / -1;
    padding: 32px;
    text-align: center;
    background: rgba(255, 255, 255, 0.02);
    border-radius: 12px;
    border: 1px dashed var(--border-color);
  }
</style></head><body>
<!-- Global Hidden SVG for Gradients -->
<svg style="display:none">
  <defs>
    <linearGradient id="spark-stroke-grad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#818cf8" />
      <stop offset="100%" stop-color="#34d399" />
    </linearGradient>
    <linearGradient id="spark-area-grad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#818cf8" stop-opacity="1" />
      <stop offset="100%" stop-color="#34d399" stop-opacity="0" />
    </linearGradient>
  </defs>
</svg>

<header>
  <div class="logo-container">
    <svg class="bh-logo" viewBox="0 0 100 100" width="44" height="44">
      <defs>
        <radialGradient id="bh-glow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#6366f1" stop-opacity="1"/>
          <stop offset="30%" stop-color="#4f46e5" stop-opacity="0.8"/>
          <stop offset="70%" stop-color="#10b981" stop-opacity="0.3"/>
          <stop offset="100%" stop-color="#000000" stop-opacity="0"/>
        </radialGradient>
      </defs>
      <circle cx="50" cy="50" r="45" fill="url(#bh-glow)" />
      <circle cx="50" cy="50" r="14" fill="#000" />
      <ellipse class="bh-ring-1" cx="50" cy="50" rx="30" ry="7" fill="none" stroke="#6366f1" stroke-width="1.5" opacity="0.8" transform="rotate(-15 50 50)"/>
      <ellipse class="bh-ring-2" cx="50" cy="50" rx="38" ry="11" fill="none" stroke="#10b981" stroke-width="1" opacity="0.6" transform="rotate(-15 50 50)"/>
    </svg>
    <div>
      <h1>BLACKHOLE</h1>
      <div class="subtitle">Run & Progress Monitor</div>
    </div>
  </div>
  <div class="status-badge" id="sub"><span class="dot green"></span>scanning…</div>
</header>

<h2>Active Runs</h2>
<div class="grid" id="active"></div>

<h2>Finished / Idle Runs</h2>
<div class="grid idle-grid" id="idle"></div>

<script>
const hist = {};   // run -> [(t, step)] for client-side rate/ETA
function fmtAge(s){ 
  if(s<60) return Math.round(s)+"s ago";
  if(s<3600) return Math.round(s/60)+"m ago";
  if(s<86400) return (s/3600).toFixed(1)+"h ago";
  return (s/86400).toFixed(1)+"d ago"; 
}
function fmtEta(s){ 
  if(!isFinite(s)||s<0) return "—";
  if(s<90) return Math.round(s)+"s"; 
  if(s<5400) return Math.round(s/60)+"m";
  return (s/3600).toFixed(1)+"h"; 
}
function rateEta(key, step, total){
  const now = Date.now()/1000;
  (hist[key] = hist[key]||[]).push([now, step]);
  if(hist[key].length>40) hist[key].shift();
  const h = hist[key].filter(p => now-p[0] < 300);
  if(h.length<2) return [null,null];
  const dt = h[h.length-1][0]-h[0][0], ds = h[h.length-1][1]-h[0][1];
  if(dt<=0||ds<=0) return [null,null];
  return [ds/dt, (total-step)/(ds/dt)];
}
function spark(history){
  if(!history || history.length<3) return "";
  const v = history.map(p=>p[1]), mn=Math.min(...v), mx=Math.max(...v), r=(mx-mn)||1;
  const pts = v.map((y,i)=>`${(i/(v.length-1)*120).toFixed(1)},${(22-(y-mn)/r*20).toFixed(1)}`);
  const polyPoints = `0,24 ${pts.join(" ")} 120,24`;
  const linePoints = pts.join(" ");
  return `<div class="spark-row">
    <div class="spark-wrapper">
      <svg class="spark" width="120" height="24">
        <polygon points="${polyPoints}" fill="url(#spark-area-grad)" opacity="0.15" />
        <polyline points="${linePoints}" fill="none" stroke="url(#spark-stroke-grad)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </div>
    <span class="chip font-mono chip-highlight">loss ${v[v.length-1].toPrecision(4)}</span>
  </div>`;
}
async function tick(){
  let runs;
  try { runs = await (await fetch("/data")).json(); }
  catch(e){ document.getElementById("sub").innerHTML = "<span class='dot yellow'></span>server unreachable"; return; }
  runs.sort((a,b)=>a.age-b.age);
  const act = runs.filter(r=>r.age<120 && r.step<r.total);
  const idle = runs.filter(r=>!act.includes(r));
  document.getElementById("sub").innerHTML =
    `<span class="dot green"></span>${act.length} active · ${idle.length} idle · ${new Date().toLocaleTimeString()}`;
  document.getElementById("active").innerHTML = act.length ? act.map(r=>{
    const key = r.project+"/"+r.run, pct = r.total? (100*r.step/r.total) : 0;
    const [rate, eta] = rateEta(key, r.step, r.total);
    const chips = Object.entries(r.metrics||{}).map(([k,v])=>
      `<span class="chip font-mono">${k} ${(+v).toPrecision(4)}</span>`).join("");
    return `<div class="card active-card">
      <div class="card-header">
        <div class="run-info">
          <span class="run-name">${r.run}</span>
          <span class="project-tag">${r.project}</span>
        </div>
        <div class="status-indicator">
          <span class="dot green"></span>
          <span style="color: var(--accent-green)">active</span>
        </div>
      </div>
      <div class="progress-container">
        <div class="progress-labels">
          <span class="progress-pct font-mono">${pct.toFixed(1)}%</span>
          <span class="progress-steps font-mono">${r.step} / ${r.total}</span>
        </div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" style="width:${pct.toFixed(1)}%"></div>
        </div>
      </div>
      <div class="meta-grid">
        <div class="meta-item">
          <span class="meta-label">Rate</span>
          <span class="meta-value font-mono">${rate? rate.toFixed(2)+"/s" : "measuring…"}</span>
        </div>
        <div class="meta-item">
          <span class="meta-label">ETA</span>
          <span class="meta-value font-mono">${rate? fmtEta(eta) : "measuring…"}</span>
        </div>
      </div>
      <div class="chips-container">${chips}</div>
      ${spark(r.history)}
    </div>`;
  }).join("") : '<div class="empty">no active runs</div>';
  document.getElementById("idle").innerHTML = idle.length ? idle.map(r=>{
    const pct = r.total? Math.round(100*r.step/r.total) : 0;
    const isFinished = r.step>=r.total;
    const dot = isFinished ? "gray" : "yellow";
    const statusText = isFinished ? "finished" : "idle";
    return `<div class="card idle-card">
      <div class="card-header">
        <div class="run-info">
          <span class="run-name">${r.run}</span>
          <span class="project-tag">${r.project}</span>
        </div>
        <div class="status-indicator">
          <span class="dot ${dot}"></span>
          <span style="color: ${isFinished?'var(--text-muted)':'var(--accent-amber)'}">${statusText}</span>
        </div>
      </div>
      <div class="idle-progress">
        <span class="pct font-mono">${pct}%</span>
        <span class="steps font-mono">${r.step} / ${r.total}</span>
      </div>
      <div class="idle-footer font-mono">
        <span>Updated</span>
        <span>${fmtAge(r.age)}</span>
      </div>
    </div>`;
  }).join("") : '<div class="empty">none</div>';
}
tick(); setInterval(tick, 2000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/data":
            body = json.dumps(scan()).encode()
            ctype = "application/json"
        elif self.path == "/":
            body = PAGE.encode()
            ctype = "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass  # keep the terminal quiet; the page is the output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    # record our own PID so this instance can be stopped precisely —
    # stop with:  kill "$(cat .dashboard.pid)"   (never a broad pkill -f)
    pidfile = ROOT / ".dashboard.pid"
    pidfile.write_text(str(os.getpid()))
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"dashboard: http://localhost:{args.port}  "
          f"(pid {os.getpid()} -> {pidfile.name}, Ctrl-C to stop)")
    try:
        srv.serve_forever()
    finally:
        pidfile.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
