"""
Agent Lightning Dashboard

Lightweight FastAPI + HTML dashboard for monitoring agent runs.
No React, no webpack. Pure HTML/CSS/JS served from FastAPI.

Run with:
    python -m agent_lightning.dashboard
    # or
    from agent_lightning.dashboard import run_dashboard
    run_dashboard()
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from agent_lightning.core.run_store import RunStore


def get_app(store: Optional[RunStore] = None):
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.responses import HTMLResponse, JSONResponse
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        raise ImportError("Dashboard requires fastapi and uvicorn. Install with: pip install agent-lightning[dashboard]")

    _store = store or RunStore()
    app = FastAPI(title="Agent Lightning Dashboard", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse(content=_get_dashboard_html())

    @app.get("/api/stats")
    async def stats(agent_id: Optional[str] = None):
        return _store.stats(agent_id=agent_id)

    @app.get("/api/runs")
    async def list_runs(
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ):
        runs = _store.list_runs(agent_id=agent_id, status=status, limit=limit, offset=offset)
        return [r.to_dict() for r in runs]

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str):
        run = _store.get(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return run.to_dict()

    @app.get("/api/agents")
    async def list_agents():
        # Get distinct agent IDs from the store
        import sqlite3
        with sqlite3.connect(_store.db_path) as conn:
            rows = conn.execute("SELECT DISTINCT agent_id, COUNT(*) as cnt FROM runs GROUP BY agent_id ORDER BY cnt DESC").fetchall()
        return [{"agent_id": r[0], "run_count": r[1]} for r in rows]

    @app.get("/api/optimized_prompt/{agent_id}")
    async def get_optimized_prompt(agent_id: str):
        prompt = _store.get_optimized_prompt(agent_id)
        return {"agent_id": agent_id, "prompt": prompt}

    return app


def run_dashboard(
    store: Optional[RunStore] = None,
    host: str = "127.0.0.1",
    port: int = 7860,
    reload: bool = False,
):
    """Start the dashboard web server."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError("Dashboard requires uvicorn. Install with: pip install agent-lightning[dashboard]")

    app = get_app(store=store)
    print(f"\n⚡ Agent Lightning Dashboard running at http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, reload=reload)


def _get_dashboard_html() -> str:
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agent Lightning ⚡</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');

  :root {
    --bg: #080A0E;
    --surface: #0F1218;
    --surface2: #161B24;
    --border: #1E2535;
    --accent: #00E5FF;
    --accent2: #7C3AED;
    --success: #10B981;
    --warn: #F59E0B;
    --danger: #EF4444;
    --text: #E2E8F0;
    --muted: #64748B;
    --mono: 'Space Mono', monospace;
    --sans: 'Syne', sans-serif;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Grid noise texture overlay */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: 
      radial-gradient(circle at 20% 20%, rgba(0,229,255,0.03) 0%, transparent 50%),
      radial-gradient(circle at 80% 80%, rgba(124,58,237,0.04) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
  }

  header {
    position: sticky;
    top: 0;
    z-index: 100;
    background: rgba(8,10,14,0.95);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    padding: 0 2rem;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .logo {
    font-family: var(--sans);
    font-weight: 800;
    font-size: 1.1rem;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .logo-bolt {
    color: var(--accent);
    font-size: 1.3rem;
  }

  .logo-text { color: var(--text); }
  .logo-sub { color: var(--muted); font-weight: 400; font-size: 0.75rem; margin-left: 0.5rem; font-family: var(--mono); }

  .header-actions { display: flex; align-items: center; gap: 1rem; }

  .status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 8px var(--success);
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  .refresh-btn {
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 0.4rem 0.9rem;
    border-radius: 6px;
    font-family: var(--mono);
    font-size: 0.75rem;
    cursor: pointer;
    transition: all 0.15s;
  }
  .refresh-btn:hover { border-color: var(--accent); color: var(--accent); }

  main {
    position: relative;
    z-index: 1;
    padding: 2rem;
    max-width: 1400px;
    margin: 0 auto;
  }

  /* Stats row */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
  }

  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
  }
  .stat-card:hover { border-color: rgba(0,229,255,0.3); }
  .stat-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    opacity: 0;
    transition: opacity 0.3s;
  }
  .stat-card:hover::after { opacity: 1; }

  .stat-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
  }

  .stat-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--text);
    line-height: 1;
  }

  .stat-value.accent { color: var(--accent); }
  .stat-value.success { color: var(--success); }
  .stat-value.warn { color: var(--warn); }

  /* Two-col layout */
  .layout { display: grid; grid-template-columns: 280px 1fr; gap: 1.5rem; }

  @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }

  /* Sidebar */
  .sidebar {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }

  .panel-header {
    padding: 0.8rem 1.2rem;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .agent-list { padding: 0.5rem; }

  .agent-item {
    padding: 0.6rem 0.8rem;
    border-radius: 6px;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.85rem;
    transition: all 0.15s;
    border: 1px solid transparent;
  }
  .agent-item:hover { background: var(--surface2); border-color: var(--border); }
  .agent-item.active { background: rgba(0,229,255,0.08); border-color: rgba(0,229,255,0.3); color: var(--accent); }

  .agent-badge {
    background: var(--surface2);
    border-radius: 999px;
    padding: 0.1rem 0.5rem;
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--muted);
  }

  /* Main content */
  .content { display: flex; flex-direction: column; gap: 1rem; }

  /* Filter bar */
  .filter-bar {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    align-items: center;
  }

  .filter-btn {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--muted);
    padding: 0.35rem 0.8rem;
    border-radius: 6px;
    font-family: var(--mono);
    font-size: 0.72rem;
    cursor: pointer;
    transition: all 0.15s;
  }
  .filter-btn:hover { border-color: var(--accent); color: var(--accent); }
  .filter-btn.active { background: rgba(0,229,255,0.1); border-color: var(--accent); color: var(--accent); }

  /* Runs table */
  .runs-table {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }

  table { width: 100%; border-collapse: collapse; }
  
  thead tr {
    border-bottom: 1px solid var(--border);
    background: var(--surface2);
  }

  th {
    padding: 0.7rem 1rem;
    text-align: left;
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 400;
  }

  tbody tr {
    border-bottom: 1px solid rgba(30,37,53,0.6);
    cursor: pointer;
    transition: background 0.1s;
  }
  tbody tr:hover { background: rgba(0,229,255,0.04); }
  tbody tr:last-child { border-bottom: none; }
  tbody tr.selected { background: rgba(0,229,255,0.06); }

  td {
    padding: 0.7rem 1rem;
    font-size: 0.83rem;
    vertical-align: middle;
  }

  .td-id {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--muted);
  }

  .status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-family: var(--mono);
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .status-completed { background: rgba(16,185,129,0.1); color: var(--success); border: 1px solid rgba(16,185,129,0.2); }
  .status-failed { background: rgba(239,68,68,0.1); color: var(--danger); border: 1px solid rgba(239,68,68,0.2); }
  .status-running { background: rgba(245,158,11,0.1); color: var(--warn); border: 1px solid rgba(245,158,11,0.2); }

  .reward-bar-wrap {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    min-width: 100px;
  }

  .reward-bar {
    flex: 1;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
  }

  .reward-bar-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
    transition: width 0.4s ease;
  }

  .reward-val {
    font-family: var(--mono);
    font-size: 0.72rem;
    min-width: 35px;
    text-align: right;
  }

  /* Detail panel */
  .detail-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    display: none;
  }
  .detail-panel.visible { display: block; }

  .detail-header {
    padding: 1rem 1.4rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .detail-title {
    font-family: var(--mono);
    font-size: 0.8rem;
    color: var(--accent);
  }

  .close-btn {
    background: none;
    border: 1px solid var(--border);
    color: var(--muted);
    width: 28px; height: 28px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 1rem;
    transition: all 0.15s;
  }
  .close-btn:hover { border-color: var(--danger); color: var(--danger); }

  .detail-body { padding: 1.4rem; }

  .detail-meta {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.8rem;
    margin-bottom: 1.4rem;
  }

  .meta-item { }
  .meta-key {
    font-family: var(--mono);
    font-size: 0.62rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.2rem;
  }
  .meta-val { font-size: 0.85rem; font-weight: 600; }

  .steps-list { display: flex; flex-direction: column; gap: 0.6rem; }

  .step-item {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    border-left: 3px solid var(--border);
  }
  .step-prompt { border-left-color: var(--accent2); }
  .step-llm_response { border-left-color: var(--accent); }
  .step-tool_call { border-left-color: var(--warn); }
  .step-tool_result { border-left-color: var(--success); }
  .step-reward { border-left-color: #F97316; }

  .step-type {
    font-family: var(--mono);
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 0.3rem;
  }

  .step-content {
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--text);
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 150px;
    overflow-y: auto;
    line-height: 1.5;
  }

  .step-meta {
    margin-top: 0.4rem;
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
  }

  .step-meta-item {
    font-family: var(--mono);
    font-size: 0.62rem;
    color: var(--muted);
  }
  .step-meta-item span { color: var(--text); }

  /* Reward chart */
  .chart-wrap {
    height: 120px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
  }

  .chart-label {
    font-family: var(--mono);
    font-size: 0.62rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
  }

  canvas { width: 100% !important; }

  .empty-state {
    text-align: center;
    padding: 3rem;
    color: var(--muted);
    font-family: var(--mono);
    font-size: 0.8rem;
  }

  .empty-state-icon { font-size: 2rem; margin-bottom: 0.5rem; opacity: 0.5; }

  .loading {
    text-align: center;
    padding: 2rem;
    color: var(--muted);
    font-family: var(--mono);
    font-size: 0.75rem;
  }

  .tag {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    background: rgba(124,58,237,0.1);
    border: 1px solid rgba(124,58,237,0.2);
    color: #A78BFA;
    font-family: var(--mono);
    font-size: 0.62rem;
    margin-right: 0.2rem;
  }

  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--surface); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>
<header>
  <div class="logo">
    <span class="logo-bolt">⚡</span>
    <span class="logo-text">Agent Lightning</span>
    <span class="logo-sub">v0.1.0</span>
  </div>
  <div class="header-actions">
    <div class="status-dot"></div>
    <button class="refresh-btn" onclick="loadAll()">↻ Refresh</button>
  </div>
</header>

<main>
  <div class="stats-grid" id="statsGrid">
    <div class="stat-card"><div class="stat-label">Total Runs</div><div class="stat-value accent" id="st-total">—</div></div>
    <div class="stat-card"><div class="stat-label">Completed</div><div class="stat-value success" id="st-completed">—</div></div>
    <div class="stat-card"><div class="stat-label">Failed</div><div class="stat-value" id="st-failed">—</div></div>
    <div class="stat-card"><div class="stat-label">Avg Reward</div><div class="stat-value" id="st-avg">—</div></div>
    <div class="stat-card"><div class="stat-label">Max Reward</div><div class="stat-value warn" id="st-max">—</div></div>
  </div>

  <div class="layout">
    <div class="sidebar">
      <div class="panel">
        <div class="panel-header">Agents</div>
        <div class="agent-list" id="agentList"><div class="loading">Loading...</div></div>
      </div>
    </div>

    <div class="content">
      <div class="filter-bar">
        <button class="filter-btn active" data-status="" onclick="setFilter(this,'')">All</button>
        <button class="filter-btn" data-status="completed" onclick="setFilter(this,'completed')">Completed</button>
        <button class="filter-btn" data-status="failed" onclick="setFilter(this,'failed')">Failed</button>
        <button class="filter-btn" data-status="running" onclick="setFilter(this,'running')">Running</button>
      </div>

      <div class="runs-table">
        <div id="runsTableWrap"><div class="loading">Loading runs...</div></div>
      </div>

      <div class="detail-panel" id="detailPanel">
        <div class="detail-header">
          <span class="detail-title" id="detailTitle">Run Detail</span>
          <button class="close-btn" onclick="closeDetail()">×</button>
        </div>
        <div class="detail-body" id="detailBody"></div>
      </div>
    </div>
  </div>
</main>

<script>
let currentAgentId = null;
let currentStatus = '';
let allRuns = [];

async function api(path) {
  const r = await fetch('/api' + path);
  return r.json();
}

async function loadStats(agentId) {
  const s = await api('/stats' + (agentId ? '?agent_id=' + agentId : ''));
  document.getElementById('st-total').textContent = s.total ?? '0';
  document.getElementById('st-completed').textContent = s.completed ?? '0';
  document.getElementById('st-failed').textContent = s.failed ?? '0';
  document.getElementById('st-avg').textContent = s.avg_reward != null ? s.avg_reward.toFixed(3) : '0.000';
  document.getElementById('st-max').textContent = s.max_reward != null ? s.max_reward.toFixed(3) : '0.000';
}

async function loadAgents() {
  const agents = await api('/agents');
  const el = document.getElementById('agentList');
  if (!agents.length) {
    el.innerHTML = '<div class="empty-state"><div class="empty-state-icon">🤖</div>No agents yet</div>';
    return;
  }
  el.innerHTML = `
    <div class="agent-item ${!currentAgentId ? 'active' : ''}" onclick="selectAgent(null)">
      <span>All Agents</span>
      <span class="agent-badge">${agents.reduce((a,x)=>a+x.run_count,0)}</span>
    </div>
    ${agents.map(a => `
      <div class="agent-item ${currentAgentId===a.agent_id?'active':''}" onclick="selectAgent('${a.agent_id}')">
        <span>${a.agent_id}</span>
        <span class="agent-badge">${a.run_count}</span>
      </div>
    `).join('')}
  `;
}

async function loadRuns() {
  const params = new URLSearchParams();
  if (currentAgentId) params.set('agent_id', currentAgentId);
  if (currentStatus) params.set('status', currentStatus);
  params.set('limit', '100');

  const runs = await api('/runs?' + params.toString());
  allRuns = runs;
  renderRunsTable(runs);
}

function renderRunsTable(runs) {
  const wrap = document.getElementById('runsTableWrap');
  if (!runs.length) {
    wrap.innerHTML = '<div class="empty-state"><div class="empty-state-icon">📭</div>No runs found</div>';
    return;
  }

  const maxReward = Math.max(...runs.map(r => Math.abs(r.total_reward || 0)), 1);

  wrap.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Run ID</th>
          <th>Agent</th>
          <th>Status</th>
          <th>Reward</th>
          <th>Steps</th>
          <th>Duration</th>
          <th>Tags</th>
          <th>Created</th>
        </tr>
      </thead>
      <tbody>
        ${runs.map(run => {
          const reward = run.total_reward || 0;
          const pct = Math.min(100, (Math.abs(reward) / maxReward) * 100);
          const ts = new Date(run.created_at).toLocaleString();
          const dur = run.duration_ms ? (run.duration_ms/1000).toFixed(1)+'s' : '—';
          return `
            <tr onclick="showDetail('${run.run_id}')">
              <td class="td-id">${run.run_id.slice(0,8)}…</td>
              <td>${run.agent_id}</td>
              <td><span class="status-badge status-${run.status}">${run.status}</span></td>
              <td>
                <div class="reward-bar-wrap">
                  <div class="reward-bar"><div class="reward-bar-fill" style="width:${pct}%"></div></div>
                  <span class="reward-val">${reward.toFixed(2)}</span>
                </div>
              </td>
              <td>${run.steps?.length ?? 0}</td>
              <td style="font-family:var(--mono);font-size:0.72rem;color:var(--muted)">${dur}</td>
              <td>${(run.tags||[]).map(t=>`<span class="tag">${t}</span>`).join('')}</td>
              <td style="font-family:var(--mono);font-size:0.7rem;color:var(--muted)">${ts}</td>
            </tr>`;
        }).join('')}
      </tbody>
    </table>
  `;
}

async function showDetail(runId) {
  const run = await api('/runs/' + runId);
  document.getElementById('detailTitle').textContent = '⚡ Run: ' + runId.slice(0,16) + '…';
  document.getElementById('detailBody').innerHTML = renderDetail(run);
  document.getElementById('detailPanel').classList.add('visible');
}

function renderDetail(run) {
  const stepTypeColors = {
    prompt: '#7C3AED',
    llm_response: '#00E5FF',
    tool_call: '#F59E0B',
    tool_result: '#10B981',
    reward: '#F97316',
    custom: '#64748B',
  };

  const metaHtml = `
    <div class="detail-meta">
      <div class="meta-item"><div class="meta-key">Status</div><div class="meta-val">${run.status}</div></div>
      <div class="meta-item"><div class="meta-key">Total Reward</div><div class="meta-val" style="color:var(--accent)">${(run.total_reward||0).toFixed(4)}</div></div>
      <div class="meta-item"><div class="meta-key">Steps</div><div class="meta-val">${run.steps?.length ?? 0}</div></div>
      <div class="meta-item"><div class="meta-key">Duration</div><div class="meta-val">${run.duration_ms ? (run.duration_ms/1000).toFixed(2)+'s' : '—'}</div></div>
      <div class="meta-item"><div class="meta-key">Tokens</div><div class="meta-val">${run.total_tokens || 0}</div></div>
      <div class="meta-item"><div class="meta-key">Session</div><div class="meta-val" style="font-family:var(--mono);font-size:0.75rem">${run.session_id || '—'}</div></div>
    </div>`;

  const stepsHtml = (run.steps || []).map(s => {
    const color = stepTypeColors[s.step_type] || '#64748B';
    const content = typeof s.content === 'string' ? s.content : JSON.stringify(s.content, null, 2);
    const metaItems = [];
    if (s.model) metaItems.push(`model: <span>${s.model}</span>`);
    if (s.latency_ms) metaItems.push(`latency: <span>${s.latency_ms.toFixed(0)}ms</span>`);
    if (s.input_tokens) metaItems.push(`in: <span>${s.input_tokens} tok</span>`);
    if (s.output_tokens) metaItems.push(`out: <span>${s.output_tokens} tok</span>`);
    if (s.tool_name) metaItems.push(`tool: <span>${s.tool_name}</span>`);
    return `
      <div class="step-item step-${s.step_type}" style="border-left-color:${color}">
        <div class="step-type">${s.step_type}</div>
        <div class="step-content">${escHtml(content?.slice(0, 600) || '')}</div>
        ${metaItems.length ? `<div class="step-meta">${metaItems.map(m=>`<div class="step-meta-item">${m}</div>`).join('')}</div>` : ''}
      </div>`;
  }).join('');

  const rewardsHtml = (run.rewards || []).length ? `
    <div style="margin-bottom:1rem">
      <div class="panel-header" style="padding:0 0 0.5rem 0;border:none">Rewards</div>
      ${run.rewards.map(r => `
        <div style="display:flex;justify-content:space-between;padding:0.3rem 0;border-bottom:1px solid var(--border);font-size:0.82rem">
          <span style="color:var(--muted);font-family:var(--mono);font-size:0.72rem">${r.label}</span>
          <span style="color:#F97316;font-family:var(--mono);font-weight:700">${r.value.toFixed(4)}</span>
        </div>
      `).join('')}
    </div>` : '';

  return metaHtml + rewardsHtml + `<div class="panel-header" style="padding:0 0 0.8rem 0;border:none">Steps (${(run.steps||[]).length})</div><div class="steps-list">${stepsHtml || '<div class="empty-state">No steps recorded</div>'}</div>`;
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function closeDetail() {
  document.getElementById('detailPanel').classList.remove('visible');
}

function selectAgent(id) {
  currentAgentId = id;
  loadStats(id);
  loadRuns();
  loadAgents();
}

function setFilter(btn, status) {
  currentStatus = status;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  loadRuns();
}

async function loadAll() {
  await Promise.all([loadStats(currentAgentId), loadAgents(), loadRuns()]);
}

loadAll();
setInterval(loadAll, 10000);
</script>
</body>
</html>"""
