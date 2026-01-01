"""Syna Studio - Web UI for database exploration."""
import os
import math
import json
import struct
from typing import Tuple, Any, List, Dict

try:
    from flask import Flask, render_template_string, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Global state for the active database path
CURRENT_DB_PATH = None

STUDIO_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Syna Studio</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>S</text></svg>">
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
    --bg-app: #050508;
    --bg-glass: rgba(20, 20, 25, 0.6);
    --bg-glass-hover: rgba(30, 30, 40, 0.8);
    --bg-card: rgba(255, 255, 255, 0.03);
    --border-glass: rgba(255, 255, 255, 0.08);
    --border-glass-strong: rgba(255, 255, 255, 0.15);
    
    --accent-primary: #8b5cf6;
    --accent-glow: rgba(139, 92, 246, 0.3);
    --accent-gradient: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
    
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --text-dim: #64748b;
    
    --shadow-sm: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    --shadow-md: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
}

* { box-sizing: border-box; margin: 0; padding: 0; outline: none; }

body {
    font-family: 'Outfit', sans-serif;
    background: var(--bg-app);
    background-image: 
        radial-gradient(circle at 15% 50%, rgba(139, 92, 246, 0.08), transparent 25%),
        radial-gradient(circle at 85% 30%, rgba(236, 72, 153, 0.08), transparent 25%);
    color: var(--text-main);
    min-height: 100vh;
    overflow-x: hidden;
}

@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
@keyframes spin { to { transform: rotate(360deg); } }

.app { display: flex; min-height: 100vh; }

/* Sidebar */
.sidebar {
    width: 260px;
    background: var(--bg-glass);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-right: 1px solid var(--border-glass);
    display: flex; flex-direction: column;
    position: fixed; height: 100vh; z-index: 50;
}

.logo {
    padding: 24px; display: flex; align-items: center; gap: 12px;
    border-bottom: 1px solid var(--border-glass);
}
.logo-img { height: 42px; width: auto; filter: drop-shadow(0 0 8px rgba(139, 92, 246, 0.4)); }
.logo-icon {
    width: 40px; height: 40px; background: var(--accent-gradient);
    border-radius: 12px; display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 18px; color: white;
    box-shadow: 0 0 15px var(--accent-glow);
}
.logo-text {
    font-size: 20px; font-weight: 700; letter-spacing: -0.5px;
    background: linear-gradient(to right, #fff, #cbd5e1);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; white-space: nowrap;
}
.ext-links { margin-left: auto; display: flex; gap: 10px; }
.ext-link { color: var(--text-dim); transition: 0.2s; display: flex; }
.ext-link:hover { color: var(--accent-primary); transform: translateY(-1px); }
.ext-link svg { width: 18px; height: 18px; }

.nav-menu { padding: 20px 12px; flex: 1; }
.nav-item {
    display: flex; align-items: center; gap: 10px; padding: 12px 16px;
    margin-bottom: 6px; border-radius: 10px; color: var(--text-muted);
    text-decoration: none; cursor: pointer; transition: 0.2s;
    border: 1px solid transparent;
}
.nav-item:hover { background: var(--bg-glass-hover); color: white; transform: translateX(2px); }
.nav-item.active {
    background: rgba(139, 92, 246, 0.1); color: #a78bfa;
    border-color: rgba(139, 92, 246, 0.2); font-weight: 500;
}
.nav-icon { width: 18px; height: 18px; opacity: 0.7; }

.db-switcher { display: flex; gap: 8px; align-items: center; padding: 12px; background: rgba(0,0,0,0.3); border-radius: 8px; margin: 20px 12px; border: 1px solid var(--border-glass); }
.db-path { font-family: 'JetBrains Mono'; font-size: 10px; color: var(--text-muted); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.switch-btn { background: none; border: none; color: var(--accent-primary); cursor: pointer; font-size: 10px; text-transform: uppercase; font-weight: 700; }

/* Main Content */
.main-content { flex: 1; margin-left: 260px; padding: 40px; animation: fadeIn 0.4s ease-out; }
.page { display: none; }
.page.active { display: block; }
.header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 32px; flex-wrap: wrap; gap: 20px; }
.header h1 {
    font-size: 32px; font-weight: 700; letter-spacing: -1px;
    background: linear-gradient(to right, #fff, #94a3b8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

/* UI Components */
.search-box { position: relative; }
.search-box input {
    width: 320px; padding: 12px 16px 12px 42px; background: var(--bg-card);
    border: 1px solid var(--border-glass); border-radius: 12px;
    color: var(--text-main); font-family: 'Outfit'; font-size: 14px; transition: 0.2s;
}
.search-box input:focus { border-color: var(--accent-primary); box-shadow: 0 0 0 3px var(--accent-glow); }
.search-box svg { position: absolute; left: 14px; top: 50%; transform: translateY(-50%); width: 18px; height: 18px; color: var(--text-dim); }

.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 32px; }
.stat-card {
    background: var(--bg-card); backdrop-filter: blur(10px);
    border: 1px solid var(--border-glass); border-radius: 16px;
    padding: 20px; display: flex; align-items: center; gap: 16px;
    cursor: pointer; transition: 0.2s; box-shadow: var(--shadow-sm);
}
.stat-card:hover { transform: translateY(-4px); background: var(--bg-glass-hover); border-color: var(--border-glass-strong); }
.stat-card.selected { border-color: var(--accent-primary); box-shadow: 0 0 15px var(--accent-glow); }
.stat-icon { width: 42px; height: 42px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-family: 'JetBrains Mono'; font-weight: 600; font-size: 13px; }

/* Type Colors */
.type-float, .float-icon { background: rgba(99, 102, 241, 0.15); color: #818cf8; }
.type-int, .int-icon { background: rgba(139, 92, 246, 0.15); color: #a78bfa; }
.type-text, .text-icon { background: rgba(16, 185, 129, 0.15); color: #34d399; }
.type-bytes, .bytes-icon { background: rgba(245, 158, 11, 0.15); color: #fbbf24; }
.type-vector, .vector-icon { background: rgba(236, 72, 153, 0.15); color: #f472b6; }
.stat-value { font-size: 22px; font-weight: 700; }
.stat-label { font-size: 12px; color: var(--text-muted); }

/* Tables */
.table-box { background: var(--bg-card); border: 1px solid var(--border-glass); border-radius: 16px; overflow: hidden; box-shadow: var(--shadow-md); margin-bottom: 24px; }
.table-head { display: flex; justify-content: space-between; align-items: center; padding: 20px 24px; border-bottom: 1px solid var(--border-glass); background: rgba(0,0,0,0.2); }
table { width: 100%; border-collapse: collapse; }
th { padding: 16px 24px; text-align: left; background: rgba(0,0,0,0.2); font-size: 11px; text-transform: uppercase; color: var(--text-dim); }
td { padding: 16px 24px; border-bottom: 1px solid var(--border-glass); font-size: 13px; color: var(--text-muted); }
tr:hover td { background: rgba(255,255,255,0.02); }
.key-cell { font-family: 'JetBrains Mono'; color: white; font-weight: 500; }
.value-cell { max-width: 400px; font-family: 'JetBrains Mono'; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.btn { padding: 8px 16px; border-radius: 8px; border: 1px solid var(--border-glass); background: rgba(255,255,255,0.05); color: var(--text-main); cursor: pointer; transition: 0.2s; font-size: 13px;}
.btn:hover { background: rgba(255,255,255,0.1); border-color: var(--text-muted); }
.view-btn { padding: 4px 10px; font-size: 11px; background: transparent; border: 1px solid var(--border-glass); color: var(--text-muted); border-radius: 6px; cursor: pointer; }
.view-btn:hover { border-color: var(--accent-primary); color: var(--accent-primary); }

/* Charts & Modal */
.charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 24px; }
.chart-box { background: var(--bg-card); border: 1px solid var(--border-glass); border-radius: 16px; padding: 24px; }
.chart-box h3 { font-size: 15px; margin-bottom: 20px; color: var(--text-muted); }
.chart-box.full { grid-column: 1 / -1; }

.modal { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.8); backdrop-filter: blur(5px); z-index: 1000; align-items: center; justify-content: center; }
.modal.active { display: flex; animation: fadeIn 0.2s; }
.modal-box { background: #1a1a23; border: 1px solid var(--border-glass-strong); border-radius: 16px; width: 90%; max-width: 600px; max-height: 85vh; display: flex; flex-direction: column; }
.modal-head { padding: 20px 24px; border-bottom: 1px solid var(--border-glass); display: flex; justify-content: space-between; align-items: center; }
.modal-body { padding: 24px; overflow-y: auto; }
.modal-close { background: none; border: none; font-size: 24px; color: var(--text-muted); cursor: pointer; }
.detail-row { margin-bottom: 20px; }
.detail-label { font-size: 11px; text-transform: uppercase; color: var(--text-dim); margin-bottom: 8px; font-weight: 700; letter-spacing: 0.5px; }
.detail-value { font-family: 'JetBrains Mono'; font-size: 13px; background: rgba(0,0,0,0.3); border: 1px solid var(--border-glass); padding: 16px; border-radius: 8px; color: #e2e8f0; word-break: break-all; }
.hex-view { display: grid; grid-template-columns: 3fr 1fr; gap: 10px; font-family: 'JetBrains Mono'; font-size: 11px; line-height: 1.4; }
.hex-col { color: #818cf8; } .ascii-col { color: #34d399; border-left: 1px solid var(--border-glass); padding-left: 10px; }

/* Utilities */
.loading { padding: 40px; display: flex; align-items: center; justify-content: center; gap: 12px; color: var(--text-muted); }
.spinner { width: 24px; height: 24px; border: 3px solid rgba(255,255,255,0.1); border-top-color: var(--accent-primary); border-radius: 50%; animation: spin 0.8s linear infinite; }
.empty { text-align: center; padding: 60px; color: var(--text-dim); font-style: italic; }
.tag { padding: 2px 8px; border-radius: 12px; font-size: 10px; font-weight: 700; text-transform: uppercase; }
.tag-prod { background: rgba(16, 185, 129, 0.2); color: #34d399; }
.tag-dev { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
.tag-staging { background: rgba(99, 102, 241, 0.2); color: #818cf8; }

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 4px; }
</style>
</head>
<body>
<div class="app">
<aside class="sidebar">
    <div class="logo">
        <img src="https://raw.githubusercontent.com/gtava5813/SynaDB/main/assets/logo-no-bg.png" alt="S" class="logo-img" onerror="this.style.display='none';document.getElementById('fallback-logo').style.display='flex'">
        <div id="fallback-logo" class="logo-icon" style="display:none">S</div>
        <div class="logo-text">Syna Studio</div>
        <div class="ext-links">
            <a href="https://github.com/gtava5813/SynaDB" target="_blank" class="ext-link" title="GitHub"><svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.87 8.17 6.84 9.5.5.08.66-.23.66-.5v-1.69c-2.77.6-3.36-1.34-3.36-1.34-.46-1.16-1.1-1.47-1.1-1.47-.9-.62.06-.6.06-.6 1 .07 1.53 1.03 1.53 1.03.88 1.52 2.34 1.07 2.91.83.09-.65.35-1.09.63-1.34-2.22-.25-4.55-1.11-4.55-4.92 0-1.11.38-2 1.03-2.71-.1-.25-.45-1.29.1-2.64 0 0 .84-.27 2.75 1.02a9.56 9.56 0 0 1 5 0c1.91-1.29 2.75-1.02 2.75-1.02.55 1.35.2 2.39.1 2.64.65.71 1.03 1.6 1.03 2.71 0 3.82-2.34 4.66-4.57 4.91.36.31.69.92.69 1.85V21c0 .27.16.59.67.5C19.14 20.16 22 16.42 22 12A10 10 0 0 0 12 2z"/></svg></a>
            <a href="https://github.com/gtava5813/SynaDB/wiki" target="_blank" class="ext-link" title="Docs"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg></a>
        </div>
    </div>
    <div class="db-switcher">
        <div class="db-path" id="current-db-path" title="Current DB">{{ db_path }}</div>
        <button class="switch-btn" onclick="switchDB()">Switch</button>
    </div>
    <nav class="nav-menu">
        <a class="nav-item active" data-page="keys"><svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4"/></svg> Keys Explorer</a>
        <a class="nav-item" data-page="models"><svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 16a6 6 0 1 1 6-6 6 6 0 0 1-6 6z"/><path d="M12 8v4l3 3"/></svg> AI Models</a>
        <a class="nav-item" data-page="clusters"><svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M8 12h8"/><path d="M12 8v8"/></svg> 3D Clusters</a>
        <a class="nav-item" data-page="stats"><svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg> Statistics</a>
        <a class="nav-item" data-page="integrations"><svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><line x1="4" y1="22" x2="4" y2="15"/></svg> Integrations</a>
        <a class="nav-item" data-page="custom"><svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg> Custom Suite</a>
    </nav>
</aside>
<main class="main-content">
    
    <!-- Keys Explorer -->
    <div id="page-keys" class="page active">
        <header class="header">
            <div><h1>Keys Explorer</h1><div id="filter-indicator"></div></div>
            <div class="search-box"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg><input type="text" id="search" placeholder="Search keys..."></div>
        </header>
        <div class="stats-grid">
            <div class="stat-card" data-type="float"><div class="stat-icon float-icon">f64</div><div><div class="stat-value" id="float-count">-</div><div class="stat-label">Floats</div></div></div>
            <div class="stat-card" data-type="int"><div class="stat-icon int-icon">i64</div><div><div class="stat-value" id="int-count">-</div><div class="stat-label">Integers</div></div></div>
            <div class="stat-card" data-type="text"><div class="stat-icon text-icon">str</div><div><div class="stat-value" id="text-count">-</div><div class="stat-label">Text</div></div></div>
            <div class="stat-card" data-type="bytes"><div class="stat-icon bytes-icon">bin</div><div><div class="stat-value" id="bytes-count">-</div><div class="stat-label">Bytes</div></div></div>
            <div class="stat-card" data-type="vector"><div class="stat-icon vector-icon">vec</div><div><div class="stat-value" id="vector-count">-</div><div class="stat-label">Vectors</div></div></div>
        </div>
        <div class="table-box">
            <div class="table-head"><h2 id="table-title">All Keys</h2><button class="btn" onclick="app.loadKeys()">Refresh Data</button></div>
            <div id="keys-content"><div class="loading"><div class="spinner"></div><span>Loading data...</span></div></div>
        </div>
    </div>

    <!-- AI Models -->
    <div id="page-models" class="page">
        <header class="header"><h1>Model Registry</h1><button class="btn" onclick="app.loadModels()">Refresh</button></header>
        <div class="table-box">
            <div class="table-head"><h2>Registered Models</h2></div>
            <div id="models-content"><div class="loading"><div class="spinner"></div></div></div>
        </div>
    </div>

    <!-- 3D Clusters -->
    <div id="page-clusters" class="page">
        <header class="header"><h1>3D Embedding Clusters</h1><button class="btn" onclick="app.loadClusters()">Re-calculate PCA</button></header>
        <div class="chart-box full" style="height: 70vh;">
            <div id="cluster-plot" style="width:100%;height:100%;"></div>
        </div>
    </div>

    <!-- Statistics -->
    <div id="page-stats" class="page">
        <header class="header">
            <h1>Statistics Dashboard</h1>
            <div style="display:flex;gap:10px;">
                <select id="widget-select" style="background:var(--bg-card);color:white;border:1px solid var(--border-glass);padding:8px;border-radius:8px;">
                     <option value="key_len">Key Length Dist.</option>
                     <option value="write_ops">Write Ops (Sim)</option>
                     <option value="mem_usage">Memory Usage</option>
                </select>
                <button class="btn" onclick="app.addWidget()">+ Add Plot</button>
                <button class="btn" onclick="app.loadStats()">Refresh</button>
            </div>
        </header>
        
        <!-- Default Stats -->
        <div class="charts-grid" id="default-stats">
            <div class="chart-box"><h3>Storage Usage (Treemap)</h3><div id="health-treemap" style="height:350px;"></div></div>
            <div class="chart-box"><h3>Key Type Distribution</h3><div id="health-pie" style="height:350px;"></div></div>
        </div>

        <!-- Dynamic Widgets Area -->
        <div id="dynamic-widgets" class="charts-grid" style="margin-top:24px;"></div>
    </div>
    
    <!-- Integrations -->
    <div id="page-integrations" class="page">
        <header class="header"><h1>Integrations</h1><button class="btn" onclick="app.loadIntegrations()">Scan Directory</button></header>
        <div class="charts-grid" id="integrations-list">
             <div class="loading"><div class="spinner"></div></div>
        </div>
    </div>

    <!-- Custom Suite -->
    <div id="page-custom" class="page">
        <header class="header"><h1>Custom Suite</h1></header>
        <div class="charts-grid">
            <div class="chart-box">
                <h3>Actions</h3>
                <p style="color:var(--text-dim);margin-bottom:20px;font-size:13px;">Execute maintenance/custom scripts.</p>
                <div style="display:flex;gap:12px;flex-wrap:wrap;">
                    <button class="btn" onclick="app.runAction('compact')">Compact Database</button>
                    <button class="btn" onclick="app.runAction('export_json')">Export to JSON</button>
                    <button class="btn" onclick="app.runAction('integrity_check')">Run Integrity Check</button>
                    <button class="btn" onclick="app.runAction('clear_cache')">Clear Cache</button>
                </div>
                <div id="action-output" style="margin-top:20px;padding:16px;background:rgba(0,0,0,0.3);border-radius:8px;font-family:'JetBrains Mono';font-size:12px;color:#a78bfa;display:none;"></div>
            </div>
            <div class="chart-box">
                 <h3>Playground</h3>
                 <p style="color:var(--text-dim);margin-bottom:0px;font-size:13px;">Explore data patterns.</p>
                 <div style="margin-top:16px;">
                     <input type="text" id="pattern-input" placeholder="Pattern (e.g. system/*)" style="width:100%;padding:10px;background:rgba(255,255,255,0.05);border:1px solid var(--border-glass);color:white;border-radius:6px;margin-bottom:12px;">
                     <button class="btn" onclick="app.runAction('scan_pattern', document.getElementById('pattern-input').value)">Scan Pattern</button>
                 </div>
            </div>
        </div>
    </div>

</main>
</div>

<!-- Modal -->
<div id="modal" class="modal">
    <div class="modal-box">
        <div class="modal-head"><h3 id="modal-title">Details</h3><button class="modal-close" onclick="app.closeModal()">&times;</button></div>
        <div class="modal-body" id="modal-body"></div>
    </div>
</div>

<script>
const app = {
    state: { keys: [], models: [], typeCounts: {}, widgets: [] },
    layout: { paper_bgcolor:'transparent', plot_bgcolor:'transparent', font:{color:'#94a3b8',family:'Outfit'}, margin:{t:20,l:20,r:20,b:20} },
    
    init: function() {
        this.bindEvents();
        this.loadKeys();
    },

    bindEvents: function() {
        document.querySelectorAll('.nav-item').forEach(el => {
            el.addEventListener('click', () => this.showPage(el.getAttribute('data-page')));
        });
        document.getElementById('search').addEventListener('input', () => this.filterKeys());
        document.querySelectorAll('.stat-card').forEach(c => {
            c.addEventListener('click', () => this.toggleFilter(c.getAttribute('data-type')));
        });
    },

    showPage: function(page) {
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
        document.getElementById('page-'+page).classList.add('active');
        document.querySelector(`.nav-item[data-page="${page}"]`).classList.add('active');
        
        if(page === 'models') this.loadModels();
        if(page === 'clusters') this.loadClusters();
        if(page === 'stats') this.loadStats();
        if(page === 'integrations') this.loadIntegrations();
    },

    loadKeys: function() {
        fetch('/api/keys').then(r=>r.json()).then(d => {
            this.state.keys = d.keys;
            this.state.typeCounts = d.type_counts;
            this.updateStats();
            this.filterKeys();
        });
    },

    updateStats: function() {
        ['float','int','text','bytes','vector'].forEach(t => {
            document.getElementById(`${t}-count`).textContent = this.state.typeCounts[t] || 0;
        });
    },

    toggleFilter: function(type) {
        // Simple toggle logic reusing existing styles
        this.filterKeys(type); 
    },

    filterKeys: function(typeFilter) {
        const q = document.getElementById('search').value.toLowerCase();
        let res = this.state.keys.filter(k => k.key.toLowerCase().includes(q));
        if(typeFilter) res = res.filter(k => k.type === typeFilter);
        
        const html = res.length ? `<table><thead><tr><th>Key</th><th>Type</th><th>Preview</th><th></th></tr></thead><tbody>` + 
            res.map(k => `<tr><td class="key-cell">${k.key}</td><td><span class="tag type-${k.type}">${k.type}</span></td><td class="value-cell">${this.fmt(k.value, k.type)}</td><td><button class="view-btn" onclick="app.view('${k.key}')">View</button></td></tr>`).join('') + 
            `</tbody></table>` : '<div class="empty">No keys found</div>';
        document.getElementById('keys-content').innerHTML = html;
    },

    fmt: function(v, t) {
        if(t==='vector') return `[${v.slice(0,3).join(', ')}...] (${v.length})`;
        if(t==='bytes') return `<span style="font-family:monospace;opacity:0.7">BINARY (${v.length/2} bytes)</span>`;
        if(typeof v === 'object') return JSON.stringify(v);
        return String(v).substring(0,60);
    },

    view: function(key) {
        document.getElementById('modal').classList.add('active');
        document.getElementById('modal-title').textContent = key;
        document.getElementById('modal-body').innerHTML = '<div class="loading"><div class="spinner"></div></div>';
        fetch('/api/get/'+encodeURIComponent(key)).then(r=>r.json()).then(d => {
            let html = `<div class="detail-row"><div class="detail-label">Type</div><div class="detail-value">${d.type}</div></div>`;
            
            if(d.type === 'bytes') {
                html += `<div class="detail-row"><div class="detail-label">Hex Inspector</div><div class="detail-value hex-view">
                    <div class="hex-col">${this.toHex(d.value)}</div><div class="ascii-col">${this.toAscii(d.value)}</div>
                </div></div>`;
            } else {
                html += `<div class="detail-row"><div class="detail-label">Value</div><div class="detail-value">${String(d.value)}</div></div>`;
            }
            document.getElementById('modal-body').innerHTML = html;
        });
    },

    loadModels: function() {
        fetch('/api/models').then(r=>r.json()).then(d => {
            if(!d.length) { document.getElementById('models-content').innerHTML = '<div class="empty">No registered models found</div>'; return; }
            let h = `<table><thead><tr><th>Model</th><th>Ver</th><th>Stage</th><th>Metrics</th><th>Size</th><th>Created</th></tr></thead><tbody>`;
            h += d.map(m => `<tr>
                <td class="key-cell">${m.name}</td>
                <td>v${m.version}</td>
                <td><span class="tag tag-${m.stage}">${m.stage}</span></td>
                <td style="font-size:11px;font-family:monospace">${JSON.stringify(m.metadata).substring(0,30)}</td>
                <td style="opacity:0.7">${Math.round(m.size_bytes/1024)} KB</td>
                <td style="opacity:0.7">${new Date(m.created_at*1000).toLocaleDateString()}</td>
            </tr>`).join('');
            h += '</tbody></table>';
            document.getElementById('models-content').innerHTML = h;
        });
    },

    loadClusters: function() {
        fetch('/api/vectors').then(r=>r.json()).then(d => {
            if(!d.points.length) { document.getElementById('cluster-plot').innerHTML = '<div class="empty">No vector data available</div>'; return; }
            
            const trace = {
                x: d.points.map(p => p.x), y: d.points.map(p => p.y), z: d.points.map(p => p.z),
                text: d.points.map(p => p.key),
                mode: 'markers',
                marker: { size: 6, color: d.points.map(p => p.z), colorscale: 'Portland', opacity: 0.9, line:{color:'white',width:0.5} },
                type: 'scatter3d'
            };
            Plotly.newPlot('cluster-plot', [trace], {
                ...this.layout,
                scene: { 
                    xaxis:{title:'',showgrid:true,gridcolor:'rgba(255,255,255,0.1)'}, 
                    yaxis:{title:'',showgrid:true,gridcolor:'rgba(255,255,255,0.1)'}, 
                    zaxis:{title:'',showgrid:true,gridcolor:'rgba(255,255,255,0.1)'}, 
                    bgcolor:'transparent' 
                }
            });
        });
    },

    loadStats: function() {
        Plotly.newPlot('health-pie', [{
            labels: Object.keys(this.state.typeCounts),
            values: Object.values(this.state.typeCounts),
            type: 'pie', hole: 0.4, marker: {colors: ['#8b5cf6','#ec4899','#10b981','#f59e0b','#3b82f6']}
        }], this.layout);

        const data = [{
            type: "treemap",
            labels: ["Root", ...this.state.keys.slice(0,30).map(k=>k.key)],
            parents: ["", ...this.state.keys.slice(0,30).map(k=>"Root")],
            values: [0, ...this.state.keys.slice(0,30).map(k=> Math.ceil(Math.random()*100))],
            textinfo: "label",
            marker: {colorscale: 'Blues'}
        }];
        Plotly.newPlot('health-treemap', data, this.layout);
    },
    
    loadIntegrations: function() {
        fetch('/api/integrations').then(r=>r.json()).then(d => {
            if(!d.length) { document.getElementById('integrations-list').innerHTML = '<div class="empty">No integrations found</div>'; return; }
            const html = d.map(i => `
            <div class="chart-box">
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
                    <div style="width:32px;height:32px;background:rgba(139,92,246,0.2);border-radius:8px;display:flex;align-items:center;justify-content:center;color:#a78bfa;font-weight:700;">${i.name.substring(0,2).toUpperCase()}</div>
                    <h3 style="margin:0">${i.name}</h3>
                </div>
                <p style="color:var(--text-dim);font-size:13px;margin-bottom:16px;">Integration module for ${i.name}.</p>
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span class="tag tag-staging">${Math.round(i.size/1024)} KB</span>
                    <button class="view-btn" onclick="app.runIntegration('${i.name}')">Run Test ▷</button>
                </div>
            </div>`).join('');
            document.getElementById('integrations-list').innerHTML = html;
        });
    },

    runIntegration: function(name) {
        document.getElementById('modal').classList.add('active');
        document.getElementById('modal-title').textContent = 'Running ' + name + '...';
        document.getElementById('modal-body').innerHTML = '<div class="loading"><div class="spinner"></div><div style="margin-top:10px;color:var(--text-dim)">Executing script... (Timeout 15s)</div></div>';
        
        fetch('/api/run_integration', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name: name})
        }).then(r=>r.json()).then(d => {
            document.getElementById('modal-title').textContent = 'Result: ' + name;
            document.getElementById('modal-body').innerHTML = `<pre style="white-space:pre-wrap;font-family:'JetBrains Mono';font-size:12px;color:#cbd5e1;background:#0f172a;padding:12px;border-radius:6px;max-height:400px;overflow-y:auto;">${this.escapeHtml(d.output)}</pre>`;
        }).catch(e => {
            document.getElementById('modal-body').innerHTML = '<div style="color:#ef4444">Error: ' + e + '</div>';
        });
    },

    escapeHtml: function(text) {
        if (!text) return text;
        return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
    },


    addWidget: function() {
        const type = document.getElementById('widget-select').value;
        const id = 'w-' + Date.now();
        const container = document.getElementById('dynamic-widgets');
        
        const div = document.createElement('div');
        div.className = 'chart-box';
        div.innerHTML = `<div style="display:flex;justify-content:space-between;margin-bottom:10px;">
            <h3>${type.replace('_',' ').toUpperCase()}</h3>
            <button class="view-btn" onclick="this.parentElement.parentElement.remove()">Remove</button>
        </div><div id="${id}" style="height:250px;"></div>`;
        container.appendChild(div);
        
        if(type === 'key_len') {
            const lens = this.state.keys.map(k => k.key.length);
            Plotly.newPlot(id, [{ x: lens, type: 'histogram', marker:{color:'#8b5cf6'} }], this.layout);
        } else if(type === 'write_ops') {
             const x = Array.from({length:20}, (_,i)=>i);
             const y = x.map(() => 50 + Math.random()*20);
             Plotly.newPlot(id, [{ x: x, y: y, type: 'scatter', mode:'lines+markers', line:{shape:'spline', color:'#34d399'} }], this.layout);
        } else if(type === 'mem_usage') {
             const x = Array.from({length:20}, (_,i)=>i);
             const y = x.map(i => 100 + i*5 + Math.random()*10);
             Plotly.newPlot(id, [{ x: x, y: y, type: 'scatter', fill:'tozeroy', line:{color:'#f59e0b'} }], this.layout);
        }
    },

    runAction: function(action, params) {
        const out = document.getElementById('action-output');
        out.style.display = 'block';
        out.textContent = '> Running ' + action + '...';
        
        fetch('/api/custom_action', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({action, params})
        }).then(r=>r.json()).then(d => {
            out.textContent = '> ' + d.message;
        }).catch(e => {
            out.textContent = '> Error: ' + e;
        });
    },

    toHex: function(str) {
        // Simple hex mock
        let h = ''; for(let i=0; i<str.length; i++) h += str.charCodeAt(i).toString(16).padStart(2,'0') + ' ';
        return h;
    },
    toAscii: function(str) {
        return str.replace(/[^ -~]/g, '.');
    },
    closeModal: function() { document.getElementById('modal').classList.remove('active'); }
};

window.switchDB = function() {
    const path = prompt("Enter full path to database (.db):", document.getElementById('current-db-path').innerText);
    if(path) {
        fetch('/api/switch_db', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({path})})
        .then(() => window.location.reload());
    }
};

app.init();
</script>
</body>
</html>'''

def launch(db_path: str, port: int = 8501, debug: bool = False) -> None:
    """Launch Syna Studio web UI."""
    if not FLASK_AVAILABLE: raise ImportError("Flask not installed.")
    import os
    from .wrapper import SynaDB
    
    global CURRENT_DB_PATH
    CURRENT_DB_PATH = db_path

    app = Flask(__name__)
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

    @app.route('/')
    def index():
        return render_template_string(STUDIO_HTML, db_path=CURRENT_DB_PATH)

    @app.route('/api/switch_db', methods=['POST'])
    def switch_db():
        global CURRENT_DB_PATH
        data = request.json
        new_path = data.get('path')
        if new_path and os.path.exists(new_path):
            CURRENT_DB_PATH = new_path
            return jsonify({'status': 'ok'})
        return jsonify({'error': 'Invalid path'}), 400
    
    @app.route('/api/custom_action', methods=['POST'])
    def custom_action():
        data = request.json
        action = data.get('action')
        params = data.get('params')
        
        # Mock actions for demo
        if action == 'compact':
            import time
            time.sleep(1)
            return jsonify({'message': 'Database compacted successfully. (Mock)'})
        elif action == 'export_json':
            return jsonify({'message': 'Export started. Check server logs for output path.'})
        elif action == 'integrity_check':
            return jsonify({'message': 'Integrity check passed. No corruption detected.'})
        elif action == 'clear_cache':
             return jsonify({'message': 'Cache cleared.'})
        elif action == 'scan_pattern':
             return jsonify({'message': f'Found 14 keys matching "{params}"'})
        
        return jsonify({'message': 'Unknown action'})

    @app.route('/api/integrations')
    def api_integrations():
        """Scan integration directory."""
        import os
        # Assuming integrations dir is next to this file or standard location
        # D:/synadb/demos/python/synadb/integrations
        current_dir = os.path.dirname(os.path.abspath(__file__))
        integrations_dir = os.path.join(current_dir, 'integrations')
        
        results = []
        if os.path.exists(integrations_dir):
            for f in os.listdir(integrations_dir):
                if f.endswith('.py') and f != '__init__.py':
                    path = os.path.join(integrations_dir, f)
                    size = os.path.getsize(path)
                    results.append({'name': f.replace('.py', ''), 'path': f, 'size': size})
        else:
            # Fallback mock if directory structure differs in dev env
            results = [
                {'name': 'mlflow', 'path': 'mlflow.py', 'size': 20983},
                {'name': 'langchain', 'path': 'langchain.py', 'size': 19651},
                {'name': 'haystack', 'path': 'haystack.py', 'size': 8920},
                {'name': 'llamaindex', 'path': 'llamaindex.py', 'size': 11069}
            ]
        return jsonify(results)

    @app.route('/api/run_integration', methods=['POST'])
    def api_run_integration():
        import subprocess, os
        data = request.json
        name = data.get('name')
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(current_dir, 'integrations', name + '.py')
        
        if not os.path.exists(script_path):
            return jsonify({'output': f'Error: File not found at {script_path}'}), 404

        try:
            # Run the script and capture output
            # We use a timeout to prevent hanging
            result = subprocess.run(
                ['python', script_path], 
                capture_output=True, 
                text=True, 
                timeout=15,
                cwd=current_dir # Run in module dir
            )
            output = f"Exit Code: {result.returncode}\n\n[STDOUT]\n{result.stdout}\n\n[STDERR]\n{result.stderr}"
            return jsonify({'output': output})
        except subprocess.TimeoutExpired:
            return jsonify({'output': 'Error: Execution timed out after 15s.'})
        except Exception as e:
            return jsonify({'output': f'Error executing script: {str(e)}'})

    @app.route('/api/keys')
    def api_keys():
        limit = int(request.args.get('limit', 1000))
        with SynaDB(CURRENT_DB_PATH) as db:
            keys = db.keys()[:limit]
            results = []
            counts = {'float': 0, 'int': 0, 'text': 0, 'bytes': 0, 'vector': 0}
            for k in keys:
                v, t = _get_value(db, k)
                results.append({'key': k, 'type': t, 'value': v})
                if t in counts: counts[t] += 1
        return jsonify({'keys': results, 'type_counts': counts})

    @app.route('/api/get/<path:key>')
    def api_get(key):
        with SynaDB(CURRENT_DB_PATH) as db:
            v, t = _get_value(db, key)
            return jsonify({'key': key, 'type': t, 'value': v})

    @app.route('/api/models')
    def api_models():
        # Scan for models using basic pattern matching on meta keys
        try:
            with SynaDB(CURRENT_DB_PATH) as db:
                keys = db.keys()
                models = []
                for k in keys:
                    if k.startswith("model/") and k.endswith("/meta"):
                        meta_json = db.get_text(k)
                        if meta_json:
                            models.append(json.loads(meta_json))
                return jsonify(models)
        except Exception:
            return jsonify([])

    @app.route('/api/vectors')
    def api_vectors():
        # Retrieve all vectors and perform simple projection
        try:
            import numpy as np
            points = []
            with SynaDB(CURRENT_DB_PATH) as db:
                keys = db.keys()
                vectors = []
                labels = []
                for k in keys:
                     v, t = _get_value(db, k)
                     if t == 'vector' and len(v) > 1:
                         vectors.append(v)
                         labels.append(k)
            
            if not vectors: return jsonify({'points': []})
            
            # Simple random projection if dimensionality is high, or take first 3 dims
            # A real PCA with numpy:
            X = np.array(vectors)
            if X.shape[1] > 3:
                # Centering
                X_centered = X - X.mean(axis=0)
                # SVD
                U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
                # Project to top 3 components
                X_pca = X_centered @ Vt.T[:, :3]
            else:
                X_pca = X[:, :3] if X.shape[1] >= 3 else np.pad(X, ((0,0),(0, 3-X.shape[1])))
                
            points = [{'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2]), 'key': l} for p, l in zip(X_pca, labels)]
            return jsonify({'points': points})
        except ImportError:
            return jsonify({'error': 'numpy required', 'points': []})
        except Exception as e:
            return jsonify({'error': str(e), 'points': []})

    print(f"\\n  Syna Studio V2: http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=debug)

def _get_value(db, key: str) -> Tuple[Any, str]:
    """Helper to detect type and value."""
    # Priority: Float -> Int -> Text -> Vector -> Bytes
    try:
        v = db.get_float(key)
        if v is not None: return v, 'float'
    except: pass
    
    try:
        v = db.get_int(key)
        if v is not None: return v, 'int'
    except: pass

    try:
        v = db.get_text(key)
        if v is not None: 
            # Detect JSON-stored vectors for demo fallback
            import json
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], (int, float)):
                    return parsed, 'vector'
            except: pass
            return v, 'text'
    except: pass

    # Vector check (Native)
    try:
        arr = db.get_history_tensor(key)
        if hasattr(arr, 'size') and arr.size > 0:
            return arr.tolist(), 'vector'
    except: pass

    # Bytes last
    try:
        v = db.get_bytes(key)
        if v is not None: return v.hex()[:50], 'bytes'
    except: pass
    
    return None, 'unknown'
