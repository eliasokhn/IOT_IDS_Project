"""
demo_server.py — IoT IDS Live Demo
===================================
Loads all 6 trained models and serves a web UI for real-time predictions.

Run:
    python demo_server.py

Then open http://localhost:7860 (auto-opens in your browser).
"""

import logging
import webbrowser
from threading import Timer

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

TASKS = ["binary", "8class", "34class"]
MODEL_NAMES = ["lr", "gb"]
MODELS_DIR = "models"
PORT = 7860

predictors: dict = {}

app = FastAPI(title="IoT IDS Demo", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class PredictRequest(BaseModel):
    features: dict


@app.on_event("startup")
async def _startup():
    log.info("Loading all 6 models...")
    try:
        from src.serving.predictor import Predictor
    except ImportError as e:
        log.error(f"Import error: {e}")
        return
    for task in TASKS:
        for model_name in MODEL_NAMES:
            key = f"{model_name}_{task}"
            try:
                predictors[key] = Predictor.load(MODELS_DIR, task=task, model_name=model_name)
                log.info(f"  OK  {key}")
            except FileNotFoundError:
                log.warning(f"  MISSING  {key} — run training first")


@app.post("/predict_all")
def predict_all(body: PredictRequest):
    out = {}
    for task in TASKS:
        for model_name in MODEL_NAMES:
            key = f"{model_name}_{task}"
            p = predictors.get(key)
            if p is None:
                out[key] = {"error": "Model not loaded — run training first"}
                continue
            try:
                out[key] = p.predict(body.features)
            except Exception as e:
                log.error(f"Prediction error {key}: {e}")
                out[key] = {"error": str(e)}
    return out


@app.get("/status")
def status():
    return {k: "loaded" for k in predictors} | {
        f"{m}_{t}": "missing"
        for t in TASKS for m in MODEL_NAMES
        if f"{m}_{t}" not in predictors
    }


@app.get("/", response_class=HTMLResponse)
def root():
    return _HTML


# ──────────────────────────────────────────────────────────────────────────────
#  Embedded HTML / CSS / JS  (single-file, no external dependencies)
# ──────────────────────────────────────────────────────────────────────────────
_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>IoT IDS — Live Demo</title>
<style>
  :root {
    --bg:      #0d1117;
    --surface: #161b22;
    --card:    #1c2128;
    --border:  #30363d;
    --blue:    #58a6ff;
    --green:   #3fb950;
    --red:     #f85149;
    --orange:  #f0883e;
    --yellow:  #d29922;
    --purple:  #bc8cff;
    --text:    #e6edf3;
    --sub:     #8b949e;
    --radius:  10px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 14px; min-height: 100vh;
  }

  /* ── Header ─────────────────────────────────────────── */
  header {
    background: linear-gradient(135deg, #0d1117 0%, #1a2744 100%);
    border-bottom: 1px solid var(--border);
    padding: 24px 40px;
    display: flex; align-items: center; gap: 20px;
  }
  .logo {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, var(--blue), var(--purple));
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px; flex-shrink: 0;
  }
  header h1 { font-size: 22px; font-weight: 700; letter-spacing: -0.3px; }
  header p  { font-size: 13px; color: var(--sub); margin-top: 3px; }
  .status-dots {
    margin-left: auto; display: flex; gap: 6px; align-items: center;
    font-size: 12px; color: var(--sub);
  }
  .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--sub);
  }
  .dot.ok  { background: var(--green); }
  .dot.err { background: var(--red); }

  /* ── Main layout ─────────────────────────────────────── */
  main { max-width: 1300px; margin: 0 auto; padding: 28px 32px 60px; }

  /* ── Presets ─────────────────────────────────────────── */
  .presets-bar {
    display: flex; gap: 10px; flex-wrap: wrap;
    margin-bottom: 24px; align-items: center;
  }
  .presets-label {
    font-size: 12px; color: var(--sub); font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.6px;
    margin-right: 4px;
  }
  .preset-btn {
    padding: 7px 16px; border-radius: 20px; border: 1px solid var(--border);
    background: var(--surface); color: var(--text);
    font-size: 13px; cursor: pointer; transition: all 0.15s;
    font-weight: 500;
  }
  .preset-btn:hover { border-color: var(--blue); color: var(--blue); }
  .preset-btn.benign:hover  { border-color: var(--green); color: var(--green); }
  .preset-btn.attack:hover  { border-color: var(--red);   color: var(--red);   }
  .preset-btn.recon:hover   { border-color: var(--yellow); color: var(--yellow); }
  .preset-btn.brute:hover   { border-color: var(--orange); color: var(--orange); }
  .preset-btn.mirai:hover   { border-color: var(--purple); color: var(--purple); }

  /* ── Feature groups grid ─────────────────────────────── */
  .groups-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 22px;
  }

  .group {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
  }
  .group-title {
    font-size: 11px; font-weight: 700; color: var(--sub);
    text-transform: uppercase; letter-spacing: 0.8px;
    margin-bottom: 12px; padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }
  .field-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 12px; }
  .field-grid.three-col { grid-template-columns: 1fr 1fr 1fr; }
  .field-grid.four-col  { grid-template-columns: 1fr 1fr 1fr 1fr; }
  .field-grid.five-col  { grid-template-columns: repeat(5, 1fr); }
  .field-grid.one-col   { grid-template-columns: 1fr; }

  label.field { display: flex; flex-direction: column; gap: 4px; }
  label.field span {
    font-size: 11px; color: var(--sub); font-weight: 500;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  input[type=number], select {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 6px; color: var(--text);
    padding: 6px 8px; font-size: 13px; width: 100%;
    transition: border-color 0.15s;
    -moz-appearance: textfield;
  }
  input[type=number]::-webkit-inner-spin-button { opacity: 0.3; }
  input[type=number]:focus, select:focus {
    outline: none; border-color: var(--blue);
  }
  select option { background: var(--card); }

  /* toggle for binary flags */
  .toggle-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px;
  }
  .toggle-grid.seven { grid-template-columns: repeat(4, 1fr); }
  .toggle-item {
    display: flex; flex-direction: column; align-items: center; gap: 4px;
  }
  .toggle-item span { font-size: 10px; color: var(--sub); text-align: center; }
  .toggle-wrap {
    position: relative; width: 38px; height: 20px; cursor: pointer;
  }
  .toggle-wrap input { display: none; }
  .slider {
    position: absolute; inset: 0;
    background: var(--border); border-radius: 10px;
    transition: background 0.2s;
  }
  .slider::after {
    content: ''; position: absolute;
    width: 14px; height: 14px; border-radius: 50%;
    background: #fff; top: 3px; left: 3px;
    transition: transform 0.2s;
  }
  .toggle-wrap input:checked + .slider { background: var(--blue); }
  .toggle-wrap input:checked + .slider::after { transform: translateX(18px); }

  /* ── Predict button ──────────────────────────────────── */
  .predict-row {
    display: flex; justify-content: center; margin-bottom: 28px;
  }
  #predict-btn {
    padding: 14px 60px;
    background: linear-gradient(135deg, #1d6ae5, #7c3aed);
    border: none; border-radius: 30px;
    color: #fff; font-size: 16px; font-weight: 700;
    cursor: pointer; letter-spacing: 0.3px;
    transition: opacity 0.2s, transform 0.1s;
    box-shadow: 0 4px 20px rgba(88, 166, 255, 0.3);
  }
  #predict-btn:hover   { opacity: 0.9; }
  #predict-btn:active  { transform: scale(0.98); }
  #predict-btn:disabled { opacity: 0.5; cursor: not-allowed; }

  /* ── Results ─────────────────────────────────────────── */
  #results { display: none; }
  .results-title {
    text-align: center; font-size: 13px; color: var(--sub);
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px;
    margin-bottom: 16px;
  }
  .results-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
  }
  .model-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 18px; position: relative;
    overflow: hidden; transition: border-color 0.2s;
  }
  .model-card.benign { border-color: rgba(63, 185, 80, 0.5); }
  .model-card.attack { border-color: rgba(248, 81, 73, 0.5); }
  .card-accent {
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
  }
  .benign .card-accent { background: var(--green); }
  .attack .card-accent { background: var(--red); }

  .card-header {
    display: flex; justify-content: space-between; align-items: flex-start;
    margin-bottom: 14px;
  }
  .card-model-badge {
    font-size: 11px; font-weight: 700; letter-spacing: 0.6px;
    padding: 3px 10px; border-radius: 20px; text-transform: uppercase;
  }
  .badge-lr { background: rgba(88, 166, 255, 0.15); color: var(--blue); }
  .badge-gb { background: rgba(188, 140, 255, 0.15); color: var(--purple); }
  .card-task { font-size: 12px; color: var(--sub); font-weight: 600; }

  .card-label {
    font-size: 22px; font-weight: 800;
    margin-bottom: 4px; letter-spacing: -0.5px;
    word-break: break-word;
  }
  .benign .card-label { color: var(--green); }
  .attack .card-label { color: var(--red); }

  .card-confidence {
    font-size: 13px; color: var(--sub); margin-bottom: 12px;
  }
  .conf-bar {
    height: 5px; background: var(--border); border-radius: 3px; margin-bottom: 14px;
    overflow: hidden;
  }
  .conf-fill {
    height: 100%; border-radius: 3px; transition: width 0.6s ease;
  }
  .benign .conf-fill { background: var(--green); }
  .attack .conf-fill { background: var(--red); }

  .top-probs {
    display: flex; flex-direction: column; gap: 4px;
  }
  .prob-row {
    display: flex; align-items: center; gap: 8px; font-size: 11px;
  }
  .prob-name {
    color: var(--sub); flex: 1; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap;
  }
  .prob-bar-wrap {
    width: 60px; height: 4px; background: var(--border);
    border-radius: 2px; overflow: hidden;
  }
  .prob-bar-fill { height: 100%; background: var(--blue); border-radius: 2px; }
  .prob-pct { width: 38px; text-align: right; color: var(--text); font-weight: 600; }

  .card-error {
    color: var(--red); font-size: 12px; font-style: italic;
  }

  /* ── Loading spinner ─────────────────────────────────── */
  #spinner {
    display: none; text-align: center; margin: 12px 0;
    color: var(--sub); font-size: 13px;
  }
  .spin {
    display: inline-block; width: 18px; height: 18px;
    border: 2px solid var(--border); border-top-color: var(--blue);
    border-radius: 50%; animation: spin 0.7s linear infinite;
    vertical-align: middle; margin-right: 8px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  @media (max-width: 900px) {
    .groups-grid   { grid-template-columns: 1fr 1fr; }
    .results-grid  { grid-template-columns: 1fr 1fr; }
  }
  @media (max-width: 600px) {
    .groups-grid   { grid-template-columns: 1fr; }
    .results-grid  { grid-template-columns: 1fr; }
    header { padding: 16px 20px; }
    main   { padding: 16px; }
  }
</style>
</head>
<body>

<!-- ── Header ──────────────────────────────────────────── -->
<header>
  <div class="logo">🛡️</div>
  <div>
    <h1>IoT Intrusion Detection System</h1>
    <p>CICIoT2023 · Logistic Regression &amp; Gradient Boosting · Binary / 8-class / 34-class</p>
  </div>
  <div class="status-dots" id="status-dots">
    <span>Models:</span>
    <span id="dot-lr-binary"  class="dot" title="LR Binary"></span>
    <span id="dot-lr-8class"  class="dot" title="LR 8-class"></span>
    <span id="dot-lr-34class" class="dot" title="LR 34-class"></span>
    <span id="dot-gb-binary"  class="dot" title="GB Binary"></span>
    <span id="dot-gb-8class"  class="dot" title="GB 8-class"></span>
    <span id="dot-gb-34class" class="dot" title="GB 34-class"></span>
  </div>
</header>

<!-- ── Main ────────────────────────────────────────────── -->
<main>

  <!-- Presets -->
  <div class="presets-bar">
    <span class="presets-label">Scenario:</span>
    <button class="preset-btn benign" onclick="loadPreset('benign')">✅ Normal Traffic</button>
    <button class="preset-btn attack" onclick="loadPreset('ddos_syn')">⚡ DDoS SYN Flood</button>
    <button class="preset-btn attack" onclick="loadPreset('ddos_udp')">🌊 DDoS UDP Flood</button>
    <button class="preset-btn attack" onclick="loadPreset('dos_http')">💥 DoS HTTP Flood</button>
    <button class="preset-btn recon"  onclick="loadPreset('portscan')">🔍 Port Scan</button>
    <button class="preset-btn mirai"  onclick="loadPreset('mirai')">🤖 Mirai Botnet</button>
    <button class="preset-btn brute"  onclick="loadPreset('bruteforce')">🔑 Brute Force</button>
  </div>

  <!-- Feature groups -->
  <div class="groups-grid">

    <!-- Protocol -->
    <div class="group">
      <div class="group-title">Protocol</div>
      <div class="field-grid one-col">
        <label class="field">
          <span>Protocol Type</span>
          <select id="Protocol Type">
            <option value="0">0 — HOPOPT</option>
            <option value="1">1 — ICMP</option>
            <option value="2">2 — IGMP</option>
            <option value="6" selected>6 — TCP</option>
            <option value="17">17 — UDP</option>
            <option value="47">47 — GRE</option>
          </select>
        </label>
      </div>
    </div>

    <!-- Flow Statistics -->
    <div class="group">
      <div class="group-title">Flow Statistics</div>
      <div class="field-grid">
        <label class="field"><span>Header Length</span>
          <input type="number" id="Header_Length" value="32" min="0"></label>
        <label class="field"><span>Time To Live</span>
          <input type="number" id="Time_To_Live" value="64" min="0" max="255"></label>
        <label class="field"><span>Rate (pkt/s)</span>
          <input type="number" id="Rate" value="150" min="0" step="any"></label>
        <label class="field"><span>Number</span>
          <input type="number" id="Number" value="20" min="0"></label>
        <label class="field"><span>IAT (s)</span>
          <input type="number" id="IAT" value="0.05" min="0" step="any"></label>
        <label class="field"><span>IPv</span>
          <input type="number" id="IPv" value="1" min="0"></label>
      </div>
    </div>

    <!-- Packet Sizes -->
    <div class="group">
      <div class="group-title">Packet Size Statistics</div>
      <div class="field-grid">
        <label class="field"><span>Tot Sum</span>
          <input type="number" id="Tot sum" value="5000" min="0"></label>
        <label class="field"><span>Tot Size</span>
          <input type="number" id="Tot size" value="5000" min="0"></label>
        <label class="field"><span>Min</span>
          <input type="number" id="Min" value="40" min="0"></label>
        <label class="field"><span>Max</span>
          <input type="number" id="Max" value="1460" min="0"></label>
        <label class="field"><span>AVG</span>
          <input type="number" id="AVG" value="500" min="0" step="any"></label>
        <label class="field"><span>Std</span>
          <input type="number" id="Std" value="300" min="0" step="any"></label>
      </div>
    </div>

    <!-- TCP Flags -->
    <div class="group">
      <div class="group-title">TCP Flag Numbers</div>
      <div class="toggle-grid seven">
        <div class="toggle-item">
          <span>SYN</span>
          <label class="toggle-wrap">
            <input type="checkbox" id="syn_flag_number">
            <div class="slider"></div>
          </label>
        </div>
        <div class="toggle-item">
          <span>ACK</span>
          <label class="toggle-wrap">
            <input type="checkbox" id="ack_flag_number" checked>
            <div class="slider"></div>
          </label>
        </div>
        <div class="toggle-item">
          <span>PSH</span>
          <label class="toggle-wrap">
            <input type="checkbox" id="psh_flag_number">
            <div class="slider"></div>
          </label>
        </div>
        <div class="toggle-item">
          <span>FIN</span>
          <label class="toggle-wrap">
            <input type="checkbox" id="fin_flag_number">
            <div class="slider"></div>
          </label>
        </div>
        <div class="toggle-item">
          <span>RST</span>
          <label class="toggle-wrap">
            <input type="checkbox" id="rst_flag_number">
            <div class="slider"></div>
          </label>
        </div>
        <div class="toggle-item">
          <span>ECE</span>
          <label class="toggle-wrap">
            <input type="checkbox" id="ece_flag_number">
            <div class="slider"></div>
          </label>
        </div>
        <div class="toggle-item">
          <span>CWR</span>
          <label class="toggle-wrap">
            <input type="checkbox" id="cwr_flag_number">
            <div class="slider"></div>
          </label>
        </div>
      </div>
    </div>

    <!-- Flag Counts -->
    <div class="group">
      <div class="group-title">Flag Counts</div>
      <div class="field-grid">
        <label class="field"><span>ACK Count</span>
          <input type="number" id="ack_count" value="15" min="0"></label>
        <label class="field"><span>SYN Count</span>
          <input type="number" id="syn_count" value="1" min="0"></label>
        <label class="field"><span>FIN Count</span>
          <input type="number" id="fin_count" value="1" min="0"></label>
        <label class="field"><span>RST Count</span>
          <input type="number" id="rst_count" value="0" min="0"></label>
      </div>
    </div>

    <!-- Protocol Indicators -->
    <div class="group">
      <div class="group-title">Protocol Indicators</div>
      <div class="toggle-grid">
        <div class="toggle-item"><span>HTTP</span>
          <label class="toggle-wrap"><input type="checkbox" id="HTTP"><div class="slider"></div></label></div>
        <div class="toggle-item"><span>HTTPS</span>
          <label class="toggle-wrap"><input type="checkbox" id="HTTPS" checked><div class="slider"></div></label></div>
        <div class="toggle-item"><span>DNS</span>
          <label class="toggle-wrap"><input type="checkbox" id="DNS"><div class="slider"></div></label></div>
        <div class="toggle-item"><span>Telnet</span>
          <label class="toggle-wrap"><input type="checkbox" id="Telnet"><div class="slider"></div></label></div>
        <div class="toggle-item"><span>SMTP</span>
          <label class="toggle-wrap"><input type="checkbox" id="SMTP"><div class="slider"></div></label></div>
        <div class="toggle-item"><span>SSH</span>
          <label class="toggle-wrap"><input type="checkbox" id="SSH"><div class="slider"></div></label></div>
        <div class="toggle-item"><span>IRC</span>
          <label class="toggle-wrap"><input type="checkbox" id="IRC"><div class="slider"></div></label></div>
        <div class="toggle-item"><span>TCP</span>
          <label class="toggle-wrap"><input type="checkbox" id="TCP" checked><div class="slider"></div></label></div>
        <div class="toggle-item"><span>UDP</span>
          <label class="toggle-wrap"><input type="checkbox" id="UDP"><div class="slider"></div></label></div>
        <div class="toggle-item"><span>DHCP</span>
          <label class="toggle-wrap"><input type="checkbox" id="DHCP"><div class="slider"></div></label></div>
        <div class="toggle-item"><span>ARP</span>
          <label class="toggle-wrap"><input type="checkbox" id="ARP"><div class="slider"></div></label></div>
        <div class="toggle-item"><span>ICMP</span>
          <label class="toggle-wrap"><input type="checkbox" id="ICMP"><div class="slider"></div></label></div>
        <div class="toggle-item"><span>IGMP</span>
          <label class="toggle-wrap"><input type="checkbox" id="IGMP"><div class="slider"></div></label></div>
        <div class="toggle-item"><span>LLC</span>
          <label class="toggle-wrap"><input type="checkbox" id="LLC"><div class="slider"></div></label></div>
      </div>
    </div>

  </div><!-- /groups-grid -->

  <!-- Predict -->
  <div class="predict-row">
    <button id="predict-btn" onclick="runPredict()">🔍 Analyze Traffic — All 6 Models</button>
  </div>
  <div id="spinner"><span class="spin"></span>Running inference on all 6 models…</div>

  <!-- Results -->
  <div id="results">
    <div class="results-title">Model Predictions</div>
    <div class="results-grid" id="results-grid"></div>
  </div>

</main>

<script>
// ─── Feature IDs (maps element id → feature name for the API) ────────────────
const NUM_FIELDS = [
  "Header_Length", "Time_To_Live", "Rate", "Number", "IAT",
  "IPv", "Tot sum", "Tot size", "Min", "Max", "AVG", "Std",
  "ack_count", "syn_count", "fin_count", "rst_count"
];
const TOGGLE_FIELDS = [
  "fin_flag_number", "syn_flag_number", "rst_flag_number",
  "psh_flag_number", "ack_flag_number", "ece_flag_number", "cwr_flag_number",
  "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
  "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "LLC"
];

// ─── Presets ──────────────────────────────────────────────────────────────────
const PRESETS = {
  benign: {
    "Protocol Type": 6,
    Header_Length: 32, Time_To_Live: 64, Rate: 150, Number: 20, IAT: 0.05,
    "Tot sum": 9800, "Tot size": 9800, Min: 40, Max: 1460, AVG: 490, Std: 380,
    IPv: 1,
    fin_flag_number: 0, syn_flag_number: 0, rst_flag_number: 0,
    psh_flag_number: 1, ack_flag_number: 1, ece_flag_number: 0, cwr_flag_number: 0,
    ack_count: 18, syn_count: 1, fin_count: 1, rst_count: 0,
    HTTP: 0, HTTPS: 1, DNS: 0, Telnet: 0, SMTP: 0, SSH: 0, IRC: 0,
    TCP: 1, UDP: 0, DHCP: 0, ARP: 0, ICMP: 0, IGMP: 0, LLC: 0
  },
  ddos_syn: {
    "Protocol Type": 6,
    Header_Length: 20, Time_To_Live: 128, Rate: 95000, Number: 9500, IAT: 0.00001,
    "Tot sum": 190000, "Tot size": 190000, Min: 20, Max: 20, AVG: 20, Std: 0,
    IPv: 1,
    fin_flag_number: 0, syn_flag_number: 1, rst_flag_number: 0,
    psh_flag_number: 0, ack_flag_number: 0, ece_flag_number: 0, cwr_flag_number: 0,
    ack_count: 0, syn_count: 9500, fin_count: 0, rst_count: 500,
    HTTP: 0, HTTPS: 0, DNS: 0, Telnet: 0, SMTP: 0, SSH: 0, IRC: 0,
    TCP: 1, UDP: 0, DHCP: 0, ARP: 0, ICMP: 0, IGMP: 0, LLC: 0
  },
  ddos_udp: {
    "Protocol Type": 17,
    Header_Length: 8, Time_To_Live: 64, Rate: 80000, Number: 8000, IAT: 0.0000125,
    "Tot sum": 6400000, "Tot size": 6400000, Min: 500, Max: 1400, AVG: 800, Std: 200,
    IPv: 1,
    fin_flag_number: 0, syn_flag_number: 0, rst_flag_number: 0,
    psh_flag_number: 0, ack_flag_number: 0, ece_flag_number: 0, cwr_flag_number: 0,
    ack_count: 0, syn_count: 0, fin_count: 0, rst_count: 0,
    HTTP: 0, HTTPS: 0, DNS: 0, Telnet: 0, SMTP: 0, SSH: 0, IRC: 0,
    TCP: 0, UDP: 1, DHCP: 0, ARP: 0, ICMP: 0, IGMP: 0, LLC: 0
  },
  dos_http: {
    "Protocol Type": 6,
    Header_Length: 32, Time_To_Live: 64, Rate: 4500, Number: 900, IAT: 0.00022,
    "Tot sum": 540000, "Tot size": 540000, Min: 60, Max: 1460, AVG: 600, Std: 380,
    IPv: 1,
    fin_flag_number: 0, syn_flag_number: 0, rst_flag_number: 0,
    psh_flag_number: 1, ack_flag_number: 1, ece_flag_number: 0, cwr_flag_number: 0,
    ack_count: 900, syn_count: 50, fin_count: 0, rst_count: 30,
    HTTP: 1, HTTPS: 0, DNS: 0, Telnet: 0, SMTP: 0, SSH: 0, IRC: 0,
    TCP: 1, UDP: 0, DHCP: 0, ARP: 0, ICMP: 0, IGMP: 0, LLC: 0
  },
  portscan: {
    "Protocol Type": 6,
    Header_Length: 20, Time_To_Live: 64, Rate: 200, Number: 500, IAT: 0.005,
    "Tot sum": 10000, "Tot size": 10000, Min: 20, Max: 20, AVG: 20, Std: 0,
    IPv: 1,
    fin_flag_number: 0, syn_flag_number: 1, rst_flag_number: 1,
    psh_flag_number: 0, ack_flag_number: 0, ece_flag_number: 0, cwr_flag_number: 0,
    ack_count: 0, syn_count: 500, fin_count: 0, rst_count: 480,
    HTTP: 0, HTTPS: 0, DNS: 0, Telnet: 0, SMTP: 0, SSH: 0, IRC: 0,
    TCP: 1, UDP: 0, DHCP: 0, ARP: 0, ICMP: 0, IGMP: 0, LLC: 0
  },
  mirai: {
    "Protocol Type": 17,
    Header_Length: 8, Time_To_Live: 64, Rate: 60000, Number: 6000, IAT: 0.0000167,
    "Tot sum": 4800000, "Tot size": 4800000, Min: 400, Max: 1000, AVG: 800, Std: 150,
    IPv: 1,
    fin_flag_number: 0, syn_flag_number: 0, rst_flag_number: 0,
    psh_flag_number: 0, ack_flag_number: 0, ece_flag_number: 0, cwr_flag_number: 0,
    ack_count: 0, syn_count: 0, fin_count: 0, rst_count: 0,
    HTTP: 0, HTTPS: 0, DNS: 0, Telnet: 0, SMTP: 0, SSH: 0, IRC: 0,
    TCP: 0, UDP: 1, DHCP: 0, ARP: 0, ICMP: 0, IGMP: 0, LLC: 0
  },
  bruteforce: {
    "Protocol Type": 6,
    Header_Length: 32, Time_To_Live: 64, Rate: 25, Number: 120, IAT: 0.2,
    "Tot sum": 24000, "Tot size": 24000, Min: 40, Max: 500, AVG: 200, Std: 110,
    IPv: 1,
    fin_flag_number: 1, syn_flag_number: 1, rst_flag_number: 0,
    psh_flag_number: 1, ack_flag_number: 1, ece_flag_number: 0, cwr_flag_number: 0,
    ack_count: 120, syn_count: 60, fin_count: 55, rst_count: 2,
    HTTP: 0, HTTPS: 0, DNS: 0, Telnet: 0, SMTP: 0, SSH: 1, IRC: 0,
    TCP: 1, UDP: 0, DHCP: 0, ARP: 0, ICMP: 0, IGMP: 0, LLC: 0
  }
};

// ─── Load preset values into all inputs ──────────────────────────────────────
function loadPreset(name) {
  const p = PRESETS[name];
  if (!p) return;

  // Protocol Type dropdown
  const proto = document.getElementById("Protocol Type");
  if (proto) proto.value = String(p["Protocol Type"] || 6);

  // Numeric fields
  for (const f of NUM_FIELDS) {
    const el = document.getElementById(f);
    if (el && f in p) el.value = p[f];
  }

  // Toggle fields
  for (const f of TOGGLE_FIELDS) {
    const el = document.getElementById(f);
    if (el) el.checked = !!p[f];
  }
}

// ─── Build feature dict from current form values ──────────────────────────────
function buildFeatures() {
  const features = {};

  // Protocol Type
  const proto = document.getElementById("Protocol Type");
  features["Protocol Type"] = parseFloat(proto ? proto.value : "6");

  // Numeric
  for (const f of NUM_FIELDS) {
    const el = document.getElementById(f);
    features[f] = el ? parseFloat(el.value) || 0 : 0;
  }

  // Toggles → 0 or 1
  for (const f of TOGGLE_FIELDS) {
    const el = document.getElementById(f);
    features[f] = el && el.checked ? 1 : 0;
  }

  return features;
}

// ─── Run inference ────────────────────────────────────────────────────────────
async function runPredict() {
  const btn     = document.getElementById("predict-btn");
  const spinner = document.getElementById("spinner");
  const results = document.getElementById("results");

  btn.disabled   = true;
  spinner.style.display = "block";
  results.style.display = "none";

  const features = buildFeatures();
  let data;
  try {
    const resp = await fetch("/predict_all", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features })
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    data = await resp.json();
  } catch (err) {
    alert("Error calling API: " + err.message);
    return;
  } finally {
    btn.disabled = false;
    spinner.style.display = "none";
  }

  displayResults(data);
}

// ─── Display results ──────────────────────────────────────────────────────────
const TASK_LABELS = { binary: "Binary (2 classes)", "8class": "8-Class", "34class": "34-Class" };
const MODEL_LABELS = { lr: "Logistic Regression", gb: "Gradient Boosting" };

function displayResults(data) {
  const grid    = document.getElementById("results-grid");
  const results = document.getElementById("results");
  grid.innerHTML = "";

  const ORDER = [
    ["lr", "binary"], ["lr", "8class"], ["lr", "34class"],
    ["gb", "binary"], ["gb", "8class"], ["gb", "34class"]
  ];

  for (const [model, task] of ORDER) {
    const key  = `${model}_${task}`;
    const info = data[key];
    grid.appendChild(buildCard(model, task, info));
  }

  results.style.display = "block";
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}

function buildCard(model, task, info) {
  const card = document.createElement("div");
  card.className = "model-card";

  if (!info || info.error) {
    card.innerHTML = `
      <div class="card-accent" style="background:var(--sub)"></div>
      <div class="card-header">
        <span class="card-model-badge badge-${model}">${model.toUpperCase()}</span>
        <span class="card-task">${TASK_LABELS[task]}</span>
      </div>
      <div class="card-error">⚠ ${info ? info.error : "No data"}</div>`;
    return card;
  }

  const isMalicious = info.is_malicious;
  card.classList.add(isMalicious ? "attack" : "benign");

  const label      = info.predicted_label || String(info.predicted_class);
  const probs      = info.probabilities || {};
  const topProbs   = getTopProbs(probs, task === "34class" ? 3 : 5);
  const confidence = probs[label] !== undefined ? probs[label] : 1.0;
  const confPct    = Math.round(confidence * 100);

  const probRows = topProbs.map(([name, val]) => {
    const pct  = Math.round(val * 100);
    const fill = Math.round(val * 100);
    return `<div class="prob-row">
      <span class="prob-name" title="${name}">${name}</span>
      <div class="prob-bar-wrap"><div class="prob-bar-fill" style="width:${fill}%"></div></div>
      <span class="prob-pct">${pct}%</span>
    </div>`;
  }).join("");

  card.innerHTML = `
    <div class="card-accent"></div>
    <div class="card-header">
      <span class="card-model-badge badge-${model}">${MODEL_LABELS[model]}</span>
      <span class="card-task">${TASK_LABELS[task]}</span>
    </div>
    <div class="card-label">${label}</div>
    <div class="card-confidence">Confidence: ${confPct}%</div>
    <div class="conf-bar"><div class="conf-fill" style="width:${confPct}%"></div></div>
    <div class="top-probs">${probRows}</div>`;

  return card;
}

function getTopProbs(probs, n) {
  return Object.entries(probs)
    .sort((a, b) => b[1] - a[1])
    .slice(0, n);
}

// ─── On load: check model status + load benign preset ─────────────────────────
window.addEventListener("load", async () => {
  loadPreset("benign");
  try {
    const resp = await fetch("/status");
    if (!resp.ok) return;
    const status = await resp.json();
    const keys = ["lr_binary","lr_8class","lr_34class","gb_binary","gb_8class","gb_34class"];
    for (const k of keys) {
      const dot = document.getElementById("dot-" + k);
      if (dot) dot.className = "dot " + (status[k] === "loaded" ? "ok" : "err");
    }
  } catch (_) {}
});
</script>
</body>
</html>"""

# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n  IoT IDS Demo  ->  http://localhost:{PORT}\n")
    Timer(1.5, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
