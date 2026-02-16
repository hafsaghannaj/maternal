import os
from flask import Blueprint, request, jsonify, Response, send_file
import torch

from app.data.synthetic_data import generate_synthetic_maternal_data, split_data_for_federated_learning, prepare_dataloaders
from app.data.storage import (
    init_db,
    get_prediction_count,
    get_training_history as fetch_training_history,
    list_model_versions,
    get_latest_model_version,
    get_model_version,
    record_prediction,
    save_model_version,
)
from app.federated_learning.coordinator import FederatedLearningCoordinator
from app.federated_learning.hospital_node import HospitalNode
from config import config

api_bp = Blueprint('api', __name__)

# Global variables to store the coordinator
coordinator = None

init_db()

@api_bp.route('/', methods=['GET'])
def health_check():
    """Simple HTML status page"""
    html = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Safeguarding Maternal Health with Privacy-First Predictive Analytics</title>
    <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@300;500;700&display=swap" rel="stylesheet">
    <style>
      :root {
        --bg: #f6f7fb;
        --glass: rgba(255, 255, 255, 0.78);
        --ink: #101828;
        --muted: #667085;
        --accent: #3aa0ff;
        --accent-2: #6dd3a0;
        --shadow: 0 20px 60px rgba(16, 24, 40, 0.12);
        --card-border: rgba(255, 255, 255, 0.6);
        --row-bg: rgba(255, 255, 255, 0.6);
        --stat-bg: rgba(255, 255, 255, 0.7);
        --bg-grad-1: #dbe9ff;
        --bg-grad-2: #dff6eb;
      }
      [data-theme="dark"] {
        --bg: #0b1020;
        --glass: rgba(15, 23, 42, 0.78);
        --ink: #e2e8f0;
        --muted: #94a3b8;
        --accent: #6aa9ff;
        --accent-2: #6dd3a0;
        --shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
        --card-border: rgba(148, 163, 184, 0.18);
        --row-bg: rgba(15, 23, 42, 0.7);
        --stat-bg: rgba(15, 23, 42, 0.6);
        --bg-grad-1: #1f2a44;
        --bg-grad-2: #123b2c;
      }
      * { box-sizing: border-box; }
      body {
        font-family: "Manrope", "Avenir Next", sans-serif;
        margin: 0;
        color: var(--ink);
        background: radial-gradient(1200px 600px at 20% -10%, var(--bg-grad-1) 0%, transparent 60%),
                    radial-gradient(900px 500px at 120% 20%, var(--bg-grad-2) 0%, transparent 60%),
                    var(--bg);
        min-height: 100vh;
        display: flex;
        align-items: flex-start;
        justify-content: center;
        padding: 32px;
      }
      .shell {
        width: min(920px, 100%);
        display: grid;
        gap: 20px;
        grid-template-columns: 1.2fr 0.8fr;
        margin-top: 5vh;
      }
      .card {
        background: var(--glass);
        backdrop-filter: blur(18px);
        border-radius: 20px;
        padding: 28px 30px;
        box-shadow: var(--shadow);
        border: 1px solid var(--card-border);
        animation: rise 600ms ease-out both;
      }
      .card:nth-child(2) { animation-delay: 120ms; }
      h1 { margin: 0 0 8px; font-size: 28px; letter-spacing: 0.2px; }
      .pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(58, 160, 255, 0.12);
        color: #1b4b91;
        font-weight: 600;
        font-size: 13px;
      }
      .muted { color: var(--muted); font-size: 14px; margin-top: 8px; }
      .endpoints {
        display: grid;
        gap: 10px;
        margin: 18px 0 0;
      }
      .row {
        display: grid;
        grid-template-columns: 80px 1fr;
        gap: 12px;
        padding: 10px 12px;
        border-radius: 12px;
        background: var(--row-bg);
      }
      .method {
        font-weight: 700;
        font-size: 12px;
        letter-spacing: 0.6px;
        color: #0b6bcb;
      }
      .path { font-weight: 600; }
      .note { color: var(--muted); font-size: 13px; }
      .mini {
        display: grid;
        gap: 12px;
      }
      .stat {
        padding: 14px 16px;
        border-radius: 14px;
        background: var(--stat-bg);
      }
      .stat h3 { margin: 0 0 6px; font-size: 14px; color: var(--muted); }
      .stat p { margin: 0; font-weight: 700; }
      .badge {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        box-shadow: 0 0 0 6px rgba(58, 160, 255, 0.1);
      }
      @keyframes rise {
        from { transform: translateY(12px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
      }
      .tabs {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 14px;
        padding: 6px;
        border-radius: 999px;
        background: var(--row-bg);
      }
      .tab-group {
        display: inline-flex;
        gap: 8px;
      }
      .tab {
        text-decoration: none;
        font-size: 13px;
        font-weight: 700;
        padding: 6px 12px;
        border-radius: 999px;
        color: var(--muted);
      }
      .tab.active {
        color: var(--ink);
        background: rgba(58, 160, 255, 0.16);
      }
      .theme-toggle {
        border: none;
        background: var(--stat-bg);
        color: var(--ink);
        font-size: 12px;
        font-weight: 700;
        padding: 6px 12px;
        border-radius: 999px;
        cursor: pointer;
      }
      .chart-card {
        margin-top: 18px;
        padding: 16px;
        border-radius: 16px;
        background: var(--stat-bg);
        min-height: 280px;
      }
      .chart-title {
        font-size: 14px;
        font-weight: 700;
        color: var(--muted);
        margin-bottom: 10px;
      }
      #training-chart {
        width: 100%;
        height: 220px;
        display: block;
      }
      #chart-empty {
        margin-top: 8px;
        font-size: 13px;
        color: var(--muted);
      }
      @media (max-width: 820px) {
        .shell { grid-template-columns: 1fr; }
        body { padding: 20px; }
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <div class="card">
        <div class="tabs">
          <div class="tab-group">
            <a class="tab active" href="/">Overview</a>
            <a class="tab" href="/metrics">Metrics</a>
            <a class="tab" href="/about">About</a>
          </div>
          <button class="theme-toggle" id="theme-toggle" type="button">Dark</button>
        </div>
        <div class="pill"><span class="badge"></span>API Status: OK</div>
        <h1>Safeguarding Maternal Health with Privacy-First Predictive Analytics</h1>
        <div class="muted">Federated maternal risk prediction service</div>
        <div class="endpoints">
          <div class="row">
            <div class="method">POST</div>
            <div><span class="path">/api/initialize</span> <span class="note">Initialize federated learning</span></div>
          </div>
          <div class="row">
            <div class="method">POST</div>
            <div><span class="path">/api/train</span> <span class="note">Run federated training</span></div>
          </div>
          <div class="row">
            <div class="method">GET</div>
            <div><span class="path">/api/evaluate</span> <span class="note">Evaluate current model</span></div>
          </div>
          <div class="row">
            <div class="method">POST</div>
            <div><span class="path">/api/predict</span> <span class="note">Predict maternal risk</span></div>
          </div>
          <div class="row">
            <div class="method">GET</div>
            <div><span class="path">/api/history</span> <span class="note">Get training history</span></div>
          </div>
        </div>
        <div class="chart-card">
          <div class="chart-title">Training Metrics</div>
          <canvas id="training-chart" width="500" height="220"></canvas>
          <div id="chart-empty">No training data yet.</div>
        </div>
      </div>
      <div class="card mini">
        <div class="stat">
          <h3>Device</h3>
          <p>""" + str(config.DEVICE) + """</p>
        </div>
        <div class="stat">
          <h3>Hospitals</h3>
          <p>""" + str(config.NUM_HOSPITALS) + """</p>
        </div>
        <div class="stat">
          <h3>Features</h3>
          <p>""" + str(config.NUM_FEATURES) + """</p>
        </div>
        <div class="stat">
          <h3>Predictions served</h3>
          <p id="prediction-count">--</p>
        </div>
        <div class="stat">
          <h3>Training rounds</h3>
          <p id="training-rounds">--</p>
        </div>
        <div class="stat">
          <h3>Latest model</h3>
          <p id="latest-model">--</p>
          <a id="model-download" class="note" href="#" style="text-decoration:none; display:inline-block; margin-top:6px;">Download</a>
        </div>
      </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
      const root = document.documentElement;
      const themeToggle = document.getElementById("theme-toggle");
      const savedTheme = localStorage.getItem("theme");
      if (savedTheme) {
        root.setAttribute("data-theme", savedTheme);
      }
      function updateToggleLabel() {
        const isDark = root.getAttribute("data-theme") === "dark";
        themeToggle.textContent = isDark ? "Light" : "Dark";
      }
      themeToggle.addEventListener("click", () => {
        const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
        root.setAttribute("data-theme", next);
        localStorage.setItem("theme", next);
        updateToggleLabel();
      });
      updateToggleLabel();

      let trainingChart = null;
      let lastStatsJSON = "";
      let lastChartJSON = "";

      async function refreshStats() {
        try {
          const res = await fetch("/api/stats");
          if (!res.ok) return;
          const text = await res.text();
          if (text === lastStatsJSON) return;
          lastStatsJSON = text;
          const data = JSON.parse(text);
          const predictions = document.getElementById("prediction-count");
          const rounds = document.getElementById("training-rounds");
          const latestModel = document.getElementById("latest-model");
          const download = document.getElementById("model-download");
          if (predictions) predictions.textContent = data.predictions_served;
          if (rounds) rounds.textContent = data.training_rounds;
          if (latestModel) {
            latestModel.textContent = data.latest_model_version ? ("v" + data.latest_model_version) : "--";
          }
          if (download) {
            if (data.latest_model_version) {
              download.href = "/api/model/download/" + data.latest_model_version;
              download.style.pointerEvents = "auto";
              download.style.opacity = "1";
            } else {
              download.href = "#";
              download.style.pointerEvents = "none";
              download.style.opacity = "0.5";
            }
          }
        } catch (err) {
          // Ignore transient fetch errors.
        }
      }

      async function refreshChart() {
        try {
          const res = await fetch("/api/history");
          if (!res.ok) return;
          const text = await res.text();
          if (text === lastChartJSON) return;
          lastChartJSON = text;
          const payload = JSON.parse(text);
          if (payload.status !== "success") return;
          const history = payload.history || [];
          const empty = document.getElementById("chart-empty");
          if (!history.length) {
            if (empty) empty.style.display = "block";
            return;
          }
          if (empty) empty.style.display = "none";
          const labels = history.map((row) => "Round " + row.round);
          const trainLoss = history.map((row) => row.train_loss);
          const testAcc = history.map((row) => row.test_accuracy);

          if (!window.Chart) return;
          const ctx = document.getElementById("training-chart");
          if (!ctx) return;
          if (!trainingChart) {
            trainingChart = new Chart(ctx, {
              type: "line",
              data: {
                labels,
                datasets: [
                  {
                    label: "Train Loss",
                    data: trainLoss,
                    borderColor: "#3aa0ff",
                    backgroundColor: "rgba(58, 160, 255, 0.2)",
                    yAxisID: "y",
                    tension: 0.35
                  },
                  {
                    label: "Test Accuracy",
                    data: testAcc,
                    borderColor: "#6dd3a0",
                    backgroundColor: "rgba(109, 211, 160, 0.2)",
                    yAxisID: "y1",
                    tension: 0.35
                  }
                ]
              },
              options: {
                responsive: false,
                animation: false,
                scales: {
                  y: {
                    position: "left",
                    title: { display: true, text: "Loss" }
                  },
                  y1: {
                    position: "right",
                    title: { display: true, text: "Accuracy" },
                    grid: { drawOnChartArea: false },
                    min: 0,
                    max: 1
                  }
                }
              }
            });
          } else {
            trainingChart.data.labels = labels;
            trainingChart.data.datasets[0].data = trainLoss;
            trainingChart.data.datasets[1].data = testAcc;
            trainingChart.update("none");
          }
        } catch (err) {
          // Ignore transient fetch errors.
        }
      }

      refreshStats();
      refreshChart();
      setInterval(refreshStats, 5000);
      setInterval(refreshChart, 10000);
    </script>
  </body>
</html>
"""
    return Response(html, mimetype="text/html")


@api_bp.route('/about', methods=['GET'])
def about_page():
    """About page for the demo."""
    html = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>About • Safeguarding Maternal Health with Privacy-First Predictive Analytics</title>
    <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@300;500;700&display=swap" rel="stylesheet">
    <style>
      :root {
        --bg: #f6f7fb;
        --glass: rgba(255, 255, 255, 0.78);
        --ink: #101828;
        --muted: #667085;
        --accent: #3aa0ff;
        --accent-2: #6dd3a0;
        --shadow: 0 20px 60px rgba(16, 24, 40, 0.12);
        --card-border: rgba(255, 255, 255, 0.6);
        --row-bg: rgba(255, 255, 255, 0.6);
        --stat-bg: rgba(255, 255, 255, 0.7);
        --bg-grad-1: #dbe9ff;
        --bg-grad-2: #dff6eb;
      }
      [data-theme="dark"] {
        --bg: #0b1020;
        --glass: rgba(15, 23, 42, 0.78);
        --ink: #e2e8f0;
        --muted: #94a3b8;
        --accent: #6aa9ff;
        --accent-2: #6dd3a0;
        --shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
        --card-border: rgba(148, 163, 184, 0.18);
        --row-bg: rgba(15, 23, 42, 0.7);
        --stat-bg: rgba(15, 23, 42, 0.6);
        --bg-grad-1: #1f2a44;
        --bg-grad-2: #123b2c;
      }
      * { box-sizing: border-box; }
      body {
        font-family: "Manrope", "Avenir Next", sans-serif;
        margin: 0;
        color: var(--ink);
        background: radial-gradient(1200px 600px at 20% -10%, var(--bg-grad-1) 0%, transparent 60%),
                    radial-gradient(900px 500px at 120% 20%, var(--bg-grad-2) 0%, transparent 60%),
                    var(--bg);
        min-height: 100vh;
        display: flex;
        align-items: flex-start;
        justify-content: center;
        padding: 32px;
      }
      .card {
        margin-top: 5vh;
        width: min(900px, 100%);
        background: var(--glass);
        backdrop-filter: blur(18px);
        border-radius: 20px;
        padding: 28px 30px;
        box-shadow: var(--shadow);
        border: 1px solid var(--card-border);
        animation: rise 600ms ease-out both;
      }
      h1 { margin: 0 0 8px; font-size: 28px; letter-spacing: 0.2px; }
      .muted { color: var(--muted); font-size: 14px; }
      .tabs {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 14px;
        padding: 6px;
        border-radius: 999px;
        background: var(--row-bg);
      }
      .tab-group {
        display: inline-flex;
        gap: 8px;
      }
      .tab {
        text-decoration: none;
        font-size: 13px;
        font-weight: 700;
        padding: 6px 12px;
        border-radius: 999px;
        color: var(--muted);
      }
      .tab.active {
        color: var(--ink);
        background: rgba(58, 160, 255, 0.16);
      }
      .theme-toggle {
        border: none;
        background: var(--stat-bg);
        color: var(--ink);
        font-size: 12px;
        font-weight: 700;
        padding: 6px 12px;
        border-radius: 999px;
        cursor: pointer;
      }
      .grid {
        margin-top: 18px;
        display: grid;
        gap: 16px;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      }
      .box {
        padding: 14px 16px;
        border-radius: 14px;
        background: var(--stat-bg);
      }
      .box h3 { margin: 0 0 6px; font-size: 14px; color: var(--muted); }
      .box p { margin: 0; font-size: 14px; line-height: 1.5; }
      @keyframes rise {
        from { transform: translateY(12px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
      }
    </style>
  </head>
  <body>
    <div class="card">
      <div class="tabs">
        <div class="tab-group">
          <a class="tab" href="/">Overview</a>
          <a class="tab" href="/metrics">Metrics</a>
          <a class="tab active" href="/about">About</a>
        </div>
        <button class="theme-toggle" id="theme-toggle" type="button">Dark</button>
      </div>
      <h1>Safeguarding Maternal Health with Privacy-First Predictive Analytics</h1>
      <div class="muted">A federated learning demo for maternal health risk prediction.</div>
      <div class="grid">
        <div class="box">
          <h3>Purpose</h3>
          <p>Show how multiple hospitals can train a shared model without sharing raw patient data.</p>
        </div>
        <div class="box">
          <h3>How it works</h3>
          <p>Each node trains locally, the coordinator averages weights, and the API serves predictions.</p>
        </div>
        <div class="box">
          <h3>Why it matters</h3>
          <p>Improves model quality while preserving privacy and data ownership.</p>
        </div>
      </div>
    </div>
    <script>
      const root = document.documentElement;
      const themeToggle = document.getElementById("theme-toggle");
      const savedTheme = localStorage.getItem("theme");
      if (savedTheme) {
        root.setAttribute("data-theme", savedTheme);
      }
      function updateToggleLabel() {
        const isDark = root.getAttribute("data-theme") === "dark";
        themeToggle.textContent = isDark ? "Light" : "Dark";
      }
      themeToggle.addEventListener("click", () => {
        const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
        root.setAttribute("data-theme", next);
        localStorage.setItem("theme", next);
        updateToggleLabel();
      });
      updateToggleLabel();
    </script>
  </body>
</html>
"""
    return Response(html, mimetype="text/html")


@api_bp.route('/metrics', methods=['GET'])
def metrics_page():
    """Metrics dashboard page."""
    html = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Metrics • Safeguarding Maternal Health with Privacy-First Predictive Analytics</title>
    <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@300;500;700&display=swap" rel="stylesheet">
    <style>
      :root {
        --bg: #f6f7fb;
        --glass: rgba(255, 255, 255, 0.78);
        --ink: #101828;
        --muted: #667085;
        --accent: #3aa0ff;
        --accent-2: #6dd3a0;
        --shadow: 0 20px 60px rgba(16, 24, 40, 0.12);
        --card-border: rgba(255, 255, 255, 0.6);
        --row-bg: rgba(255, 255, 255, 0.6);
        --stat-bg: rgba(255, 255, 255, 0.7);
        --bg-grad-1: #dbe9ff;
        --bg-grad-2: #dff6eb;
      }
      [data-theme="dark"] {
        --bg: #0b1020;
        --glass: rgba(15, 23, 42, 0.78);
        --ink: #e2e8f0;
        --muted: #94a3b8;
        --accent: #6aa9ff;
        --accent-2: #6dd3a0;
        --shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
        --card-border: rgba(148, 163, 184, 0.18);
        --row-bg: rgba(15, 23, 42, 0.7);
        --stat-bg: rgba(15, 23, 42, 0.6);
        --bg-grad-1: #1f2a44;
        --bg-grad-2: #123b2c;
      }
      * { box-sizing: border-box; }
      body {
        font-family: "Manrope", "Avenir Next", sans-serif;
        margin: 0;
        color: var(--ink);
        background: radial-gradient(1200px 600px at 20% -10%, var(--bg-grad-1) 0%, transparent 60%),
                    radial-gradient(900px 500px at 120% 20%, var(--bg-grad-2) 0%, transparent 60%),
                    var(--bg);
        min-height: 100vh;
        display: flex;
        align-items: flex-start;
        justify-content: center;
        padding: 32px;
      }
      .shell {
        margin-top: 5vh;
        width: min(1050px, 100%);
        display: grid;
        gap: 20px;
      }
      .card {
        background: var(--glass);
        backdrop-filter: blur(18px);
        border-radius: 20px;
        padding: 28px 30px;
        box-shadow: var(--shadow);
        border: 1px solid var(--card-border);
        animation: rise 600ms ease-out both;
      }
      .tabs {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 14px;
        padding: 6px;
        border-radius: 999px;
        background: var(--row-bg);
      }
      .tab-group {
        display: inline-flex;
        gap: 8px;
      }
      .tab {
        text-decoration: none;
        font-size: 13px;
        font-weight: 700;
        padding: 6px 12px;
        border-radius: 999px;
        color: var(--muted);
      }
      .tab.active {
        color: var(--ink);
        background: rgba(58, 160, 255, 0.16);
      }
      .theme-toggle {
        border: none;
        background: var(--stat-bg);
        color: var(--ink);
        font-size: 12px;
        font-weight: 700;
        padding: 6px 12px;
        border-radius: 999px;
        cursor: pointer;
      }
      h1 { margin: 0; font-size: 26px; }
      .muted { color: var(--muted); font-size: 14px; margin-top: 6px; }
      .grid {
        margin-top: 18px;
        display: grid;
        gap: 18px;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      }
      .orbital {
        position: relative;
        width: 240px;
        height: 240px;
        margin: 0 auto;
        border-radius: 50%;
        background:
          conic-gradient(from -90deg, rgba(109, 211, 160, 0.35) calc(var(--secondary, 0) * 1turn), transparent 0),
          conic-gradient(from -90deg, rgba(58, 160, 255, 0.5) calc(var(--primary, 0) * 1turn), transparent 0),
          repeating-conic-gradient(rgba(148, 163, 184, 0.25) 0deg, rgba(148, 163, 184, 0.25) 2deg, transparent 2deg, transparent 14deg),
          radial-gradient(circle at center, rgba(255, 255, 255, 0.9) 0%, rgba(255, 255, 255, 0.7) 55%, rgba(255, 255, 255, 0.2) 100%);
        box-shadow: inset 0 0 40px rgba(15, 23, 42, 0.1);
      }
      [data-theme="dark"] .orbital {
        background:
          conic-gradient(from -90deg, rgba(109, 211, 160, 0.35) calc(var(--secondary, 0) * 1turn), transparent 0),
          conic-gradient(from -90deg, rgba(106, 169, 255, 0.55) calc(var(--primary, 0) * 1turn), transparent 0),
          repeating-conic-gradient(rgba(148, 163, 184, 0.18) 0deg, rgba(148, 163, 184, 0.18) 2deg, transparent 2deg, transparent 14deg),
          radial-gradient(circle at center, rgba(15, 23, 42, 0.9) 0%, rgba(15, 23, 42, 0.7) 55%, rgba(15, 23, 42, 0.2) 100%);
        box-shadow: inset 0 0 40px rgba(0, 0, 0, 0.35);
      }
      .orbit {
        position: absolute;
        inset: 16px;
        border-radius: 50%;
        animation: spin 10s linear infinite;
      }
      .orbit::after {
        content: "";
        position: absolute;
        top: -6px;
        left: 50%;
        width: 10px;
        height: 10px;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        border-radius: 50%;
        transform: translateX(-50%);
        box-shadow: 0 0 12px rgba(58, 160, 255, 0.6);
      }
      .orbit.secondary {
        inset: 32px;
        animation-duration: 14s;
      }
      .orbit.secondary::after {
        width: 8px;
        height: 8px;
        box-shadow: 0 0 12px rgba(109, 211, 160, 0.6);
      }
      .center {
        position: absolute;
        inset: 38px;
        border-radius: 50%;
        background: var(--glass);
        display: grid;
        place-items: center;
        text-align: center;
        padding: 16px;
      }
      .center h2 {
        margin: 0;
        font-size: 18px;
      }
      .center p {
        margin: 6px 0 0;
        font-size: 14px;
        color: var(--muted);
      }
      .pair {
        margin-top: 10px;
        font-size: 13px;
        color: var(--muted);
      }
      .metric-card {
        padding: 18px;
        border-radius: 16px;
        background: var(--stat-bg);
      }
      .metric-card h3 {
        margin: 0 0 8px;
        font-size: 15px;
        color: var(--muted);
      }
      .empty {
        margin-top: 14px;
        font-size: 13px;
        color: var(--muted);
      }
      @keyframes rise {
        from { transform: translateY(12px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
      }
      @keyframes spin {
        to { transform: rotate(360deg); }
      }
      @media (max-width: 820px) {
        body { padding: 20px; }
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <div class="card">
        <div class="tabs">
          <div class="tab-group">
            <a class="tab" href="/">Overview</a>
            <a class="tab active" href="/metrics">Metrics</a>
            <a class="tab" href="/about">About</a>
          </div>
          <button class="theme-toggle" id="theme-toggle" type="button">Dark</button>
        </div>
        <h1>Metrics Observatory</h1>
        <div class="muted">Orbiting indicators and radial spokes for the latest training results.</div>
        <div class="grid">
          <div class="metric-card">
            <h3>Train Loss + Test Accuracy</h3>
            <div class="orbital" id="gauge-loss-acc">
              <div class="orbit"></div>
              <div class="orbit secondary"></div>
              <div class="center">
                <div>
                  <h2 id="loss-value">--</h2>
                  <p>Train Loss</p>
                  <div class="pair" id="acc-value">Test Accuracy: --</div>
                </div>
              </div>
            </div>
          </div>
          <div class="metric-card">
            <h3>AUC + F1</h3>
            <div class="orbital" id="gauge-auc-f1">
              <div class="orbit"></div>
              <div class="orbit secondary"></div>
              <div class="center">
                <div>
                  <h2 id="auc-value">--</h2>
                  <p>Test AUC</p>
                  <div class="pair" id="f1-value">F1: --</div>
                </div>
              </div>
            </div>
          </div>
          <div class="metric-card">
            <h3>Precision + Recall</h3>
            <div class="orbital" id="gauge-prec-rec">
              <div class="orbit"></div>
              <div class="orbit secondary"></div>
              <div class="center">
                <div>
                  <h2 id="precision-value">--</h2>
                  <p>Precision</p>
                  <div class="pair" id="recall-value">Recall: --</div>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div class="empty" id="metrics-empty">No training history yet. Run /api/initialize then /api/train.</div>
      </div>
    </div>
    <script>
      const root = document.documentElement;
      const themeToggle = document.getElementById("theme-toggle");
      const savedTheme = localStorage.getItem("theme");
      if (savedTheme) {
        root.setAttribute("data-theme", savedTheme);
      }
      function updateToggleLabel() {
        const isDark = root.getAttribute("data-theme") === "dark";
        themeToggle.textContent = isDark ? "Light" : "Dark";
      }
      themeToggle.addEventListener("click", () => {
        const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
        root.setAttribute("data-theme", next);
        localStorage.setItem("theme", next);
        updateToggleLabel();
      });
      updateToggleLabel();

      function setGauge(el, primary, secondary) {
        if (!el) return;
        el.style.setProperty("--primary", primary);
        el.style.setProperty("--secondary", secondary);
      }

      function formatNumber(value) {
        if (value === null || value === undefined || Number.isNaN(value)) return "--";
        return Number(value).toFixed(3);
      }

      let lastMetricsJSON = "";

      async function refreshMetrics() {
        try {
          const res = await fetch("/api/history");
          if (!res.ok) return;
          const text = await res.text();
          if (text === lastMetricsJSON) return;
          lastMetricsJSON = text;
          const payload = JSON.parse(text);
          if (payload.status !== "success") return;
          const history = payload.history || [];
          const empty = document.getElementById("metrics-empty");
          if (!history.length) {
            if (empty) empty.style.display = "block";
            return;
          }
          if (empty) empty.style.display = "none";
          const latest = history[history.length - 1];

          const trainLoss = Number(latest.train_loss);
          const testAcc = Number(latest.test_accuracy);
          const auc = Number(latest.test_auc);
          const f1 = Number(latest.test_f1);
          const precision = Number(latest.test_precision);
          const recall = Number(latest.test_recall);

          const lossGauge = document.getElementById("gauge-loss-acc");
          const aucGauge = document.getElementById("gauge-auc-f1");
          const precGauge = document.getElementById("gauge-prec-rec");

          const lossNormalized = 1 / (1 + (Number.isNaN(trainLoss) ? 0 : trainLoss));
          setGauge(lossGauge, lossNormalized, Number.isNaN(testAcc) ? 0 : testAcc);
          setGauge(aucGauge, Number.isNaN(auc) ? 0 : auc, Number.isNaN(f1) ? 0 : f1);
          setGauge(precGauge, Number.isNaN(precision) ? 0 : precision, Number.isNaN(recall) ? 0 : recall);

          const lossValue = document.getElementById("loss-value");
          const accValue = document.getElementById("acc-value");
          const aucValue = document.getElementById("auc-value");
          const f1Value = document.getElementById("f1-value");
          const precisionValue = document.getElementById("precision-value");
          const recallValue = document.getElementById("recall-value");

          if (lossValue) lossValue.textContent = formatNumber(trainLoss);
          if (accValue) accValue.textContent = "Test Accuracy: " + formatNumber(testAcc);
          if (aucValue) aucValue.textContent = formatNumber(auc);
          if (f1Value) f1Value.textContent = "F1: " + formatNumber(f1);
          if (precisionValue) precisionValue.textContent = formatNumber(precision);
          if (recallValue) recallValue.textContent = "Recall: " + formatNumber(recall);
        } catch (err) {
          // Ignore transient fetch errors.
        }
      }

      refreshMetrics();
      setInterval(refreshMetrics, 10000);
    </script>
  </body>
</html>
"""
    return Response(html, mimetype="text/html")


@api_bp.route('/api/stats', methods=['GET'])
def get_stats():
    """Return lightweight runtime stats for the UI."""
    history = fetch_training_history()
    latest_model = get_latest_model_version()
    return jsonify({
        'predictions_served': get_prediction_count(),
        'training_rounds': len(history),
        'latest_model_version': latest_model['version'] if latest_model else None
    })

@api_bp.route('/api/model/versions', methods=['GET'])
def get_model_versions():
    """List saved model versions."""
    return jsonify({
        'status': 'success',
        'versions': list_model_versions()
    })

@api_bp.route('/api/model/latest', methods=['GET'])
def get_latest_model():
    """Return latest model version metadata."""
    latest_model = get_latest_model_version()
    if not latest_model:
        return jsonify({
            'status': 'error',
            'message': 'No model versions available'
        }), 404
    return jsonify({
        'status': 'success',
        'model': latest_model
    })

@api_bp.route('/api/model/download/<int:version>', methods=['GET'])
def download_model(version):
    """Download a specific model version."""
    model_info = get_model_version(version)
    if not model_info:
        return jsonify({
            'status': 'error',
            'message': 'Model version not found'
        }), 404
    model_path = model_info['path']
    if not os.path.exists(model_path):
        return jsonify({
            'status': 'error',
            'message': 'Model file missing on disk'
        }), 404
    return send_file(model_path, as_attachment=True, download_name=os.path.basename(model_path))

@api_bp.route('/api/initialize', methods=['POST'])
def initialize_federated_learning():
    """Initialize the federated learning system with synthetic data"""
    global coordinator
    
    try:
        # Generate synthetic data
        data = generate_synthetic_maternal_data(
            n_samples=config.NUM_SAMPLES_PER_HOSPITAL * config.NUM_HOSPITALS,
            n_features=config.NUM_FEATURES
        )
        
        # Split data for federated learning
        hospital_dfs, test_df = split_data_for_federated_learning(
            data,
            n_hospitals=config.NUM_HOSPITALS,
            test_size=config.TEST_SIZE
        )
        
        # Prepare dataloaders
        hospital_dataloaders, test_dataloader, pos_weight = prepare_dataloaders(
            hospital_dfs,
            test_df,
            batch_size=config.BATCH_SIZE
        )

        # Create hospital nodes
        hospital_nodes = []
        for i, dataloader in enumerate(hospital_dataloaders):
            hospital = HospitalNode(
                node_id=i,
                dataloader=dataloader,
                device=config.DEVICE,
                config=config,
                pos_weight=pos_weight
            )
            hospital_nodes.append(hospital)
        
        # Create coordinator
        coordinator = FederatedLearningCoordinator(
            hospital_nodes,
            test_dataloader,
            config
        )
        
        return jsonify({
            'status': 'success',
            'message': f'Federated learning initialized with {config.NUM_HOSPITALS} hospitals',
            'data_stats': {
                'total_samples': len(data),
                'high_risk_percentage': (data['high_risk'].sum() / len(data)) * 100,
                'test_samples': len(test_df)
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to initialize: {str(e)}'
        }), 500

@api_bp.route('/api/train', methods=['POST'])
def train_federated_model():
    """Run federated training for specified number of rounds"""
    global coordinator
    
    if coordinator is None:
        return jsonify({
            'status': 'error',
            'message': 'Federated learning not initialized. Call /api/initialize first.'
        }), 400
    
    try:
        data = request.get_json(silent=True) or {}
        rounds = data.get('rounds', config.FEDERATED_ROUNDS)
        try:
            rounds = int(rounds)
        except (TypeError, ValueError):
            return jsonify({
                'status': 'error',
                'message': 'Rounds must be a positive integer.'
            }), 400
        if rounds <= 0:
            return jsonify({
                'status': 'error',
                'message': 'Rounds must be a positive integer.'
            }), 400
        
        # Run federated training
        history = coordinator.run_federated_training(rounds)
        model_info = save_model_version(coordinator.global_model)
        
        return jsonify({
            'status': 'success',
            'message': f'Completed {rounds} federated learning rounds',
            'history': history,
            'model_version': model_info['version']
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Training failed: {str(e)}'
        }), 500

@api_bp.route('/api/evaluate', methods=['GET'])
def evaluate_model():
    """Evaluate the current global model"""
    global coordinator
    
    if coordinator is None:
        return jsonify({
            'status': 'error',
            'message': 'No model available for evaluation'
        }), 400
    
    try:
        metrics = coordinator.evaluate_global_model()
        
        return jsonify({
            'status': 'success',
            'metrics': metrics
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Evaluation failed: {str(e)}'
        }), 500

@api_bp.route('/api/predict', methods=['POST'])
def predict_risk():
    """Predict maternal health risk for a patient"""
    global coordinator
    global prediction_count
    
    if coordinator is None:
        return jsonify({
            'status': 'error',
            'message': 'No model available for prediction'
        }), 400
    
    try:
        data = request.get_json(silent=True) or {}
        patient_data = data.get('patient_data')

        if patient_data is None:
            return jsonify({
                'status': 'error',
                'message': 'patient_data is required.'
            }), 400
        if not isinstance(patient_data, (list, tuple)):
            return jsonify({
                'status': 'error',
                'message': 'patient_data must be a list of features.'
            }), 400
        
        if len(patient_data) != config.NUM_FEATURES:
            return jsonify({
                'status': 'error',
                'message': f'Expected {config.NUM_FEATURES} features, got {len(patient_data)}'
            }), 400
        
        # Convert to tensor
        features = torch.tensor([patient_data], dtype=torch.float32).to(config.DEVICE)
        
        # Make prediction
        coordinator.global_model.eval()
        with torch.no_grad():
            logits = coordinator.global_model(features)
            risk_score = torch.sigmoid(logits).item()
        risk_category = 'High Risk' if risk_score > 0.5 else 'Low Risk'
        record_prediction(risk_score, risk_category)
        
        return jsonify({
            'status': 'success',
            'risk_score': risk_score,
            'risk_category': risk_category
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Prediction failed: {str(e)}'
        }), 500

@api_bp.route('/api/history', methods=['GET'])
def get_training_history():
    """Get the training history"""
    history = fetch_training_history()
    return jsonify({
        'status': 'success',
        'history': history
    })
