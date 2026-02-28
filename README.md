# 🏙️ Smart City IoT Analytics Platform

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?logo=plotly&logoColor=white)
![ARIMA](https://img.shields.io/badge/Model-ARIMA-7C3AED)
![Prophet](https://img.shields.io/badge/Model-Prophet-00D4FF)
![License](https://img.shields.io/badge/License-MIT-green)

> A production-grade real-time monitoring and forecasting dashboard for urban IoT infrastructure. Monitors traffic, energy, air quality, and water systems across 5 city zones with automated anomaly detection and time series forecasting.

---

## 📸 Dashboard Preview

| Overview | Forecasting |
|---|---|
| KPI cards, zone comparison, heatmaps | ARIMA forecasts with confidence intervals |

---

## 🏗️ Architecture

```
smart-city-iot/
├── app.py                    # Streamlit dashboard (5 tabs)
├── src/
│   ├── data_generator.py     # Realistic IoT data simulation (216K records)
│   └── forecasting.py        # ARIMA, Prophet, anomaly detection
├── data/                     # Auto-generated CSV datasets
│   ├── traffic.csv           # Vehicle counts, speed, congestion
│   ├── energy.csv            # kWh consumption, solar generation
│   ├── air_quality.csv       # AQI, PM2.5, PM10, CO2
│   └── water.csv             # Usage, pressure, leak risk
├── results/                  # Forecast outputs and metrics
├── notebook/                 # EDA and model evaluation
└── requirements.txt
```

---

## 📊 Dataset

Simulated 90 days of IoT sensor data across **5 city zones** (Downtown, Industrial, Residential, Airport, Harbor):

| Domain | Records | Key Metrics |
|---|---|---|
| Traffic | 54,000 | Vehicle count, avg speed, congestion index |
| Energy | 54,000 | kWh consumption, solar generation, grid load % |
| Air Quality | 54,000 | AQI, PM2.5, PM10, CO₂ ppm |
| Water | 54,000 | Usage (L), pressure (bar), pipe leak risk |
| **Total** | **216,000** | Across 5 zones × 90 days × hourly |

**Realistic simulation features:**
- Rush-hour traffic peaks (8am, 6pm)
- Business-hour energy patterns
- Weekend demand reduction
- Solar generation curves (daylight hours)
- Injected anomalies (spikes, drops) for detection testing

---

## 🤖 Models

### Time Series Forecasting
| Model | Order | Use Case | Typical MAPE |
|---|---|---|---|
| ARIMA | (2,1,2) | Short-term energy/traffic | ~8-12% |
| Prophet | Multiplicative | Long-term with seasonality | ~6-10% |

### Anomaly Detection
| Method | Threshold | Precision |
|---|---|---|
| Z-Score (rolling 24h) | ±2.5σ | ~82% |
| IQR-based | 1.5× IQR | ~78% |

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/yourusername/smart-city-iot.git
cd smart-city-iot

# Install
pip install -r requirements.txt

# Generate data (auto-runs on first dashboard launch too)
python -m src.data_generator

# Launch dashboard
streamlit run app.py
```

Dashboard opens at `http://localhost:8501`

---

## 📱 Dashboard Tabs

| Tab | Description |
|---|---|
| **Overview** | KPI cards, cross-domain trends, AQI heatmap |
| **Traffic** | Anomaly detection, hourly heatmaps, congestion index |
| **Energy** | Consumption vs solar, grid load alerts |
| **Air Quality** | AQI trends, health category bands, PM correlation |
| **Water** | Usage patterns, pressure monitoring, leak risk |
| **Forecasting** | Interactive ARIMA forecasts, Z-score anomaly flagging |

---

## 🔍 Key Design Decisions

| Decision | Rationale |
|---|---|
| Hourly granularity | Captures intraday patterns without excessive noise |
| Zone-level simulation | Reflects real-world heterogeneity across urban areas |
| Rolling Z-score | Adapts to non-stationary IoT signals better than global stats |
| ARIMA(2,1,2) | Balances fit quality and computational cost for hourly data |
| Streamlit over Flask | Faster iteration; production deployment possible via Streamlit Cloud |

---

## 🛠️ Tech Stack

- **Streamlit** — Interactive dashboard framework
- **Plotly** — Interactive charts (heatmaps, time series, scatter)
- **statsmodels** — ARIMA implementation
- **Prophet** — Facebook's time series forecasting
- **pandas / numpy** — Data pipeline
- **scikit-learn** — Preprocessing utilities

---

## 📄 License

MIT © Mishika Ahuja
