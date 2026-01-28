# Stress Testing Adani Enterprises: The Hindenburg Crisis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-green)

## 📉 Project Overview
This project performs a quantitative stress test on **Adani Enterprises** stock during the January 2023 **Hindenburg Research crisis**. Using simulated market data calibrated to historical crash parameters, it evaluates the robustness of standard volatility models during "Black Swan" events.

The core objective is to determine if statistical models (GARCH variants and Extreme Value Theory) could have predicted the magnitude of the ~50% market value erosion triggered by reputational shocks.

## 🧮 Models Implemented
The project implements the following risk models from scratch using `scipy.optimize`:

1.  **GARCH(1,1):** Standard volatility modeling with Student-t innovations.
2.  **EGARCH(1,1):** Captures asymmetric volatility (bad news > good news).
3.  **GJR-GARCH(1,1):** Explicitly models leverage effects.
4.  **EVT-POT:** Extreme Value Theory using Peak-Over-Threshold to model tail risks.

## 🚀 Key Features
* **Synthetic Data Generation:** Simulates a realistic price path with a structural break to mimic the Hindenburg report release.
* **Risk Metrics:** Calculates **99% Value at Risk (VaR)** and **Expected Shortfall (ES)**.
* **Backtesting:** Compares forecasted risk against actual "crisis" returns.
* **Automated Reporting:** Generates a detailed text report (`adani_stress_test_report.txt`) and a CSV of results.

## 🛠️ Installation & Usage

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/adani-stress-test.git](https://github.com/yourusername/adani-stress-test.git)
    cd adani-stress-test
    ```

2.  **Install dependencies**
    ```bash
    pip install numpy pandas scipy
    ```

3.  **Run the analysis**
    ```bash
    python adani_stress_test.py
    ```

## 📊 Sample Results
*Detailed results are saved to `adani_stress_test_results.csv`*

| Date | Event | Actual Return | Model Prediction (VaR 99%) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Jan 27, 2023** | **Crisis Peak** | **-22.30%** | **-5.10% (EVT)** | ❌ **CRITICAL FAILURE** |

**Key Insight:** Even the most conservative models (EVT) underestimated the tail risk by a factor of ~4x, demonstrating that historical price data cannot predict qualitative regime changes (e.g., fraud allegations).


## ⚠️ Disclaimer
This project uses **simulated data** for educational and research purposes. It is not financial advice.
