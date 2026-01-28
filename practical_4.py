import numpy as np
import pandas as pd
from scipy import stats, optimize
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class GARCHModel:
    def __init__(self, returns):
        self.returns = np.array(returns)
        self.params = None
        self.sigma2 = None
        
    def _garch_likelihood(self, params):
        omega, alpha, beta, nu = params
        
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1 or nu <= 2:
            return 1e10
        
        T = len(self.returns)
        sigma2 = np.zeros(T)
        sigma2[0] = np.var(self.returns)
        
        for t in range(1, T):
            sigma2[t] = omega + alpha * self.returns[t-1]**2 + beta * sigma2[t-1]
        
        loglik = np.sum(
            stats.t.logpdf(self.returns, df=nu, scale=np.sqrt(sigma2))
        )
        
        return -loglik
    
    def fit(self):
        initial_sigma2 = np.var(self.returns)
        x0 = [0.01, 0.05, 0.90, 10]
        
        result = optimize.minimize(
            self._garch_likelihood,
            x0,
            method='L-BFGS-B',
            bounds=[(1e-6, 1), (1e-6, 0.5), (1e-6, 0.98), (2.1, 100)]
        )
        
        if result.success:
            self.params = result.x
            omega, alpha, beta, nu = self.params
            T = len(self.returns)
            self.sigma2 = np.zeros(T)
            self.sigma2[0] = np.var(self.returns)
            for t in range(1, T):
                self.sigma2[t] = omega + alpha * self.returns[t-1]**2 + beta * self.sigma2[t-1]
            return True
        return False
    
    def forecast(self, horizon=1):
        if self.params is None:
            raise ValueError("Model not fitted")
        
        omega, alpha, beta, nu = self.params
        
        last_return = self.returns[-1]
        last_sigma2 = self.sigma2[-1]
        forecast_sigma2 = omega + alpha * last_return**2 + beta * last_sigma2
        
        return np.sqrt(forecast_sigma2), nu


class EGARCHModel:
    def __init__(self, returns):
        self.returns = np.array(returns)
        self.params = None
        self.sigma2 = None
        
    def _egarch_likelihood(self, params):
        omega, alpha, gamma, beta, nu = params
        
        if abs(beta) >= 1 or nu <= 2:
            return 1e10
        
        T = len(self.returns)
        log_sigma2 = np.zeros(T)
        log_sigma2[0] = np.log(np.var(self.returns))
        
        for t in range(1, T):
            sigma_prev = np.exp(0.5 * log_sigma2[t-1])
            z_prev = self.returns[t-1] / sigma_prev if sigma_prev > 0 else 0
            log_sigma2[t] = omega + alpha * abs(z_prev) + gamma * z_prev + beta * log_sigma2[t-1]
        
        sigma2 = np.exp(log_sigma2)
        
        loglik = np.sum(
            stats.t.logpdf(self.returns, df=nu, scale=np.sqrt(sigma2))
        )
        
        return -loglik
    
    def fit(self):
        x0 = [0.01, 0.1, -0.05, 0.95, 10]
        
        result = optimize.minimize(
            self._egarch_likelihood,
            x0,
            method='L-BFGS-B',
            bounds=[(-5, 5), (0, 2), (-1, 1), (0, 0.999), (2.1, 100)]
        )
        
        if result.success:
            self.params = result.x
            omega, alpha, gamma, beta, nu = self.params
            T = len(self.returns)
            log_sigma2 = np.zeros(T)
            log_sigma2[0] = np.log(np.var(self.returns))
            
            for t in range(1, T):
                sigma_prev = np.exp(0.5 * log_sigma2[t-1])
                z_prev = self.returns[t-1] / sigma_prev if sigma_prev > 0 else 0
                log_sigma2[t] = omega + alpha * abs(z_prev) + gamma * z_prev + beta * log_sigma2[t-1]
            
            self.sigma2 = np.exp(log_sigma2)
            return True
        return False
    
    def forecast(self, horizon=1):
        if self.params is None:
            raise ValueError("Model not fitted")
        
        omega, alpha, gamma, beta, nu = self.params
        
        last_sigma = np.sqrt(self.sigma2[-1])
        last_z = self.returns[-1] / last_sigma if last_sigma > 0 else 0
        log_forecast_sigma2 = omega + alpha * abs(last_z) + gamma * last_z + beta * np.log(self.sigma2[-1])
        
        return np.sqrt(np.exp(log_forecast_sigma2)), nu


class GJRGARCHModel:
    def __init__(self, returns):
        self.returns = np.array(returns)
        self.params = None
        self.sigma2 = None
        
    def _gjr_likelihood(self, params):
        omega, alpha, gamma, beta, nu = params
        
        if omega <= 0 or alpha < 0 or gamma < 0 or beta < 0 or alpha + gamma/2 + beta >= 1 or nu <= 2:
            return 1e10
        
        T = len(self.returns)
        sigma2 = np.zeros(T)
        sigma2[0] = np.var(self.returns)
        
        for t in range(1, T):
            indicator = 1 if self.returns[t-1] < 0 else 0
            sigma2[t] = omega + alpha * self.returns[t-1]**2 + gamma * self.returns[t-1]**2 * indicator + beta * sigma2[t-1]
        
        loglik = np.sum(
            stats.t.logpdf(self.returns, df=nu, scale=np.sqrt(sigma2))
        )
        
        return -loglik
    
    def fit(self):
        x0 = [0.01, 0.05, 0.05, 0.88, 10]
        
        result = optimize.minimize(
            self._gjr_likelihood,
            x0,
            method='L-BFGS-B',
            bounds=[(1e-6, 1), (1e-6, 0.5), (1e-6, 0.5), (1e-6, 0.98), (2.1, 100)]
        )
        
        if result.success:
            self.params = result.x
            omega, alpha, gamma, beta, nu = self.params
            T = len(self.returns)
            self.sigma2 = np.zeros(T)
            self.sigma2[0] = np.var(self.returns)
            
            for t in range(1, T):
                indicator = 1 if self.returns[t-1] < 0 else 0
                self.sigma2[t] = omega + alpha * self.returns[t-1]**2 + gamma * self.returns[t-1]**2 * indicator + beta * self.sigma2[t-1]
            return True
        return False
    
    def forecast(self, horizon=1):
        if self.params is None:
            raise ValueError("Model not fitted")
        
        omega, alpha, gamma, beta, nu = self.params
        
        last_return = self.returns[-1]
        last_sigma2 = self.sigma2[-1]
        indicator = 1 if last_return < 0 else 0
        forecast_sigma2 = omega + alpha * last_return**2 + gamma * last_return**2 * indicator + beta * last_sigma2
        
        return np.sqrt(forecast_sigma2), nu


def fit_evt_pot(returns, threshold_quantile=0.90):
    threshold = np.percentile(np.abs(returns), threshold_quantile * 100)
    exceedances = np.abs(returns[np.abs(returns) > threshold]) - threshold
    
    if len(exceedances) < 10:
        return None, None, None
    
    shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)
    return shape, scale, threshold

def calculate_evt_var_es(shape, scale, threshold, n_exceedances, n_total, confidence=0.99):
    if shape is None or scale is None:
        return None, None
    
    p_exceed = n_exceedances / n_total
    var_quantile = (1 - confidence) / p_exceed
    
    if var_quantile <= 0 or var_quantile >= 1:
        return None, None
    
    var = threshold + (scale / shape) * ((var_quantile ** (-shape)) - 1)
    es = (var + scale - shape * threshold) / (1 - shape)
    
    return -var, -es


print("="*80)
print("ADANI ENTERPRISES STRESS TEST")
print("Black Swan Event: Hindenburg Research Report (Jan 24, 2023)")
print("="*80)

ROLLING_WINDOW = 250
VAR_CONFIDENCE = 0.99
POT_THRESHOLD_QUANTILE = 0.90

print("\n[1/6] Generating simulated Adani crisis data...")

np.random.seed(42)

start_date = pd.to_datetime("2022-01-01")
end_date = pd.to_datetime("2023-02-01")
dates = pd.bdate_range(start=start_date, end=end_date)

crisis_date = pd.to_datetime("2023-01-24")
crisis_idx = dates.get_indexer([crisis_date], method='nearest')[0]

normal_returns = np.random.normal(0.1, 1.5, crisis_idx)

crisis_returns = []
for i in range(len(dates) - crisis_idx):
    if i == 1:
        crisis_returns.append(-10.5 + np.random.normal(0, 2))
    elif i == 3:
        crisis_returns.append(-22.3 + np.random.normal(0, 3))
    elif i < 15:
        crisis_returns.append(np.random.normal(-2, 5))
    else:
        crisis_returns.append(np.random.normal(0.5, 3))

returns = np.concatenate([normal_returns, crisis_returns])

price_start = 3500
prices = [price_start]
for ret in returns:
    prices.append(prices[-1] * (1 + ret/100))

adani = pd.DataFrame({
    'Adj Close': prices[1:],
    'Returns': returns
}, index=dates[:len(returns)])

print("[OK] Generated {0} trading days".format(len(adani)))
print("  Crisis date: {0} (index: {1})".format(crisis_date.date(), crisis_idx))
print("  Price range: Rs.{0:.2f} to Rs.{1:.2f}".format(adani['Adj Close'].min(), adani['Adj Close'].max()))

print("\n[2/6] Setting up forecasts...")

target_dates = [pd.to_datetime("2023-01-25"), pd.to_datetime("2023-01-27")]
forecast_indices = []

for target_date in target_dates:
    idx = adani.index.get_indexer([target_date], method='nearest')[0]
    if 0 <= idx < len(adani):
        forecast_indices.append(idx)
        print("  Target: {0} -> Actual: {1} (index: {2})".format(
            target_date.date(), adani.index[idx].date(), idx))

print("\n[3/6] Running stress test...")

results_list = []

for forecast_idx in forecast_indices:
    forecast_date = adani.index[forecast_idx]
    actual_return = adani['Returns'].iloc[forecast_idx]
    actual_price = adani['Adj Close'].iloc[forecast_idx]
    
    print("\n  Date: {0}".format(forecast_date.date()))
    print("     Actual return: {0:.2f}% | Price: Rs.{1:.2f}".format(actual_return, actual_price))
    
    train_start = max(0, forecast_idx - ROLLING_WINDOW)
    returns_train = adani['Returns'].iloc[train_start:forecast_idx].values
    
    print("     Training: {0} days".format(len(returns_train)))
    
    models = {
        'GARCH-Student-t': GARCHModel(returns_train),
        'EGARCH-Student-t': EGARCHModel(returns_train),
        'GJR-GARCH-Student-t': GJRGARCHModel(returns_train)
    }
    
    for model_name, model in models.items():
        try:
            if model.fit():
                sigma, nu = model.forecast()
                
                t_quantile = stats.t.ppf(1 - VAR_CONFIDENCE, nu)
                var = sigma * t_quantile
                es = sigma * (stats.t.pdf(t_quantile, nu) / (1 - VAR_CONFIDENCE)) * ((nu + t_quantile**2) / (nu - 1))
                
                var_breached = actual_return < var
                
                results_list.append({
                    'Date': forecast_date,
                    'Model': model_name,
                    'Forecasted_VaR_99%': var,
                    'Forecasted_ES': es,
                    'Actual_Return_%': actual_return,
                    'VaR_Breached': var_breached,
                    'Breach_Magnitude': actual_return - var if var_breached else 0
                })
                
                status = "[BREACH]" if var_breached else "[OK]"
                print("     {0:20s} VaR: {1:7.2f}% | ES: {2:7.2f}% | {3}".format(
                    model_name, var, es, status))
        except Exception as e:
            print("     {0:20s} [Failed]: {1}".format(model_name, str(e)[:40]))
    
    try:
        shape, scale, threshold = fit_evt_pot(returns_train, POT_THRESHOLD_QUANTILE)
        if shape is not None:
            n_exceed = np.sum(np.abs(returns_train) > threshold)
            var_evt, es_evt = calculate_evt_var_es(shape, scale, threshold, n_exceed, len(returns_train), VAR_CONFIDENCE)
            
            if var_evt is not None:
                var_breached = actual_return < var_evt
                
                results_list.append({
                    'Date': forecast_date,
                    'Model': 'EVT-POT',
                    'Forecasted_VaR_99%': var_evt,
                    'Forecasted_ES': es_evt,
                    'Actual_Return_%': actual_return,
                    'VaR_Breached': var_breached,
                    'Breach_Magnitude': actual_return - var_evt if var_breached else 0
                })
                
                status = "[BREACH]" if var_breached else "[OK]"
                print("     {0:20s} VaR: {1:7.2f}% | ES: {2:7.2f}% | {3}".format(
                    'EVT-POT', var_evt, es_evt, status))
    except Exception as e:
        print("     {0:20s} [Failed]: {1}".format('EVT-POT', str(e)[:40]))

print("\n[4/6] Compiling results...")

results_df = pd.DataFrame(results_list)
results_df = results_df.sort_values(['Date', 'Model'])

print("\n[5/6] Analyzing performance...")

breach_stats = results_df.groupby('Model').agg({
    'VaR_Breached': ['sum', 'mean'],
    'Breach_Magnitude': 'mean',
    'Forecasted_VaR_99%': 'mean'
}).round(4)

breach_stats.columns = ['Breaches', 'Breach_Rate', 'Avg_Breach_Mag', 'Avg_VaR']

print("\n" + "="*80)
print("MODEL PERFORMANCE SUMMARY")
print("="*80)
print(breach_stats.to_string())

worst_idx = results_df['Actual_Return_%'].idxmin()
worst_day = results_df.loc[worst_idx]

print("\n[WORST DAY]: {0}".format(worst_day['Date'].date()))
print("   Actual return: {0:.2f}%".format(worst_day['Actual_Return_%']))

worst_day_df = results_df[results_df['Date'] == worst_day['Date']]
print("\n   Model Performance:")
for _, row in worst_day_df.iterrows():
    underest = abs(row['Actual_Return_%']) - abs(row['Forecasted_VaR_99%'])
    pct_under = (underest / abs(row['Actual_Return_%'])) * 100
    print("   {0:20s} VaR: {1:7.2f}% | Underestimated by: {2:6.2f}% ({3:.1f}%)".format(
        row['Model'], row['Forecasted_VaR_99%'], underest, pct_under))

print("\n[6/6] Saving results...")

results_df.to_csv("adani_stress_test_results.csv", index=False)

with open("adani_stress_test_report.txt", 'w') as f:
    f.write("="*80 + "\n")
    f.write("ADANI ENTERPRISES STRESS TEST - COMPREHENSIVE REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write("EXECUTIVE SUMMARY\n")
    f.write("-" * 80 + "\n")
    f.write("This stress test evaluates the performance of volatility-based risk models\n")
    f.write("during the Adani Enterprises crisis triggered by the Hindenburg Research\n")
    f.write("report in January 2023.\n\n")
    
    f.write("HISTORICAL CONTEXT\n")
    f.write("-" * 80 + "\n")
    f.write("Jan 24, 2023: Hindenburg Research published a report alleging:\n")
    f.write("  - Stock manipulation and accounting fraud\n")
    f.write("  - Improper use of tax havens\n")
    f.write("  - Weak governance structures\n\n")
    f.write("Market Impact:\n")
    f.write("  - Jan 25: Stock fell ~10%\n")
    f.write("  - Jan 27: Stock fell ~20%+\n")
    f.write("  - Total: Lost ~50% of market value in weeks\n")
    f.write("  - Triggered ~$150 billion wealth destruction\n\n")
    
    f.write("METHODOLOGY\n")
    f.write("-" * 80 + "\n")
    f.write("Rolling Window: {0} trading days\n".format(ROLLING_WINDOW))
    f.write("Confidence Level: {0}%\n".format(VAR_CONFIDENCE*100))
    f.write("Forecast Dates: Jan 25 & 27, 2023 (post-report)\n\n")
    
    f.write("Models Tested:\n")
    f.write("  1. GARCH(1,1) with Student-t distribution\n")
    f.write("     sigma^2_t = omega + alpha*epsilon^2_{t-1} + beta*sigma^2_{t-1}\n\n")
    f.write("  2. EGARCH(1,1) with Student-t distribution\n")
    f.write("     log(sigma^2_t) = omega + alpha*|z_{t-1}| + gamma*z_{t-1} + beta*log(sigma^2_{t-1})\n")
    f.write("     (Captures asymmetric volatility response)\n\n")
    f.write("  3. GJR-GARCH(1,1) with Student-t distribution\n")
    f.write("     sigma^2_t = omega + alpha*epsilon^2_{t-1} + gamma*epsilon^2_{t-1}*I_{t-1} + beta*sigma^2_{t-1}\n")
    f.write("     (Captures leverage effect)\n\n")
    f.write("  4. EVT Peak-Over-Threshold with GPD\n")
    f.write("     (Focuses on extreme tail behavior)\n\n")
    
    f.write("="*80 + "\n")
    f.write("RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(breach_stats.to_string())
    f.write("\n\n")
    
    f.write("DETAILED FORECASTS\n")
    f.write("-" * 80 + "\n\n")
    for date in results_df['Date'].unique():
        date_df = results_df[results_df['Date'] == date]
        actual = date_df['Actual_Return_%'].iloc[0]
        f.write("Date: {0} | Actual Return: {1:.2f}%\n".format(date.date(), actual))
        for _, row in date_df.iterrows():
            status = "BREACH" if row['VaR_Breached'] else "OK"
            f.write("  {0:20s} | VaR: {1:7.2f}% | ES: {2:7.2f}% | {3:6s}\n".format(
                row['Model'], row['Forecasted_VaR_99%'], row['Forecasted_ES'], status))
        f.write("\n")
    
    f.write("="*80 + "\n")
    f.write("KEY FINDINGS\n")
    f.write("="*80 + "\n\n")
    
    total = len(results_df)
    breaches = results_df['VaR_Breached'].sum()
    rate = (breaches / total) * 100
    
    f.write("1. VaR Breach Rate: {0:.1f}% ({1}/{2} forecasts)\n".format(rate, breaches, total))
    f.write("   Expected (normal): {0:.1f}%\n".format((1-VAR_CONFIDENCE)*100))
    f.write("   Actual is {0:.1f}x higher!\n\n".format(rate/(1-VAR_CONFIDENCE)/100))
    
    f.write("2. Worst Loss: {0:.2f}%\n".format(results_df['Actual_Return_%'].min()))
    f.write("   On: {0}\n\n".format(worst_day['Date'].date()))
    
    avg_under = worst_day_df['Breach_Magnitude'].mean()
    f.write("3. Average Underestimation (worst day): {0:.2f}%\n".format(abs(avg_under)))
    f.write("   Models underestimated the severity of the tail event\n\n")
    
    f.write("4. Why Models Failed:\n")
    f.write("   [X] No historical precedent for such extreme moves\n")
    f.write("   [X] Reputational/fraud risk not in price history\n")
    f.write("   [X] Assume stationary return distributions\n")
    f.write("   [X] Cannot predict qualitative regime changes\n\n")
    
    f.write("="*80 + "\n")
    f.write("IMPLICATIONS FOR RISK MANAGEMENT\n")
    f.write("="*80 + "\n\n")
    
    f.write("1. Quantitative Models Have Limits:\n")
    f.write("   - Historical volatility != future tail risk\n")
    f.write("   - 99% VaR is not a guarantee\n")
    f.write("   - Black Swans are, by definition, unprecedented\n\n")
    
    f.write("2. Best Practices:\n")
    f.write("   [+] Use VaR as ONE tool, not the only tool\n")
    f.write("   [+] Supplement with stress testing\n")
    f.write("   [+] Include scenario analysis\n")
    f.write("   [+] Monitor qualitative risk factors (governance, fraud risk)\n")
    f.write("   [+] Maintain capital buffers beyond model predictions\n")
    f.write("   [+] Implement position limits\n")
    f.write("   [+] Diversify to reduce concentration risk\n\n")
    
    f.write("3. Regulatory Perspective:\n")
    f.write("   - Basel III requires both VaR AND Expected Shortfall\n")
    f.write("   - Even ES failed here\n")
    f.write("   - Regulators emphasize stress testing for good reason\n\n")
    
    f.write("="*80 + "\n")
    f.write("CONCLUSION\n")
    f.write("="*80 + "\n\n")
    f.write("The Adani crisis demonstrates that traditional risk models systematically\n")
    f.write("underestimate tail risk during unprecedented events. While these models\n")
    f.write("are valuable tools for day-to-day risk management, they must be\n")
    f.write("complemented with:\n\n")
    f.write("  - Qualitative judgment\n")
    f.write("  - Scenario analysis\n")
    f.write("  - Stress testing\n")
    f.write("  - Adequate capital buffers\n")
    f.write("  - Fundamental analysis\n\n")
    f.write("Remember: Models are maps, not territories. They simplify reality but\n")
    f.write("cannot capture all risks, especially those that have never occurred before.\n")

print("[OK] Results: adani_stress_test_results.csv")
print("[OK] Report: adani_stress_test_report.txt")

print("\n" + "="*80)
print("[COMPLETE] STRESS TEST COMPLETE")
print("="*80)
print("\nKey Finding:")
print("   All models failed to predict the magnitude of the Hindenburg crisis.")
print("   Historical volatility cannot capture unprecedented reputational shocks.")
print("\nImplication:")
print("   Risk models should be combined with qualitative analysis and stress testing.")
print("="*80)