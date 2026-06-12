import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
import sqlite3
import numpy as np
import math

def ols_exposures(returns: pd.Series, factors: pd.DataFrame) -> pd.Series:
    """
    Full-sample OLS of stock returns on macro factors.
    Returns the beta vector (factor loadings), excluding the intercept.
    """
    y = returns.dropna()
    X = sm.add_constant(factors.loc[y.index])
    res = sm.OLS(y, X).fit()
    betas = res.params.drop('const')
    betas.name = returns.name
    beta_var = res.bse.drop('const')
    return betas, beta_var

def ols_rolling_exposures(returns: pd.Series, factors: pd.DataFrame,
                          window: int, n_components: int) -> pd.DataFrame:
    """
    Rolling OLS — returns a DataFrame of betas over time.
    """
    results = []
    for end in range(window, len(returns) + 1):
        y = returns.iloc[end - window:end]
        X = factors.iloc[end - window:end]
        try:
            pca = PCA(n_components = n_components)
            PC_scores = pd.DataFrame(
                pca.fit_transform(X),
                index=X.index,
                columns=[f'PC{i+1}' for i in range(n_components)]
            )
            betas = pca_exposures(y, PC_scores, pca, factors.columns, n_components)[0]
            # res = sm.OLS(y, X).fit()
            betas.name = returns.index[end - 1]
            results.append(betas)
            
        except Exception:
            pass
    return pd.DataFrame(results)

def pca_exposures(returns: pd.Series, pc_scores: pd.DataFrame,
                  pca_model: PCA, factor_names: list, N_COMPONENTS: int) -> pd.Series:
    """
    1. OLS of returns on PCs
    2. Back-transform PC gammas → original factor betas via W @ gamma
       where W = pca.components_.T  (shape: n_factors × n_components)
    """
    y = returns.dropna()
    X = sm.add_constant(pc_scores.loc[y.index])
    res = sm.OLS(y, X).fit()
    gammas = res.params.drop('const').values          # shape (n_components,) 

    W = pca_model.components_.T                       # shape (n_factors, n_components)
    betas = W @ gammas                                # shape (n_factors,)

    gamma_cov = res.cov_params().drop('const').drop('const', axis=1).values  # (n_components, n_components)
    beta_cov = W @ gamma_cov @ W.T                   # (n_factors, n_factors)
    beta_se = np.sqrt(np.diag(beta_cov))             # (n_factors,)

    return (pd.Series(betas, index=factor_names, name=returns.name), 
            pd.Series(beta_se, index=factor_names, name=returns.name), 
            pd.Series(gammas, index=[f"PCA{k + 1}" for k in range(N_COMPONENTS)], name=returns.name))

def load_monthly_returns(ts_code: str, prices_conn: sqlite3.Connection) -> pd.Series:
    """
    Load adjusted daily closes from prices.sqlite, resample to month-end,
    compute log returns. Tries both 'adj_close' and 'close' columns.
    """
    # try a unified daily table first

    df_price = pd.read_sql(
        f"SELECT trade_date, close FROM daily_prices "
        f"WHERE ts_code=? ORDER BY trade_date",
        prices_conn, params=(ts_code,), parse_dates = {'trade_date': '%Y%m%d'}
    ).set_index('trade_date')['close']

    df_adj = pd.read_sql(
        f"SELECT trade_date, adj_factor FROM adj_factor " 
        f"WHERE ts_code=? ORDER BY trade_date",
        prices_conn, params=(ts_code,), parse_dates = {'trade_date': '%Y%m%d'}
    ).set_index('trade_date')['adj_factor']

    if df_price is None or df_price.empty:
        raise ValueError(f'Could not load prices for {ts_code}. '
                         f'Check STOCK_UNIVERSE codes match your prices.sqlite schema.')

    # month-end resample → log returns
    monthly_price = df_price.resample('ME').last()
    monthly_adj = df_adj.resample('ME').last()
    monthly = monthly_price * monthly_adj  # adjust close by factor
    log_ret = np.log(monthly / monthly.shift(1)).dropna()
    log_ret.name = ts_code
    return log_ret

def residual_diagnostics(y: pd.Series, x: pd.Series) -> dict | None:
    pair = pd.concat([y, x], axis=1).dropna()
    if len(pair) < 500:
        return None

    yv = pair.iloc[:, 0].to_numpy(dtype=float)
    xv = pair.iloc[:, 1].to_numpy(dtype=float)
    x_design = np.column_stack([np.ones(len(xv)), xv])
    alpha, beta = np.linalg.lstsq(x_design, yv, rcond=None)[0]
    spread = yv - (alpha + beta * xv)
    spread_std = spread.std(ddof=1)
    if not np.isfinite(spread_std) or spread_std == 0:
        return None

    spread_lag = spread[:-1]
    delta = np.diff(spread)
    adf_design = np.column_stack([np.ones(len(spread_lag)), spread_lag])
    coef = np.linalg.lstsq(adf_design, delta, rcond=None)[0]
    resid = delta - adf_design @ coef
    dof = len(delta) - adf_design.shape[1]
    if dof <= 0:
        return None
    sigma2 = (resid @ resid) / dof
    xtx_inv = np.linalg.inv(adf_design.T @ adf_design)
    se_gamma = math.sqrt(sigma2 * xtx_inv[1, 1])
    gamma = coef[1]
    adf_t = gamma / se_gamma if se_gamma else np.nan

    phi = 1 + gamma
    half_life = -np.log(2) / np.log(phi) if 0 < phi < 1 else np.nan
    z_last = (spread[-1] - spread.mean()) / spread_std

    return {
        "n_obs": len(pair),
        "alpha": alpha,
        "beta": beta,
        "spread_std": spread_std,
        "adf_t_approx": adf_t,
        "phi": phi,
        "half_life_days": half_life,
        "last_zscore": z_last,
    }

