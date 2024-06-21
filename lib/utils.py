import numpy as np
import pandas as pd

from typing import Any, Optional


# Colors name -> hex mapping
colors = {
    'blue': '#1f77b4',
    'green': '#2ca02c',
    'orange': '#ff7f0e',
    'red': '#d62728',
    'purple': '#9467bd',
    'yellow': '#bcbd22',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'cyan': '#17becf',
    'grey': '#7f7f7f'
}


def get_color(
    color: str
) -> str:
    """
    Map the given color using the `colors` dictionary.

    Parameters
    ----------
    color : str
        Color key to retreive.
        If it does not exist, return the color itself.
    
    Returns
    -------
    str
        Color value.
    """
    return colors.get(color, color)


def show_summary(
    x: pd.DataFrame,
    tgt: Optional[str | pd.Series] = None,
    name: Optional[str] = None
) -> None:
    """
    Show a summary of a dataframe and its target.

    Parameters
    ----------
    x : pd.DataFrame
        Dataframe to summarize.
    tgt : str | pd.Series, optional
        Group data by this column and show the value counts.
    name : str, optional
        Name of the dataframe.
    """
    line = []

    if name is not None:
        line.append('{}'.format(name))

    line.append('Shape: {:,} x {:,}'.format(*x.shape))

    print(' | '.join(line))

    if tgt is not None:
        if isinstance(tgt, str):
            tgt = x[tgt]

        values = tgt.value_counts(dropna=False).sort_values(ascending=False)

        for v, t in values.items():
            r = 100 * t / values.sum()
            print('    {}: {:,} ({:0.2f}%)'.format(v, t, r))


def is_array(
    x: Any
) -> bool:
    """
    Check if a variable is an array-type or any kind of sequence.

    Parameters
    ----------
    x : Any
        Variable to check.
    
    Returns
    -------
    bool
        True if the variable is an array-type, False otherwise.
    """
    return isinstance(x, (list, tuple)) or (hasattr(x, '__array__') and hasattr(x, '__iter__'))


def format_number(
    n: int | float,
    decimals: Optional[int] = 2,
    dec_force: bool = False,
    dec_sep: str = ',',
    thou_sep: str = '.'
) -> str:
    """
    Format a number with the given parameters.

    Parameters
    ----------
    n : int | float
        Number to format.
    decimals : int, optional
        Number of decimals to show.
    dec_force : bool, default False
        If True, force the number of decimals to be at least the given number.
    dec_sep : str, default ','
        Decimal separator.
    thou_sep : str, default '.'
        Thousand separator.
    
    Returns
    -------
    str
        Formatted number.
    """
    if pd.isnull(n):
        return ''
    elif not isinstance(n, (int, float)):
        return str(n)

    try:
        x = float(n)
        if decimals is not None:
            x = round(x, decimals)

        parts = str(x).split('.')
        x_int = abs(int(parts[0])) if len(parts) > 0 else 0
        s_dec = parts[1] if len(parts) > 1 else '0'
    except (TypeError, ValueError):
        return str(n)

    s = '-' if x < 0 else ''

    s_int = '{:,}'.format(x_int).replace(',', thou_sep)
    s = '{}{}'.format(s, s_int)

    s_dec = s_dec.rstrip('0')
    if decimals is None:
        decimals = len(s_dec)
    if dec_force and len(s_dec) < decimals:
        s_dec = s_dec.ljust(decimals, '0')

    if s_dec != '':
        s = '{}{}{}'.format(s, dec_sep, s_dec)

    return s


def get_ewma_alpha(
    com: Optional[int] = None,
    span: Optional[int] = None,
    halflife: Optional[int] = None
) -> float:
    """
    Calculate the alpha parameter for the Exponential Weighted Moving Average (EWMA) function.

    Parameters
    ----------
    com : int, optional
        Number of observations to use for the calculation.
    span : int, optional
        Window span of each EMWA application.
    halflife : int, optional
        Number of periods for the exponential weight to decay by half.
    
    Returns
    -------
    float
        Alpha parameter for the EWMA function.
    """
    if com is not None:
        return 1 - 2 / (com + 1)
    elif span is not None:
        return 2 / (span + 1)
    elif halflife is not None:
        return 1 - 0.5 ** (1 / halflife)
    else:
        raise ValueError('One of [com], [span] or [halflife] must be provided.')

def ewma(
    x: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Calculate the Exponential Weighted Moving Average (EWMA) of a data series.

    Parameters
    ----------
    x : array_like
        Data series.
    alpha : float
        Alpha decay parameter for the EWMA function.
    
    Returns
    -------
    array_like
        EWMA of the data series.
    """
    ds = np.array(x)
    n = ds.size

    # Create an initial weight matrix of (1 - alpha)
    w0 = np.ones(shape=(n, n)) * (1 - alpha)
    # And a matrix of powers to raise the weights by
    p = np.vstack([np.arange(i, i - n, -1) for i in np.arange(n)])
    # Create the weight matrix
    w = np.tril(np.power(w0, p), 0)

    return np.dot(w, ds) / w.sum(axis=1)


def pct_change(
    x: np.ndarray,
    n: int = 1,
    tol: float = 1e-9
) -> np.ndarray:
    """
    Calculate the percentage change of a data series.

    Parameters
    ----------
    x : array_like
        Data series.
    n : int, default 1
        Number of periods to shift for the calculation.
    tol : float, default 1e-9
        The maximum allowed difference between elements for them to be considered equal.
        The default tolerance is 1e-09, which assures that the two values are the same
        within about 9 decimal digits. Must be greater than zero.
    
    Returns
    -------
    array_like
        Percentage change of the data series.
    """
    ds = np.full_like(x, np.NaN)
    ds[n:] = [i if abs(i) > tol else 0 for i in np.diff(x, n) / x[:-n]]

    return ds


def momentum(
    x: np.ndarray,
    deg: int = 3,
    span: int = 15,
    log: bool = False
) -> np.ndarray:
    """
    Calculate the momentum of a time series data.
    It measures the rate of change of a recurrent Exponential Weighted Moving Average (EWMA).
    If `deg` equals 3, this is called 'Trix' (from 'triple exponential') in technical analysis.

    A rising or falling line is an uptrend or downtrend and Momentum shows the slope of that line,
    so it's positive for a steady uptrend, negative for a downtrend, and a crossing through zero is a trend-change,
    i.e. a peak or trough in the underlying average.

    Parameters
    ----------
    d : np.ndarray
        Data series.
    deg : int, default 3
        Number of times to sequentially apply the EWMA.
    span : int, default 15
        Window span of each EMWA application.
    log : bool, default False
        If True, apply log transformation to the data.
    
    Returns
    -------
    Series
        Momentum of the time series data.
    """
    ds = np.array(x)

    # If the series has negative values, add a constant to make it positive
    n_add = ds.min() + 1 if ds.min() <= 0 else 0
    ds = ds + n_add

    if log:
        ds = np.log1p(ds)
    
    # Exponencial Moving Average (applied <deg> times)
    alpha = get_ewma_alpha(span=span)
    for _ in np.arange(deg):
        ds = ewma(ds, alpha)
    
    ds = 100 * pct_change(ds)

    return ds
