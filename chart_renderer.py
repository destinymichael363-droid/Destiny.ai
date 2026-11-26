"""
chart_renderer.py — TradingView-style candlestick chart for DESTINY AI.

Draws on a single Plotly figure:
  - Candlesticks (dark theme)
  - Volume subplot
  - Support / Resistance zones (SRZone list)
  - OB + FVG paired zones (PairedZone list)
  - Entry / SL / TP lines (optional trade)
  - Liquidity sweep markers (optional)
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------- #
# Colour palette (dark theme)
# ---------------------------------------------------------------- #

COLOR_BG = "#0e1117"
COLOR_GRID = "#1f2530"
COLOR_TEXT = "#d1d4dc"

COLOR_UP = "#26a69a"
COLOR_DOWN = "#ef5350"

COLOR_SR_SUPPORT = "rgba(38, 166, 154, 0.18)"
COLOR_SR_SUPPORT_LINE = "rgba(38, 166, 154, 0.75)"

COLOR_SR_RESISTANCE = "rgba(239, 83, 80, 0.18)"
COLOR_SR_RESISTANCE_LINE = "rgba(239, 83, 80, 0.75)"

COLOR_OB_BULL = "rgba(38, 166, 154, 0.12)"
COLOR_OB_BULL_LINE = "rgba(38, 166, 154, 0.6)"

COLOR_OB_BEAR = "rgba(239, 83, 80, 0.12)"
COLOR_OB_BEAR_LINE = "rgba(239, 83, 80, 0.6)"

COLOR_FVG_BULL = "rgba(96, 165, 250, 0.15)"
COLOR_FVG_BULL_LINE = "rgba(96, 165, 250, 0.7)"

COLOR_FVG_BEAR = "rgba(250, 204, 21, 0.15)"
COLOR_FVG_BEAR_LINE = "rgba(250, 204, 21, 0.7)"

COLOR_ENTRY = "#facc15"
COLOR_SL = "#ef5350"
COLOR_TP = "#26a69a"

COLOR_SWEEP_HIGH = "rgba(250, 204, 21, 0.9)"
COLOR_SWEEP_LOW = "rgba(250, 204, 21, 0.9)"


# ---------------------------------------------------------------- #
# Base chart
# ---------------------------------------------------------------- #

def _empty_figure(message: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        height=600,
        annotations=[
            dict(
                text=message,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color=COLOR_TEXT),
            )
        ],
    )
    return fig


def _build_base(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
) -> go.Figure:
    """Candlestick + volume subplot with dark theme."""
    if df is None or df.empty:
        return _empty_figure(f"No data for {symbol}")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.78, 0.22],
        subplot_titles=("", ""),
    )

    # --- Candlesticks ---
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            increasing=dict(
                line=dict(color=COLOR_UP, width=1),
                fillcolor=COLOR_UP,
            ),
            decreasing=dict(
                line=dict(color=COLOR_DOWN, width=1),
                fillcolor=COLOR_DOWN,
            ),
            name="Price",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # --- Volume ---
    vol_colors = [
        COLOR_UP if c >= o else COLOR_DOWN
        for o, c in zip(df["Open"], df["Close"])
    ]

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            marker_color=vol_colors,
            marker_line_width=0,
            name="Volume",
            showlegend=False,
            opacity=0.6,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        height=700,
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(
            text=f"<b>{symbol}</b>  ·  {timeframe}",
            font=dict(size=16, color=COLOR_TEXT),
            x=0.01,
        ),
        hovermode="x unified",
        dragmode="pan",
        xaxis_rangeslider_visible=False,
        showlegend=False,
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=COLOR_GRID,
        zeroline=False,
        linecolor=COLOR_GRID,
        tickfont=dict(color=COLOR_TEXT),
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=COLOR_GRID,
        zeroline=False,
        linecolor=COLOR_GRID,
        tickfont=dict(color=COLOR_TEXT),
        side="right",
    )

    fig.update_yaxes(
        title_text="",
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text="",
        row=2,
        col=1,
    )

    return fig


# ---------------------------------------------------------------- #
# Overlays
# ---------------------------------------------------------------- #

def _x_range(df: pd.DataFrame) -> tuple[Any, Any]:
    """Return (first_ts, last_ts) for shaping overlays."""
    return df.index[0], df.index[-1]


def _overlay_sr_zones(
    fig: go.Figure,
    df: pd.DataFrame,
    sr_zones: list,
) -> None:
    """Draw support / resistance bands across the whole chart."""
    if not sr_zones:
        return

    x0, x1 = _x_range(df)

    for zone in sr_zones:
        zone_type = getattr(zone, "zone_type", "").lower()
        bottom = float(getattr(zone, "bottom", 0.0))
        top = float(getattr(zone, "top", 0.0))

        if top <= bottom:
            continue

        if zone_type == "support":
            fill = COLOR_SR_SUPPORT
            line = COLOR_SR_SUPPORT_LINE
            label = "S/R Support"
        elif zone_type == "resistance":
            fill = COLOR_SR_RESISTANCE
            line = COLOR_SR_RESISTANCE_LINE
            label = "S/R Resistance"
        else:
            continue

        # Rectangle via filled scatter
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=x0,
            x1=x1,
            y0=bottom,
            y1=top,
            fillcolor=fill,
            line=dict(color=line, width=1),
            layer="below",
            row=1,
            col=1,
        )

        # Label the band on the right edge
        fig.add_annotation(
            x=x1,
            y=(bottom + top) / 2.0,
            xref="x",
            yref="y",
            text=f"{label} {bottom:.5g}–{top:.5g}",
            showarrow=False,
            xanchor="right",
            yanchor="middle",
            font=dict(size=9, color=line),
            bgcolor="rgba(0,0,0,0.4)",
            row=1,
            col=1,
        )


def _overlay_paired_zones(
    fig: go.Figure,
    df: pd.DataFrame,
    paired_zones: list,
) -> None:
    """
    Draw OB + FVG rectangles for PairedZone objects.

    The outer box (dashed) shows the OB+FVG envelope.
    The inner solid box shows the overlap (the actual zone).
    """
    if not paired_zones:
        return

    x0, x1 = _x_range(df)

    for pz in paired_zones:
        direction = int(getattr(pz, "direction", 0))
        if direction == 0:
            continue

        # Full OB+FVG range (outer)
        outer_low = float(min(getattr(pz, "ob_low", 0.0), getattr(pz, "fvg_low", 0.0)))
        outer_high = float(max(getattr(pz, "ob_high", 0.0), getattr(pz, "fvg_high", 0.0)))

        # Overlap zone (inner solid)
        inner_low = float(min(getattr(pz, "low", 0.0), getattr(pz, "high", 0.0)))
        inner_high = float(max(getattr(pz, "low", 0.0), getattr(pz, "high", 0.0)))

        if outer_high <= outer_low:
            continue

        if direction > 0:
            inner_fill = COLOR_FVG_BULL
            inner_line = COLOR_FVG_BULL_LINE
            outer_fill = COLOR_OB_BULL
            outer_line = COLOR_OB_BULL_LINE
            label = "OB+FVG Bull"
        else:
            inner_fill = COLOR_FVG_BEAR
            inner_line = COLOR_FVG_BEAR_LINE
            outer_fill = COLOR_OB_BEAR
            outer_line = COLOR_OB_BEAR_LINE
            label = "OB+FVG Bear"

        # Outer OB+FVG envelope (dashed border, faint fill)
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=x0,
            x1=x1,
            y0=outer_low,
            y1=outer_high,
            fillcolor=outer_fill,
            line=dict(color=outer_line, width=1, dash="dot"),
            layer="below",
            row=1,
            col=1,
        )

        # Inner overlap zone (solid, brighter)
        if inner_high > inner_low:
            fig.add_shape(
                type="rect",
                xref="x",
                yref="y",
                x0=x0,
                x1=x1,
                y0=inner_low,
                y1=inner_high,
                fillcolor=inner_fill,
                line=dict(color=inner_line, width=1),
                layer="below",
                row=1,
                col=1,
            )

        # Label
        fig.add_annotation(
            x=x1,
            y=(outer_low + outer_high) / 2.0,
            xref="x",
            yref="y",
            text=f"{label} {inner_low:.5g}–{inner_high:.5g}",
            showarrow=False,
            xanchor="right",
            yanchor="middle",
            font=dict(size=9, color=inner_line),
            bgcolor="rgba(0,0,0,0.4)",
            row=1,
            col=1,
        )


def _overlay_trade_levels(
    fig: go.Figure,
    df: pd.DataFrame,
    entry: float | None,
    sl: float | None,
    tp: float | None,
    direction: str = "",
) -> None:
    """Draw entry / stop-loss / take-profit horizontal lines."""
    x0, x1 = _x_range(df)

    def _hline(y: float, color: str, label: str) -> None:
        if y is None or y <= 0:
            return
        fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=x0,
            x1=x1,
            y0=float(y),
            y1=float(y),
            line=dict(color=color, width=1.5, dash="dash"),
            layer="above",
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=x1,
            y=float(y),
            xref="x",
            yref="y",
            text=f"{label} {y:.5g}",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=10, color=color),
            bgcolor="rgba(0,0,0,0.6)",
            row=1,
            col=1,
        )

    _hline(entry, COLOR_ENTRY, "Entry")
    _hline(sl, COLOR_SL, "SL")
    _hline(tp, COLOR_TP, "TP")

    if direction:
        fig.add_annotation(
            x=x0,
            y=1.0,
            xref="x",
            yref="paper",
            text=f"{direction}",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=12, color=COLOR_ENTRY),
            row=1,
            col=1,
        )


def _overlay_sweeps(
    fig: go.Figure,
    df: pd.DataFrame,
    sweep_series: pd.Series | None,
) -> None:
    """
    Mark bars where liquidity was swept (swept_liquidity_count > 0).

    Sweeps above the bar = liquidity high sweep.
    Sweeps below the bar = liquidity low sweep.
    """
    if sweep_series is None or sweep_series.empty:
        return

    try:
        aligned = sweep_series.reindex(df.index).fillna(0)
    except Exception:
        return

    hit_mask = aligned > 0

    if not hit_mask.any():
        return

    hit_times = aligned.index[hit_mask]
    hit_highs = df.loc[hit_times, "High"]
    hit_lows = df.loc[hit_times, "Low"]

    # Small markers above the bar for high sweeps
    fig.add_trace(
        go.Scatter(
            x=hit_times,
            y=hit_highs * 1.0005,
            mode="markers",
            marker=dict(
                symbol="triangle-down",
                size=8,
                color=COLOR_SWEEP_HIGH,
                line=dict(width=0),
            ),
            name="Sweep",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=hit_times,
            y=hit_lows * 0.9995,
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                size=8,
                color=COLOR_SWEEP_LOW,
                line=dict(width=0),
            ),
            name="Sweep",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )


# ---------------------------------------------------------------- #
# Public API
# ---------------------------------------------------------------- #

def make_chart(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str = "15M",
    sr_zones: list | None = None,
    paired_zones: list | None = None,
    entry: float | None = None,
    sl: float | None = None,
    tp: float | None = None,
    direction: str = "",
    sweep_series: pd.Series | None = None,
    show_sr: bool = True,
    show_ob_fvg: bool = True,
    show_trade: bool = True,
    show_sweeps: bool = True,
) -> go.Figure:
    """
    Build a complete chart.

    Parameters
    ----------
    df : OHLCV DataFrame from data_loader.fetch_ohlcv
    symbol : display label (e.g. "Gold (XAU)")
    timeframe : display label (e.g. "15M")
    sr_zones : list of SRZone objects (from run_sr_zones_engine)
    paired_zones : list of PairedZone objects (from run_ob_fvg_engine)
    entry, sl, tp : trade levels to draw (optional)
    direction : "LONG" / "SHORT" / ""
    sweep_series : per-bar swept_liquidity_count (from signals_df)
    show_* : toggles for each overlay group

    Returns a Plotly Figure ready for st.plotly_chart.
    """
    fig = _build_base(df, symbol, timeframe)

    if df is None or df.empty:
        return fig

    if show_sr and sr_zones:
        _overlay_sr_zones(fig, df, sr_zones)

    if show_ob_fvg and paired_zones:
        _overlay_paired_zones(fig, df, paired_zones)

    if show_sweeps:
        _overlay_sweeps(fig, df, sweep_series)

    if show_trade and (entry or sl or tp):
        _overlay_trade_levels(fig, df, entry, sl, tp, direction)

    return fig
