
def display_chart(data, result, symbol, timeframe):

    # Keep the chart focused on recent price action.
    chart_data = data.tail(150).copy()

    if chart_data.empty:
        st.warning("No candles available for this chart.")
        return

    start_time = chart_data.index[0]
    end_time = chart_data.index[-1]

    fig = go.Figure()

    # Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=chart_data.index,
            open=chart_data["Open"],
            high=chart_data["High"],
            low=chart_data["Low"],
            close=chart_data["Close"],
            name="Price",
            increasing_line_color="#00C087",
            decreasing_line_color="#F04452",
            increasing_fillcolor="#00C087",
            decreasing_fillcolor="#F04452",
            whiskerwidth=0.8
        )
    )

    # OB and FVG zones.
    setups = result.get("setups", pd.DataFrame())

    if not setups.empty:

        for _, setup in setups.tail(30).iterrows():

            ob_time = setup.get("ob_time")

            if pd.isna(ob_time):
                continue

            ob_time = pd.Timestamp(ob_time)

            # Avoid timezone comparison errors.
            if start_time.tzinfo is not None and ob_time.tzinfo is None:
                ob_time = ob_time.tz_localize(start_time.tzinfo)
            elif start_time.tzinfo is None and ob_time.tzinfo is not None:
                ob_time = ob_time.tz_localize(None)

            # Ignore zones outside the visible chart period.
            if ob_time < start_time or ob_time > end_time:
                continue

            bullish = setup.get("direction") == "BULLISH"

            color = (
                "rgba(0,192,135,0.13)"
                if bullish
                else "rgba(240,68,82,0.13)"
            )

            fig.add_shape(
                type="rect",
                x0=ob_time,
                x1=end_time,
                y0=setup["ob_low"],
                y1=setup["ob_high"],
                fillcolor=color,
                line=dict(width=1),
                layer="below"
            )

            fvg_time = setup.get("fvg_time")

            if pd.notna(fvg_time):

                fvg_time = pd.Timestamp(fvg_time)

                if start_time.tzinfo is not None and fvg_time.tzinfo is None:
                    fvg_time = fvg_time.tz_localize(start_time.tzinfo)
                elif start_time.tzinfo is None and fvg_time.tzinfo is not None:
                    fvg_time = fvg_time.tz_localize(None)

                if start_time <= fvg_time <= end_time:

                    fig.add_shape(
                        type="rect",
                        x0=fvg_time,
                        x1=end_time,
                        y0=setup["fvg_low"],
                        y1=setup["fvg_high"],
                        fillcolor="rgba(245,190,55,0.13)",
                        line=dict(
                            color="rgba(245,190,55,0.8)",
                            width=1
                        ),
                        layer="below"
                    )

    # Active liquidity levels.
    # No repeated labels: this prevents BSL/SSL text overlap.
    liquidity = result.get("liquidity", pd.DataFrame())

    if not liquidity.empty and "swept" in liquidity.columns:

        active = liquidity[
            liquidity["swept"] == False
        ].tail(6)

        for _, level in active.iterrows():

            price = level.get("price")

            if pd.isna(price):
                continue

            side = level.get("side")

            color = (
                "#F04452"
                if side == "BSL"
                else "#00C087"
            )

            fig.add_hline(
                y=price,
                line_dash="dot",
                line_color=color,
                line_width=1
            )

    fig.update_layout(
        title=dict(
            text=f"{symbol} | {timeframe}",
            font=dict(size=17)
        ),
        height=480,
        autosize=True,
        template="plotly_dark",
        margin=dict(
            l=10,
            r=15,
            t=50,
            b=35
        ),
        font=dict(size=13),
        xaxis=dict(
            title="Time",
            type="date",
            range=[start_time, end_time],
            rangeslider=dict(visible=False),
            tickformat="%d %b",
            nticks=6,
            tickfont=dict(size=11),
            showgrid=True,
            gridcolor="rgba(130,130,130,0.15)",
            automargin=True
        ),
        yaxis=dict(
            title="Price",
            side="right",
            autorange=True,
            tickfont=dict(size=12),
            showgrid=True,
            gridcolor="rgba(130,130,130,0.15)",
            automargin=True
        ),
        showlegend=False,
        dragmode="pan"
    )

    st.caption(
        "Green zones: bullish OB | Red zones: bearish OB | "
        "Yellow zones: FVG | Red dotted lines: BSL | "
        "Green dotted lines: SSL"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "responsive": True
        }
    )
