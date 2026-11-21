
# =========================================================
# DESTINY AI - MULTI-TIMEFRAME CONTEXT ENGINE
# Historical structure comparison only
# =========================================================

TIMEFRAME_ORDER = ["1D", "4H", "1H", "15M", "5M", "1M"]


def classify_structure(bullish_bos, bearish_bos):
    """
    Classifies historical BOS balance.
    This is a statistical description, not a trade signal.
    """

    bullish_bos = int(bullish_bos or 0)
    bearish_bos = int(bearish_bos or 0)

    total = bullish_bos + bearish_bos

    if total == 0:
        return "No BOS recorded"

    if bullish_bos > bearish_bos:
        return "Bullish structure evidence"

    if bearish_bos > bullish_bos:
        return "Bearish structure evidence"

    return "Balanced BOS history"


def get_candle_coverage(result):
    """
    Returns candle count and historical date coverage.
    """

    candles = result.get("candles")

    if candles is None:
        return {
            "candle_count": 0,
            "first_candle": "Unavailable",
            "last_candle": "Unavailable"
        }

    try:
        count = len(candles)

        if count == 0:
            return {
                "candle_count": 0,
                "first_candle": "Unavailable",
                "last_candle": "Unavailable"
            }

        index = candles.index

        return {
            "candle_count": count,
            "first_candle": str(index[0]),
            "last_candle": str(index[-1])
        }

    except (AttributeError, TypeError, IndexError):
        return {
            "candle_count": 0,
            "first_candle": "Unavailable",
            "last_candle": "Unavailable"
        }


def build_mtf_context(analyses):
    """
    Combines independent historical timeframe analysis.

    Reports:
    - BOS counts
    - Historical structure classification
    - Candle sample size
    - Historical date coverage
    - Higher/lower timeframe summaries

    Does not generate entry signals or trading instructions.
    """

    timeframe_data = {}
    available = []

    for timeframe in TIMEFRAME_ORDER:

        item = analyses.get(timeframe, {})

        if item.get("status") != "ANALYSED":

            timeframe_data[timeframe] = {
                "status": "Unavailable",
                "bullish_bos": 0,
                "bearish_bos": 0,
                "structure": "Insufficient data",
                "candle_count": 0,
                "first_candle": "Unavailable",
                "last_candle": "Unavailable"
            }

            continue

        result = item.get("data", {})
        statistics = result.get("statistics", {})

        bullish = int(
            statistics.get("bullish_bos", 0) or 0
        )

        bearish = int(
            statistics.get("bearish_bos", 0) or 0
        )

        structure = classify_structure(
            bullish,
            bearish
        )

        coverage = get_candle_coverage(result)

        timeframe_data[timeframe] = {
            "status": "Available",
            "bullish_bos": bullish,
            "bearish_bos": bearish,
            "structure": structure,
            "candle_count": coverage["candle_count"],
            "first_candle": coverage["first_candle"],
            "last_candle": coverage["last_candle"]
        }

        available.append(timeframe)

    # -----------------------------------------------------
    # HIGHER AND LOWER TIMEFRAME GROUPS
    # -----------------------------------------------------

    higher = ["1D", "4H", "1H"]
    lower = ["15M", "5M", "1M"]

    higher_labels = [
        timeframe_data[tf]["structure"]
        for tf in higher
        if timeframe_data[tf]["status"] == "Available"
    ]

    lower_labels = [
        timeframe_data[tf]["structure"]
        for tf in lower
        if timeframe_data[tf]["status"] == "Available"
    ]

    def group_summary(labels):

        bullish = labels.count(
            "Bullish structure evidence"
        )

        bearish = labels.count(
            "Bearish structure evidence"
        )

        balanced = labels.count(
            "Balanced BOS history"
        )

        none = labels.count(
            "No BOS recorded"
        )

        if not labels:
            return "Insufficient timeframe data"

        if bullish > bearish:
            result = (
                "More timeframes show bullish BOS history"
            )

        elif bearish > bullish:
            result = (
                "More timeframes show bearish BOS history"
            )

        else:
            result = (
                "Mixed or balanced structure history"
            )

        return (
            f"{result} | "
            f"Bullish: {bullish}, "
            f"Bearish: {bearish}, "
            f"Balanced: {balanced}, "
            f"No BOS: {none}"
        )

    # -----------------------------------------------------
    # FINAL OUTPUT
    # -----------------------------------------------------

    return {
        "timeframes": timeframe_data,

        "available_timeframes": available,

        "higher_timeframe_summary": group_summary(
            higher_labels
        ),

        "lower_timeframe_summary": group_summary(
            lower_labels
        ),

        "data_coverage": (
            f"{len(available)} of "
            f"{len(TIMEFRAME_ORDER)} timeframes available"
        )
    }
