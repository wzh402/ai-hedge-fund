from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
import json
from datetime import datetime, timedelta  # Added timedelta and datetime

from src.tools.api import get_financial_metrics, get_prices  # Ensure get_prices is imported


##### Fundamental Agent #####
def fundamentals_agent(state: AgentState):
    """Analyzes fundamental data and generates trading signals for multiple tickers."""
    data = state["data"]
    end_date = data["end_date"]  # This end_date changes weekly
    tickers = data["tickers"]
    metadata = state.get("metadata", {})
    data_source = metadata.get("data_source", "financialdatasets")

    fundamental_analysis = {}

    for ticker in tickers:
        progress.update_status("fundamentals_agent", ticker, f"Fetching financial metrics for end_date: {end_date}")

        # Get the financial metrics - should be sensitive to end_date for caching
        financial_metrics_list = get_financial_metrics(
            ticker=ticker,
            end_date=end_date,
            period="ttm",
            limit=1,  # Fetch the single most recent TTM record relative to end_date
            data_source=data_source,
        )

        if not financial_metrics_list:
            progress.update_status("fundamentals_agent", ticker, f"Failed: No financial metrics found for {end_date}")
            fundamental_analysis[ticker] = {
                "signal": "neutral",
                "confidence": 0,
                "reasoning": {"error": f"No financial metrics found for {ticker} as of {end_date}"},
            }
            continue

        metrics = financial_metrics_list[0]

        progress.update_status("fundamentals_agent", ticker, f"Fetching current price for {ticker} at {end_date}")
        # Fetch current price for dynamic ratio calculation.
        # end_date is analysis_end_str. We need the price on the last trading day on or before this date.
        # Let's fetch prices for the 5 days leading up to and including end_date.

        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
        price_fetch_start_date_str = (end_date_dt - timedelta(days=5)).strftime("%Y-%m-%d")

        # api.get_prices should return sorted prices (chronologically).
        # The cache key for get_prices in api.py includes start_date and end_date.
        price_history_for_current = get_prices(
            ticker=ticker,
            start_date=price_fetch_start_date_str,  # Fetch a small window
            end_date=end_date,  # Up to analysis_end_str (e.g., Sunday)
            data_source=data_source,
        )

        current_price = None
        actual_price_date_str = "N/A"
        if price_history_for_current:
            # Prices should be sorted chronologically. The last one is the latest.
            latest_price_point = price_history_for_current[-1]
            current_price = latest_price_point.close
            if current_price is None:  # Fallback
                current_price = (latest_price_point.high + latest_price_point.low) / 2 if latest_price_point.high and latest_price_point.low else latest_price_point.open
            actual_price_date_str = latest_price_point.time  # Assuming 'time' attribute holds the date string
            progress.update_status("fundamentals_agent", ticker, f"Using price from {actual_price_date_str} (target end_date: {end_date})")

        if current_price is None:
            progress.update_status("fundamentals_agent", ticker, f"Warning: Could not fetch current price for {ticker} up to {end_date}. Price-dependent ratios will use metric's own P/E, P/B, P/S or be N/A.")
            # If current price is not available, use ratios from metrics if they exist, otherwise they will be None.
            pe_ratio = metrics.price_to_earnings_ratio
            pb_ratio = metrics.price_to_book_ratio
            ps_ratio = metrics.price_to_sales_ratio
        else:
            progress.update_status("fundamentals_agent", ticker, f"Current price for {ticker} at {end_date}: {current_price}")

        signals = []
        reasoning = {}

        progress.update_status("fundamentals_agent", ticker, "Analyzing profitability")
        # 1. Profitability Analysis (Ensure metrics used are from the fetched 'metrics' object)
        return_on_equity = metrics.return_on_equity_ttm if hasattr(metrics, "return_on_equity_ttm") else metrics.return_on_equity  # backward compatible
        net_margin = metrics.net_income_margin_ttm if hasattr(metrics, "net_income_margin_ttm") else metrics.net_margin
        operating_margin = metrics.operating_margin_ttm if hasattr(metrics, "operating_margin_ttm") else metrics.operating_margin

        thresholds_profitability = [
            (return_on_equity, 0.15),  # Strong ROE above 15%
            (net_margin, 0.10),  # Healthy profit margins
            (operating_margin, 0.15),  # Strong operating efficiency
        ]
        profitability_score = sum(metric is not None and metric > threshold for metric, threshold in thresholds_profitability)

        signals.append("bullish" if profitability_score >= 2 else "bearish" if profitability_score == 0 and any(m is not None for m, _ in thresholds_profitability) else "neutral")
        reasoning["profitability_signal"] = {
            "signal": signals[-1],
            "details": (f"ROE TTM: {return_on_equity:.2%}" if return_on_equity is not None else "ROE TTM: N/A") + ", " + (f"Net Margin TTM: {net_margin:.2%}" if net_margin is not None else "Net Margin TTM: N/A") + ", " + (f"Op Margin TTM: {operating_margin:.2%}" if operating_margin is not None else "Op Margin TTM: N/A"),
        }

        progress.update_status("fundamentals_agent", ticker, "Analyzing growth")
        # 2. Growth Analysis
        revenue_growth = metrics.revenue_growth_annual if hasattr(metrics, "revenue_growth_annual") and metrics.revenue_growth_annual is not None else metrics.revenue_growth_quarterly if hasattr(metrics, "revenue_growth_quarterly") else metrics.revenue_growth
        earnings_growth = metrics.earnings_growth_annual if hasattr(metrics, "earnings_growth_annual") and metrics.earnings_growth_annual is not None else metrics.earnings_growth_quarterly if hasattr(metrics, "earnings_growth_quarterly") else metrics.earnings_growth

        thresholds_growth = [
            (revenue_growth, 0.10),  # 10% revenue growth
            (earnings_growth, 0.10),  # 10% earnings growth
        ]
        growth_score = sum(metric is not None and metric > threshold for metric, threshold in thresholds_growth)

        signals.append("bullish" if growth_score >= 1 else "bearish" if growth_score == 0 and any(m is not None for m, _ in thresholds_growth) else "neutral")
        reasoning["growth_signal"] = {
            "signal": signals[-1],
            "details": (f"Revenue Growth: {revenue_growth:.2%}" if revenue_growth is not None else "Revenue Growth: N/A") + ", " + (f"Earnings Growth: {earnings_growth:.2%}" if earnings_growth is not None else "Earnings Growth: N/A"),
        }

        progress.update_status("fundamentals_agent", ticker, "Analyzing financial health")
        # 3. Financial Health
        current_ratio = metrics.current_ratio_quarterly if hasattr(metrics, "current_ratio_quarterly") else metrics.current_ratio
        debt_to_equity = metrics.debt_to_equity_ratio_quarterly if hasattr(metrics, "debt_to_equity_ratio_quarterly") else metrics.debt_to_equity

        health_score = 0
        health_details_list = []
        if current_ratio is not None:
            if current_ratio > 1.5:
                health_score += 1
            health_details_list.append(f"Current Ratio: {current_ratio:.2f}")
        else:
            health_details_list.append("Current Ratio: N/A")

        if debt_to_equity is not None:
            if debt_to_equity < 0.5:
                health_score += 1
            health_details_list.append(f"D/E: {debt_to_equity:.2f}")
        else:
            health_details_list.append("D/E: N/A")

        signals.append("bullish" if health_score >= 2 else "bearish" if health_score == 0 and (current_ratio is not None or debt_to_equity is not None) else "neutral")
        reasoning["financial_health_signal"] = {
            "signal": signals[-1],
            "details": ", ".join(health_details_list),
        }

        progress.update_status("fundamentals_agent", ticker, "Analyzing valuation ratios")
        # 4. Price to X ratios (Dynamically Calculated or from metrics if price unavailable)
        pe_ratio, pb_ratio, ps_ratio = None, None, None

        ttm_eps = metrics.earnings_per_share_basic if hasattr(metrics, "earnings_per_share_basic") else metrics.earnings_per_share
        # Ensure book_value_per_share and revenue_per_share_ttm attributes exist or use fallbacks
        bvps = metrics.book_value_per_share if hasattr(metrics, "book_value_per_share") else None
        sps_ttm = metrics.revenue_per_share_ttm if hasattr(metrics, "revenue_per_share_ttm") else None

        if current_price is not None:
            if ttm_eps is not None and ttm_eps > 0:
                pe_ratio = current_price / ttm_eps
            if bvps is not None and bvps > 0:
                pb_ratio = current_price / bvps
            if sps_ttm is not None and sps_ttm > 0:
                ps_ratio = current_price / sps_ttm
        else:  # Fallback to metrics' own ratios if current_price is None
            pe_ratio = metrics.price_to_earnings_ratio if hasattr(metrics, "price_to_earnings_ratio") else None
            pb_ratio = metrics.price_to_book_ratio if hasattr(metrics, "price_to_book_ratio") else None
            ps_ratio = metrics.price_to_sales_ratio if hasattr(metrics, "price_to_sales_ratio") else None

        price_ratios_details_list = [
            f"Current Price ({actual_price_date_str}): {current_price:.2f}" if current_price is not None else f"Current Price (target {end_date}): N/A",
            f"TTM EPS: {ttm_eps:.2f}" if ttm_eps is not None else "TTM EPS: N/A",
            f"BVPS: {bvps:.2f}" if bvps is not None else "BVPS: N/A",
            f"SPS TTM: {sps_ttm:.2f}" if sps_ttm is not None else "SPS TTM: N/A",
            f"P/E: {pe_ratio:.2f}" if pe_ratio is not None else "P/E: N/A",
            f"P/B: {pb_ratio:.2f}" if pb_ratio is not None else "P/B: N/A",
            f"P/S: {ps_ratio:.2f}" if ps_ratio is not None else "P/S: N/A",
        ]

        thresholds_valuation = [
            (pe_ratio, 25),
            (pb_ratio, 3),
            (ps_ratio, 4),
        ]
        price_ratio_score = sum(metric is not None and metric > threshold for metric, threshold in thresholds_valuation)

        all_ratios_effectively_none = all(r is None for r in [pe_ratio, pb_ratio, ps_ratio])

        if all_ratios_effectively_none:
            signals.append("neutral")
        elif price_ratio_score >= 2:
            signals.append("bearish")
        elif price_ratio_score == 0:
            signals.append("bullish")
        else:
            signals.append("neutral")

        reasoning["price_ratios_signal"] = {
            "signal": signals[-1],
            "details": ", ".join(price_ratios_details_list),
        }

        progress.update_status("fundamentals_agent", ticker, "Calculating final signal")
        # Determine overall signal
        bullish_signals = signals.count("bullish")
        bearish_signals = signals.count("bearish")

        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        # Calculate confidence level
        # Count signals that are not neutral OR if neutral, ensure it's not due to all_ratios_none for valuation
        # This logic might need refinement based on how "neutral due to missing data" should impact confidence
        num_meaningful_signals = 0
        for i, s in enumerate(signals):
            if s != "neutral":
                num_meaningful_signals += 1
            elif i == 3 and not all_ratios_effectively_none:  # If valuation signal is neutral but ratios were calculable
                num_meaningful_signals += 1
            elif i != 3 and s == "neutral":  # For other signals, if neutral, check if underlying data was present
                if i == 0 and any(m is not None for m, _ in thresholds_profitability):
                    num_meaningful_signals += 1
                elif i == 1 and any(m is not None for m, _ in thresholds_growth):
                    num_meaningful_signals += 1
                elif i == 2 and (current_ratio is not None or debt_to_equity is not None):
                    num_meaningful_signals += 1

        if num_meaningful_signals == 0:
            confidence = 0.0
        else:
            confidence = round(max(bullish_signals, bearish_signals) / num_meaningful_signals if num_meaningful_signals > 0 else 0, 2) * 100

        if overall_signal == "neutral":
            if num_meaningful_signals > 0:
                confidence = round((num_meaningful_signals - abs(bullish_signals - bearish_signals)) / num_meaningful_signals, 2) * 100
            else:  # All signals were neutral likely due to missing data across the board
                confidence = 0.0

        fundamental_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }
        progress.update_status("fundamentals_agent", ticker, f"Done for {end_date}")

    message = HumanMessage(content=json.dumps(fundamental_analysis), name="fundamentals_agent")

    if state.get("metadata", {}).get("show_reasoning", False):
        show_agent_reasoning(fundamental_analysis, "Fundamental Analysis Agent")

    if "analyst_signals" not in state["data"]:
        state["data"]["analyst_signals"] = {}
    state["data"]["analyst_signals"]["fundamentals_agent"] = fundamental_analysis

    progress.update_status("fundamentals_agent", None, f"All tickers processed for {end_date}")

    return {"messages": [message], "data": state["data"]}
