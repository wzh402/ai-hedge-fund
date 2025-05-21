from __future__ import annotations

"""Valuation Agent

Implements four complementary valuation methodologies and aggregates them with
configurable weights. 
"""
import json
from datetime import datetime, timedelta
from statistics import median
from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress

from src.tools.api import (
    get_financial_metrics,
    get_market_cap,  # Keep for fallback
    search_line_items,
    get_prices,
)


def valuation_agent(state: AgentState):
    """Run valuation across tickers and write signals back to `state`."""

    data = state["data"]
    end_date = data["end_date"]  # This end_date changes weekly
    tickers = data["tickers"]
    metadata = state.get("metadata", {})
    data_source = metadata.get("data_source", "financialdatasets")

    valuation_analysis: dict[str, dict] = {}

    for ticker in tickers:
        progress.update_status("valuation_agent", ticker, f"Fetching financial data for {end_date}")

        # Ensure financial_metrics are fetched relative to the specific end_date for the current analysis week.
        # The cache key for get_financial_metrics in api.py must correctly use this end_date.
        financial_metrics_list = get_financial_metrics(
            ticker=ticker,
            end_date=end_date,  # This is the crucial end_date for the analysis week
            period="ttm",
            limit=8,  # For median calculations if needed by underlying models
            data_source=data_source,
        )
        if not financial_metrics_list:
            progress.update_status("valuation_agent", ticker, f"Failed: No financial metrics for {ticker} as of {end_date}")
            valuation_analysis[ticker] = {"signal": "neutral", "confidence": 0, "reasoning": {"error": f"No financial metrics for {ticker} as of {end_date}"}}
            continue
        # This should be the TTM metrics as of the specific end_date of the analysis week
        most_recent_metrics = financial_metrics_list[0]

        # --- Current Market Price ---
        # Fetch most recent price for the specific end_date to calculate current market cap dynamically.
        progress.update_status("valuation_agent", ticker, f"Fetching current price for {ticker} at {end_date}")

        # Fetch a small window of prices leading up to end_date (analysis_end_str)
        # This logic is similar to what was implemented in fundamentals_agent
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
        price_fetch_start_date_str = (end_date_dt - timedelta(days=5)).strftime("%Y-%m-%d")

        # get_prices' cache key includes start_date and end_date.
        current_price_history = get_prices(ticker=ticker, start_date=price_fetch_start_date_str, end_date=end_date, data_source=data_source)  # Fetch a small window  # Up to analysis_end_str

        current_price = None
        actual_price_date_str = "N/A"  # To store the actual date of the price used

        if current_price_history:
            # Prices should be sorted chronologically by api.py, last one is latest.
            latest_price_point = current_price_history[-1]
            current_price = latest_price_point.close
            if current_price is None:  # Fallback
                current_price = (latest_price_point.high + latest_price_point.low) / 2 if latest_price_point.high and latest_price_point.low else latest_price_point.open
            actual_price_date_str = latest_price_point.time  # Assuming 'time' attribute holds the date string
            progress.update_status("valuation_agent", ticker, f"Using price from {actual_price_date_str} (target end_date: {end_date}) for market cap")

        if current_price is None:
            progress.update_status("valuation_agent", ticker, f"Warning: Could not fetch current price for {ticker} up to {end_date}. Market cap calculation might be affected or use fallback.")
        else:
            progress.update_status("valuation_agent", ticker, f"Current price for {ticker} at {actual_price_date_str} (target {end_date}): {current_price:.2f}")

        # --- Dynamic Market Cap ---
        dynamic_market_cap = None
        final_shares_outstanding = None

        # Attempt 1: Shares outstanding from most_recent_metrics (TTM)
        # Ensure shares_outstanding is a positive number
        if hasattr(most_recent_metrics, "shares_outstanding") and isinstance(most_recent_metrics.shares_outstanding, (int, float)) and most_recent_metrics.shares_outstanding > 0:
            final_shares_outstanding = most_recent_metrics.shares_outstanding
            progress.update_status("valuation_agent", ticker, f"Using shares_outstanding from TTM: {final_shares_outstanding} (Metrics Date: {getattr(most_recent_metrics, 'date', 'N/A')}) for end_date {end_date}")
        else:
            progress.update_status("valuation_agent", ticker, f"Shares_outstanding from TTM is invalid ({getattr(most_recent_metrics, 'shares_outstanding', 'N/A')}, Metrics Date: {getattr(most_recent_metrics, 'date', 'N/A')}). Trying latest QUARTERLY for end_date {end_date}.")
            # Attempt 2: Shares outstanding from latest QUARTERLY metrics
            quarterly_metrics_list = get_financial_metrics(
                ticker=ticker,
                end_date=end_date,  # Use the same end_date to get the latest relevant quarterly report
                period="quarterly",
                limit=1,  # Fetch only the most recent quarterly report
                data_source=data_source,
            )
            if quarterly_metrics_list and hasattr(quarterly_metrics_list[0], "shares_outstanding") and isinstance(quarterly_metrics_list[0].shares_outstanding, (int, float)) and quarterly_metrics_list[0].shares_outstanding > 0:
                final_shares_outstanding = quarterly_metrics_list[0].shares_outstanding
                progress.update_status("valuation_agent", ticker, f"Using shares_outstanding from latest QUARTERLY: {final_shares_outstanding} (Quarterly Metrics Date: {getattr(quarterly_metrics_list[0], 'date', 'N/A')}) for target end_date: {end_date}")
            else:
                q_data_log = "No quarterly data found"
                if quarterly_metrics_list and hasattr(quarterly_metrics_list[0], "date"):
                    q_data_log = f"Quarterly data found (Date: {quarterly_metrics_list[0].date}), but shares_outstanding is invalid: {getattr(quarterly_metrics_list[0], 'shares_outstanding', 'N/A')}"
                elif quarterly_metrics_list:
                    q_data_log = f"Quarterly data found, but shares_outstanding attribute missing or invalid. Data: {quarterly_metrics_list[0]}"

                progress.update_status("valuation_agent", ticker, f"Shares_outstanding not found or invalid in latest QUARTERLY metrics for end_date {end_date}. {q_data_log}")

        # Log current_price and final_shares_outstanding before attempting calculation
        progress.update_status("valuation_agent", ticker, f"Pre-Dynamic Cap Calculation: current_price={current_price}, final_shares_outstanding={final_shares_outstanding}, most_recent_metrics.market_cap (TTM, Date: {getattr(most_recent_metrics, 'date', 'N/A')})={getattr(most_recent_metrics, 'market_cap', 'N/A')} for end_date {end_date}")

        if current_price is not None and final_shares_outstanding is not None:  # current_price can be 0, shares must be positive (checked above)
            dynamic_market_cap = current_price * final_shares_outstanding
            progress.update_status("valuation_agent", ticker, f"Calculated dynamic market cap: {dynamic_market_cap:,.2f} (Price: {current_price:.2f}, Shares: {final_shares_outstanding}) for end_date {end_date}")
        else:
            progress.update_status("valuation_agent", ticker, f"Dynamic market cap calculation skipped. Reason: current_price is None ({current_price is None}) or final_shares_outstanding is None ({final_shares_outstanding is None}). Using fallbacks for end_date {end_date}.")
            # Fallback 1: market_cap from most_recent_metrics (TTM)
            if hasattr(most_recent_metrics, "market_cap") and most_recent_metrics.market_cap is not None:
                dynamic_market_cap = most_recent_metrics.market_cap
                progress.update_status("valuation_agent", ticker, f"Using TTM market_cap from financial metrics: {dynamic_market_cap:,.2f} (Metrics Date: {getattr(most_recent_metrics, 'date', 'N/A')}, Shares outstanding: {final_shares_outstanding}, Current Price: {current_price}) for end_date {end_date}")
            else:
                # Fallback 2: get_market_cap API call
                progress.update_status("valuation_agent", ticker, f"TTM market_cap (from metrics dated {getattr(most_recent_metrics, 'date', 'N/A')}) also missing or None. Falling back to get_market_cap API for end_date {end_date}")
                dynamic_market_cap_api_val = get_market_cap(ticker=ticker, end_date=end_date, data_source=data_source)
                if dynamic_market_cap_api_val is not None:
                    dynamic_market_cap = dynamic_market_cap_api_val
                    progress.update_status("valuation_agent", ticker, f"Market cap from get_market_cap API (for end_date {end_date}): {dynamic_market_cap:,.2f}")
                else:
                    # This is a critical failure point if all sources fail.
                    progress.update_status("valuation_agent", ticker, f"Market cap from get_market_cap API (for end_date {end_date}) is also None. Market cap will be None.")

        # Ensure dynamic_market_cap is not None before proceeding (it could be if all fallbacks fail)
        if dynamic_market_cap is None:
            progress.update_status("valuation_agent", ticker, f"Failed: Market cap unavailable for {ticker} for {end_date} after all attempts.")
            valuation_analysis[ticker] = {"signal": "neutral", "confidence": 0, "reasoning": {"error": f"Market cap unavailable for {ticker} as of {end_date}"}}
            continue  # Skip to next ticker

        progress.update_status("valuation_agent", ticker, f"Using Market Cap: {dynamic_market_cap:,.2f} for valuations at {end_date}")

        # --- Fine‑grained line‑items ---
        # These are typically TTM and should be fetched relative to end_date.
        # The cache key for search_line_items in api.py must correctly use this end_date.
        progress.update_status("valuation_agent", ticker, f"Gathering line items for {end_date}")
        line_items_data = search_line_items(
            ticker=ticker,
            query=["free_cash_flow", "net_income", "depreciation_and_amortization", "capital_expenditure", "working_capital"],
            end_date=end_date,  # Ensure this is used for fetching/caching
            period="ttm",
            limit=2,  # For calculating working_capital_change
            data_source=data_source,
        )
        if len(line_items_data) < 2:
            progress.update_status("valuation_agent", ticker, f"Failed: Insufficient financial line items for {end_date}")
            # Add a neutral signal with error to avoid crashing the flow
            valuation_analysis[ticker] = {"signal": "neutral", "confidence": 0, "reasoning": {"error": f"Insufficient line items for {ticker} as of {end_date}"}}
            continue
        li_curr, li_prev = line_items_data[0], line_items_data[1]

        wc_change = li_curr.get("working_capital", 0) - li_prev.get("working_capital", 0)

        # Ensure earnings_growth is available, provide a default if not.
        # These growth rates are typically from TTM metrics.
        earnings_growth_rate = (
            most_recent_metrics.earnings_growth_annual if hasattr(most_recent_metrics, "earnings_growth_annual") and most_recent_metrics.earnings_growth_annual is not None else most_recent_metrics.earnings_growth_quarterly if hasattr(most_recent_metrics, "earnings_growth_quarterly") and most_recent_metrics.earnings_growth_quarterly is not None else getattr(most_recent_metrics, "earnings_growth", 0.05)
        )  # Default to 5% if not found

        book_value_growth_rate = most_recent_metrics.book_value_per_share_growth_annual if hasattr(most_recent_metrics, "book_value_per_share_growth_annual") and most_recent_metrics.book_value_per_share_growth_annual is not None else getattr(most_recent_metrics, "book_value_growth", 0.03)  # Default to 3%

        owner_val = calculate_owner_earnings_value(
            net_income=li_curr.get("net_income"),
            depreciation=li_curr.get("depreciation_and_amortization"),
            capex=li_curr.get("capital_expenditure"),
            working_capital_change=wc_change,
            growth_rate=earnings_growth_rate,
        )

        dcf_val = calculate_intrinsic_value(
            free_cash_flow=li_curr.get("free_cash_flow"),
            growth_rate=earnings_growth_rate,
        )

        # calculate_ev_ebitda_value uses financial_metrics_list which is already fetched with end_date
        ev_ebitda_val = calculate_ev_ebitda_value(financial_metrics_list)

        rim_val = calculate_residual_income_value(
            market_cap=dynamic_market_cap,  # Crucially, use the dynamic market cap
            net_income=li_curr.get("net_income"),
            price_to_book_ratio=most_recent_metrics.price_to_book_ratio,
            book_value_growth=book_value_growth_rate,
        )
        # ------------------------------------------------------------------
        # Aggregate & signal
        # ------------------------------------------------------------------
        # market_cap = get_market_cap(ticker, end_date, data_source=data_source) # Commented out, use dynamic_market_cap
        # if not market_cap:
        #     progress.update_status("valuation_agent", ticker, "Failed: Market cap unavailable")
        #     continue

        method_values = {
            "dcf": {"value": dcf_val, "weight": 0.35},
            "owner_earnings": {"value": owner_val, "weight": 0.35},
            "ev_ebitda": {"value": ev_ebitda_val, "weight": 0.20},
            "residual_income": {"value": rim_val, "weight": 0.10},
        }

        # Filter out methods where value is None or 0 before calculating total_weight and weighted_gap
        valid_method_values = {k: v for k, v in method_values.items() if v["value"] is not None and v["value"] > 0}

        if not valid_method_values:
            progress.update_status("valuation_agent", ticker, f"Failed: All valuation methods yielded no positive value for {end_date}")
            valuation_analysis[ticker] = {"signal": "neutral", "confidence": 0, "reasoning": {"error": f"All valuation methods zero or invalid for {ticker} as of {end_date}"}}
            continue

        total_weight = sum(v["weight"] for v in valid_method_values.values())

        if total_weight == 0:  # Should be caught by previous check, but as a safeguard
            progress.update_status("valuation_agent", ticker, f"Failed: Total weight is zero after filtering methods for {end_date}")
            valuation_analysis[ticker] = {"signal": "neutral", "confidence": 0, "reasoning": {"error": f"Total valuation weight zero for {ticker} as of {end_date}"}}
            continue

        for m_name, v_dict in valid_method_values.items():
            # Ensure dynamic_market_cap is not zero for gap calculation to avoid DivisionByZero
            if dynamic_market_cap != 0:
                v_dict["gap"] = (v_dict["value"] - dynamic_market_cap) / dynamic_market_cap
            else:
                v_dict["gap"] = None  # Or some other indicator of an issue

        # Calculate weighted_gap only from methods that have a valid gap
        valid_gaps = [v["gap"] for v in valid_method_values.values() if v.get("gap") is not None]
        valid_weights_for_gap = [v["weight"] for v in valid_method_values.values() if v.get("gap") is not None]

        if not valid_gaps or sum(valid_weights_for_gap) == 0:
            weighted_gap = 0  # Or handle as neutral/error
        else:
            weighted_gap = sum(v["weight"] * v["gap"] for v in valid_method_values.values() if v.get("gap") is not None) / sum(valid_weights_for_gap)

        signal = "bullish" if weighted_gap > 0.15 else "bearish" if weighted_gap < -0.15 else "neutral"
        confidence = round(min(abs(weighted_gap) / 0.30 * 100, 100)) if weighted_gap is not None else 0

        reasoning = {}
        for m, vals in method_values.items():  # Iterate original to show all attempts
            current_val = vals["value"]
            gap_val = valid_method_values.get(m, {}).get("gap") if m in valid_method_values else None  # Get calculated gap if method was valid

            details_str = f"Value: ${current_val:,.2f}" if current_val is not None else "Value: N/A"
            details_str += f", Market Cap: ${dynamic_market_cap:,.2f}"
            if gap_val is not None:
                details_str += f", Gap: {gap_val:.1%}"
            else:
                details_str += ", Gap: N/A"
            details_str += f", Weight: {vals['weight']*100:.0f}%"

            method_signal = "neutral"
            if gap_val is not None:
                method_signal = "bullish" if gap_val > 0.15 else "bearish" if gap_val < -0.15 else "neutral"

            reasoning[f"{m}_analysis"] = {
                "signal": method_signal,
                "details": details_str,
            }

        valuation_analysis[ticker] = {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }
        progress.update_status("valuation_agent", ticker, f"Done for {end_date}")

    # ---- Emit message (for LLM tool chain) ----
    msg = HumanMessage(content=json.dumps(valuation_analysis), name="valuation_agent")
    if state.get("metadata", {}).get("show_reasoning", False):  # Safer get
        show_agent_reasoning(valuation_analysis, "Valuation Analysis Agent")

    if "analyst_signals" not in state["data"]:
        state["data"]["analyst_signals"] = {}
    state["data"]["analyst_signals"]["valuation_agent"] = valuation_analysis

    progress.update_status("valuation_agent", None, f"All tickers processed for {end_date}")
    return {"messages": [msg], "data": state["data"]}


#############################
# Helper Valuation Functions
#############################


def calculate_owner_earnings_value(
    net_income: float | None,
    depreciation: float | None,
    capex: float | None,
    working_capital_change: float | None,
    growth_rate: float = 0.05,  # Default if not provided by metrics
    required_return: float = 0.15,
    margin_of_safety: float = 0.25,
    num_years: int = 5,
) -> float | None:  # Return None if calculation is not possible
    if not all(isinstance(x, (int, float)) for x in [net_income, depreciation, capex, working_capital_change]):
        return None  # Changed from 0 to None

    # Ensure required values are not None
    if net_income is None or depreciation is None or capex is None or working_capital_change is None:
        return None

    owner_earnings = net_income + depreciation - capex - working_capital_change
    if owner_earnings <= 0:  # Can be 0 if inputs lead to it, but if strictly positive is needed, then None
        return 0  # Or None if negative owner earnings are invalid for this model

    pv = 0.0
    for yr in range(1, num_years + 1):
        future = owner_earnings * (1 + growth_rate) ** yr
        pv += future / (1 + required_return) ** yr

    terminal_growth = min(growth_rate, 0.03)
    # Avoid division by zero if required_return is too close to terminal_growth
    if required_return <= terminal_growth:
        return pv  # Or handle as an error / None, returning sum of first N years

    term_val = (owner_earnings * (1 + growth_rate) ** num_years * (1 + terminal_growth)) / (required_return - terminal_growth)
    pv_term = term_val / (1 + required_return) ** num_years

    intrinsic = pv + pv_term
    return intrinsic * (1 - margin_of_safety)


def calculate_intrinsic_value(
    free_cash_flow: float | None,
    growth_rate: float = 0.05,  # Default
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.02,
    num_years: int = 5,
) -> float | None:  # Return None if calculation is not possible
    if free_cash_flow is None or free_cash_flow <= 0:
        return None  # Changed from 0 to None

    pv = 0.0
    for yr in range(1, num_years + 1):
        fcft = free_cash_flow * (1 + growth_rate) ** yr
        pv += fcft / (1 + discount_rate) ** yr

    # Avoid division by zero
    if discount_rate <= terminal_growth_rate:
        return pv  # Or handle as an error / None

    term_val = (free_cash_flow * (1 + growth_rate) ** num_years * (1 + terminal_growth_rate)) / (discount_rate - terminal_growth_rate)
    pv_term = term_val / (1 + discount_rate) ** num_years

    return pv + pv_term


def calculate_ev_ebitda_value(financial_metrics: list) -> float | None:  # Return None if not calculable
    if not financial_metrics:
        return None
    m0 = financial_metrics[0]

    # Ensure necessary attributes exist and are not None
    ev = getattr(m0, "enterprise_value", None)
    ev_to_ebitda_ratio = getattr(m0, "enterprise_value_to_ebitda_ratio", None)
    mc = getattr(m0, "market_cap", None)

    if not (ev and ev_to_ebitda_ratio):
        return None
    if ev_to_ebitda_ratio == 0:  # Avoid division by zero
        return None

    ebitda_now = ev / ev_to_ebitda_ratio

    # Filter for valid ratios before taking median
    valid_ratios = [getattr(m, "enterprise_value_to_ebitda_ratio", None) for m in financial_metrics]
    valid_ratios = [r for r in valid_ratios if isinstance(r, (int, float)) and r != 0]  # Ensure they are numbers and not zero

    if not valid_ratios:
        return None  # Cannot calculate median multiple

    med_mult = median(valid_ratios)
    ev_implied = med_mult * ebitda_now

    net_debt = (ev or 0) - (mc or 0)  # mc could be None
    equity_value = ev_implied - net_debt
    return max(equity_value, 0)  # Equity value cannot be negative


def calculate_residual_income_value(
    market_cap: float | None,
    net_income: float | None,
    price_to_book_ratio: float | None,
    book_value_growth: float = 0.03,  # Default
    cost_of_equity: float = 0.10,
    terminal_growth_rate: float = 0.03,  # Ensure this is not >= cost_of_equity
    num_years: int = 5,
) -> float | None:  # Return None if not calculable
    if not (market_cap and net_income and price_to_book_ratio and price_to_book_ratio > 0):
        return None

    # Ensure required values are not None
    if market_cap is None or net_income is None or price_to_book_ratio is None:
        return None

    book_val = market_cap / price_to_book_ratio
    ri0 = net_income - cost_of_equity * book_val
    if ri0 <= 0:  # If initial residual income is not positive, model might not be suitable or indicates issues
        return book_val  # Or None, or just book_val as a floor

    pv_ri = 0.0
    for yr in range(1, num_years + 1):
        ri_t = ri0 * (1 + book_value_growth) ** yr
        pv_ri += ri_t / (1 + cost_of_equity) ** yr

    # Avoid division by zero for terminal value
    if cost_of_equity <= terminal_growth_rate:
        # If no terminal growth or problematic rates, intrinsic value is book value + PV of RIs for N years
        return book_val + pv_ri

    term_ri = ri0 * (1 + book_value_growth) ** (num_years + 1) / (cost_of_equity - terminal_growth_rate)
    pv_term = term_ri / (1 + cost_of_equity) ** num_years

    intrinsic = book_val + pv_ri + pv_term
    return intrinsic * 0.8
