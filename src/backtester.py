import sys
from dotenv import load_dotenv  # Add this import

# Load environment variables from .env file FIRST
load_dotenv()  # Add this call

import pandas as pd
from typing import Callable, Dict, List, Any, Tuple
import numpy as np
from datetime import timedelta

# Corrected import: get_prices is the correct function name from api.py
from src.tools.api import get_prices, get_financial_metrics, get_insider_trades, get_company_news
from src.utils.progress import progress
from src.utils.display import print_backtest_results  # This will need to be heavily modified or replaced
from src.agents.portfolio_manager import WeeklyPredictionCategory  # Import the category type


# Define the structure for accuracy statistics
class AccuracyStats:
    def __init__(self):
        self.correct_predictions = 0
        self.total_predictions = 0

    def update(self, correct: bool):
        self.total_predictions += 1
        if correct:
            self.correct_predictions += 1

    def calculate_accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return (self.correct_predictions / self.total_predictions) * 100


class Backtester:
    def __init__(
        self,
        # Adjusted agent signature to match run_hedge_fund more closely
        # agent: Callable[[str, str, list[str], dict, str, str, list[str], bool, Any], dict],
        agent: Callable[..., dict],  # Use a more generic callable or a specific one matching run_hedge_fund
        tickers: list[str],
        start_date: str,
        end_date: str,
        # initial_capital: float = 1_000_000, # No longer directly used for P&L
        model_name: str = "gpt-4o",
        model_provider: str = "openai",
        selected_analysts: list[str] | None = None,
        show_reasoning: bool = False,
        data_source: str = "financialdatasets",  # Added data_source
    ):
        self.agent = agent
        self.tickers = tickers
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        # self.initial_capital = initial_capital # Not primary focus
        # self.portfolio = self._initialize_portfolio() # Portfolio for trading is no longer the focus
        # self.portfolio_values = [] # Portfolio value tracking is no longer the focus
        # self.performance_metrics = {} # Performance metrics will be accuracy based
        self.model_name = model_name
        self.model_provider = model_provider
        self.selected_analysts = selected_analysts if selected_analysts is not None else []
        self.show_reasoning = show_reasoning
        self.data_source = data_source  # Store data_source

        self.accuracy_stats: Dict[str, AccuracyStats] = {ticker: AccuracyStats() for ticker in self.tickers}
        self.overall_accuracy_stats = AccuracyStats()

        self.weekly_results_log: List[Dict[str, Any]] = []

    # _initialize_portfolio, execute_trade, calculate_portfolio_value are removed as they are trade-specific

    def prefetch_data(self):
        """Prefetches data for all tickers for the entire backtest period plus a lookback."""
        # Determine the earliest date needed for lookback (e.g., 1 year before start_date for initial analysis)
        # For weekly predictions, the lookback for the first week's analysis is key.
        # The agent itself will handle lookback based on the dates passed to it.
        # This prefetch can grab data for the entire self.start_date to self.end_date range.
        # And potentially a bit before self.start_date if the first "past week" analysis needs it.

        # Let's assume a 1-month buffer before start_date for the first analysis window
        prefetch_start_date = (self.start_date - pd.Timedelta(days=35)).strftime("%Y-%m-%d")
        prefetch_end_date = self.end_date.strftime("%Y-%m-%d")

        progress.update_status("prefetch_data", None, f"Prefetching data from {prefetch_start_date} to {prefetch_end_date} using {self.data_source}")
        for ticker in self.tickers:
            progress.update_status("prefetch_data", ticker, f"Prefetching price data from {self.data_source}")
            get_prices(ticker, prefetch_start_date, prefetch_end_date, data_source=self.data_source)
            progress.update_status("prefetch_data", ticker, f"Prefetching financial metrics from {self.data_source}")
            # get_financial_metrics(ticker, end_date, period="ttm", limit=10)
            get_financial_metrics(ticker, prefetch_end_date, data_source=self.data_source)
            progress.update_status("prefetch_data", ticker, f"Prefetching insider trades from {self.data_source}")
            # get_insider_trades(ticker, end_date, start_date=None, limit=1000)
            get_insider_trades(ticker, prefetch_end_date, start_date=prefetch_start_date, data_source=self.data_source)
            progress.update_status("prefetch_data", ticker, f"Prefetching company news from {self.data_source}")
            # get_company_news(ticker, end_date, start_date=None, limit=1000)
            get_company_news(ticker, prefetch_end_date, start_date=prefetch_start_date, data_source=self.data_source)
        progress.update_status("prefetch_data", None, "Done")

    def classify_percentage_change(self, percentage_change: float) -> WeeklyPredictionCategory:
        if percentage_change > 5:
            return "Up >5%"
        elif 3 <= percentage_change <= 5:
            return "Up 3%-5%"
        elif -3 < percentage_change < 3:
            return "Within +/-3%"
        elif -5 <= percentage_change <= -3:  # Corrected logic for negative ranges
            return "Down 3%-5%"
        elif percentage_change < -5:
            return "Down >5%"
        # This case should ideally not be hit if logic is correct, but as a fallback:
        return "Within +/-3%"  # Fallback or raise error

    def _get_price_on_or_after(self, ticker: str, date: pd.Timestamp) -> float | None:
        """Gets the closing price on the given date, or the next available trading day's close."""
        for i in range(5):  # Try for up to 5 days
            current_date_str = (date + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            try:
                price_list = get_prices(ticker, current_date_str, current_date_str, data_source=self.data_source)
                if price_list and hasattr(price_list[0], "close"):
                    return price_list[0].close
            except Exception as e:
                # print(f"DEBUG: In _get_price_on_or_after, failed to get price for {ticker} on {current_date_str}: {e}")
                pass  # Try next day
        return None

    def _get_price_on_or_before(self, ticker: str, date: pd.Timestamp) -> float | None:
        """Gets the closing price on the given date, or the previous available trading day's close."""
        for i in range(5):  # Try for up to 5 days
            current_date_str = (date - pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            try:
                price_list = get_prices(ticker, current_date_str, current_date_str, data_source=self.data_source)
                if price_list:
                    if hasattr(price_list[-1], "close"):
                        return price_list[-1].close
            except Exception as e:
                # print(f"DEBUG: In _get_price_on_or_before, failed to get price for {ticker} on {current_date_str}: {e}")
                pass  # Try previous day
        return None

    def run_backtest(self) -> dict:
        self.prefetch_data()

        # Generate weekly start dates (e.g., Mondays) within the backtest period
        # Ensure that the week_start_date allows for a full "past week" of data before it
        # and a full "future week" of data after it within the overall start/end_date.

        # We need at least 1 week of data for analysis before the first prediction week starts.
        # And the prediction week itself needs to be within the overall date range.
        effective_start_date = self.start_date + pd.Timedelta(days=7)  # Ensure there's a "past week"

        # Iterate by weeks, taking Monday as the start of the "future week" to predict
        # The "past week" will be the 5 trading days leading up to this Monday.
        date_range = pd.date_range(start=effective_start_date, end=self.end_date, freq="B")  # Business days

        # Get unique Mondays (or first business day of the week)
        # This will be the start of the "prediction week"
        prediction_week_start_dates = sorted(list(set(d - pd.Timedelta(days=d.weekday()) for d in date_range if (d - pd.Timedelta(days=d.weekday())) >= effective_start_date)))

        if not prediction_week_start_dates:
            print("Not enough data to run weekly backtest for the given date range.")
            return {"overall_accuracy": 0, "ticker_accuracy": {}, "weekly_log": []}

        for pred_week_start_dt in prediction_week_start_dates:
            # Ensure the full prediction week (Mon-Fri) is within self.end_date
            pred_week_end_dt_nominal = pred_week_start_dt + pd.Timedelta(days=4)
            if pred_week_end_dt_nominal > self.end_date:
                # print(f"Skipping week starting {pred_week_start_dt.strftime('%Y-%m-%d')} as it extends beyond end_date.")
                continue

            # Define "past week" for analysis (e.g., Monday to Friday of the week before pred_week_start_dt)
            analysis_end_dt = pred_week_start_dt - pd.Timedelta(days=1)  # Typically a Sunday, data up to prev Friday
            analysis_start_dt = analysis_end_dt - pd.Timedelta(days=6)  # Previous Monday

            analysis_start_str = analysis_start_dt.strftime("%Y-%m-%d")
            analysis_end_str = analysis_end_dt.strftime("%Y-%m-%d")

            progress.update_status("run_backtest", None, f"Analyzing week: {analysis_start_str} to {analysis_end_str} for predicting week starting {pred_week_start_dt.strftime('%Y-%m-%d')}")

            # Call the agent to get predictions for the "future week" (pred_week_start_dt to pred_week_end_dt)
            # The agent (run_hedge_fund) needs a simplified portfolio or context if any.
            # For pure prediction, portfolio state might be minimal.
            # The agent's internal graph should now use the "prediction_aggregator"
            # The agent signature: agent(start_date, end_date, tickers, portfolio, model_name, model_provider, selected_analysts, show_reasoning, None)
            # For prediction, the portfolio context passed to the agent might be minimal or just current prices at analysis_end_dt
            # Let's pass an empty portfolio for now, as the prediction agent prompt doesn't use it.
            mock_portfolio_for_agent = {"cash": 0, "positions": {}}  # Or derive from actuals if needed by other agents in graph

            agent_output_state = self.agent(
                start_date=analysis_start_str,  # \"Past week\" start
                end_date=analysis_end_str,  # \"Past week\" end
                tickers=self.tickers,
                portfolio=mock_portfolio_for_agent,
                model_name=self.model_name,
                model_provider=self.model_provider,
                selected_analysts=self.selected_analysts,
                show_reasoning=self.show_reasoning,
                data_source=self.data_source,  # Pass data_source to agent
                # Removed session_id=None as it's not in run_hedge_fund's signature
            )

            # DEBUG: Print the raw output from the agent immediately
            print(f"DEBUG: Raw agent_output_state from run_hedge_fund for week starting {pred_week_start_dt.strftime('%Y-%m-%d')}: {agent_output_state}")

            # Extract predictions from the agent's output state
            # Path 1: As per current code's primary attempt
            raw_predictions_path1 = agent_output_state.get("data", {}).get("predictions", {}).get("prediction_aggregator", {})
            # Path 2: If stored directly under "decisions"
            raw_predictions_path2 = agent_output_state.get("decisions", {})
            # Path 3: If stored directly under data.prediction_aggregator (less nested)
            raw_predictions_path3 = agent_output_state.get("data", {}).get("prediction_aggregator", {})

            print(f"DEBUG: Attempting to extract predictions for week starting {pred_week_start_dt.strftime('%Y-%m-%d')}:")
            print(f"DEBUG: Path 1 (data.predictions.prediction_aggregator): {raw_predictions_path1}")
            print(f"DEBUG: Path 2 (decisions): {raw_predictions_path2}")
            print(f"DEBUG: Path 3 (data.prediction_aggregator): {raw_predictions_path3}")

            # Initialize raw_predictions with the primary path
            raw_predictions = raw_predictions_path1

            if not raw_predictions or not isinstance(raw_predictions, dict) or not any(raw_predictions.values()):
                warning_message = f"Warning: No valid predictions received from agent for week starting {pred_week_start_dt.strftime('%Y-%m-%d')} using path data.predictions.prediction_aggregator."
                # Try path 2 if path 1 failed
                if raw_predictions_path2 and isinstance(raw_predictions_path2, dict) and any(isinstance(val, dict) and "predicted_category" in val for val in raw_predictions_path2.values()):
                    print(f"INFO: Found predictions under 'decisions' key. Using this path.")
                    raw_predictions = raw_predictions_path2
                # Try path 3 if path 1 and path 2 failed
                elif raw_predictions_path3 and isinstance(raw_predictions_path3, dict) and any(isinstance(val, dict) and "predicted_category" in val for val in raw_predictions_path3.values()):
                    print(f"INFO: Found predictions under 'data.prediction_aggregator' key. Using this path.")
                    raw_predictions = raw_predictions_path3
                else:
                    print(warning_message)  # Print original warning if all paths fail
                    print(f"DEBUG: No valid predictions found in common alternative paths. Full agent_output_state was printed above.")
                    continue  # Skip this week if no predictions found

            # Get actual prices for the "future week" (prediction week)
            # Prediction week start: pred_week_start_dt
            # Prediction week end: pred_week_start_dt + 4 business days (nominal Friday)

            actual_pred_week_start_prices: Dict[str, float] = {}
            actual_pred_week_end_prices: Dict[str, float] = {}
            valid_week_for_all_tickers = True

            # Store the previous week's end prices for the new calculation method
            # Initialize with None or a way to fetch for the very first week if necessary
            # For simplicity, we'll start calculating from the second prediction week
            # or handle the first week as a special case (e.g. skip or use a different start price)
            if "prev_week_end_prices" not in locals() and "prev_week_end_prices" not in self.__dict__:
                prev_week_end_prices: Dict[str, float | None] = {ticker: None for ticker in self.tickers}

            current_week_end_prices_for_next_iteration: Dict[str, float | None] = {ticker: None for ticker in self.tickers}

            for ticker in self.tickers:
                # Price at the end of the prediction week (e.g., Friday's close)
                potential_end_of_week = min(pred_week_start_dt + pd.Timedelta(days=4), self.end_date)
                end_price = self._get_price_on_or_before(ticker, potential_end_of_week)
                current_week_end_prices_for_next_iteration[ticker] = end_price  # Store for next week's start

                # Use the previous week's Friday close as the start price
                start_price = prev_week_end_prices[ticker]

                if start_price is None or end_price is None:
                    print(f"Warning: Missing actual start (previous Friday close) or end price for {ticker} for prediction week starting {pred_week_start_dt.strftime('%Y-%m-%d')}. Skipping ticker for this week.")
                    # actual_pred_week_start_prices[ticker] = None # Mark as invalid
                    # actual_pred_week_end_prices[ticker] = end_price
                    continue

                actual_pred_week_start_prices[ticker] = start_price
                actual_pred_week_end_prices[ticker] = end_price

            # Update prev_week_end_prices for the next iteration
            prev_week_end_prices = current_week_end_prices_for_next_iteration.copy()

            # Calculate actual changes and compare with predictions
            for ticker in self.tickers:
                if ticker not in actual_pred_week_start_prices or ticker not in actual_pred_week_end_prices:
                    continue  # Already warned, skip this ticker for the week

                if ticker not in raw_predictions:
                    print(f"Warning: No prediction from agent for {ticker} for week starting {pred_week_start_dt.strftime('%Y-%m-%d')}")
                    continue

                start_price = actual_pred_week_start_prices[ticker]
                end_price = actual_pred_week_end_prices[ticker]

                predicted_category_data = raw_predictions[ticker]  # This should be a dict like {"predicted_category": "...", "confidence": ..., "reasoning": ...}
                predicted_category = predicted_category_data.get("predicted_category")

                if predicted_category is None:
                    print(f"Warning: Prediction for {ticker} is malformed: {predicted_category_data}")
                    continue

                actual_percentage_change = ((end_price - start_price) / start_price) * 100 if start_price != 0 else 0
                actual_category = self.classify_percentage_change(actual_percentage_change)

                is_correct = actual_category == predicted_category

                self.accuracy_stats[ticker].update(is_correct)
                self.overall_accuracy_stats.update(is_correct)

                self.weekly_results_log.append(
                    {
                        "Week Start (Prediction For)": pred_week_start_dt.strftime("%Y-%m-%d"),
                        "Analysis Period Start": analysis_start_str,
                        "Analysis Period End": analysis_end_str,
                        "Ticker": ticker,
                        "Predicted Category": predicted_category,
                        "Predicted Confidence": predicted_category_data.get("confidence", "N/A"),
                        "Actual Start Price": start_price,
                        "Actual End Price": end_price,
                        "Actual Change (%)": round(actual_percentage_change, 2),
                        "Actual Category": actual_category,
                        "Correct": is_correct,
                        "Reasoning": predicted_category_data.get("reasoning", "") if self.show_reasoning else "Hidden",
                    }
                )

        # Print results
        self._print_accuracy_report()

        return {
            "overall_accuracy": self.overall_accuracy_stats.calculate_accuracy(),
            "ticker_accuracy": {ticker: stats.calculate_accuracy() for ticker, stats in self.accuracy_stats.items()},
            "weekly_log": self.weekly_results_log,
        }

    def _print_accuracy_report(self):
        print("\n--- Weekly Prediction Accuracy Report ---")

        if self.weekly_results_log:
            df_results = pd.DataFrame(self.weekly_results_log)
            # Select columns to display, adjust as needed
            display_columns = ["Week Start (Prediction For)", "Ticker", "Predicted Category", "Actual Category", "Correct", "Actual Change (%)", "Predicted Confidence"]
            if self.show_reasoning:
                display_columns.append("Reasoning")

            # Filter out columns that might not exist if reasoning is off etc.
            df_results_display = df_results[[col for col in display_columns if col in df_results.columns]]

            print("\nDetailed Weekly Results:")
            # Temporarily set pandas display options for better console output
            with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
                print(df_results_display.to_string(index=False))
        else:
            print("No weekly results to display.")

        print("\n--- Accuracy Statistics ---")
        for ticker in self.tickers:
            stats = self.accuracy_stats[ticker]
            acc = stats.calculate_accuracy()
            print(f"  {ticker}: {stats.correct_predictions}/{stats.total_predictions} = {acc:.2f}%")

        overall_acc = self.overall_accuracy_stats.calculate_accuracy()
        print(f"  Overall: {self.overall_accuracy_stats.correct_predictions}/{self.overall_accuracy_stats.total_predictions} = {overall_acc:.2f}%")
        print("-------------------------------------\n")

    # _update_performance_metrics is removed as it was for trading P&L.


# Main execution block (if running backtester.py directly)
if __name__ == "__main__":
    import argparse
    from src.main import run_hedge_fund  # The agent function

    from src.utils.analysts import ANALYST_CONFIG  # Import ANALYST_CONFIG

    # Define desired default values
    DEFAULT_MODEL_NAME = "gemini-2.0-flash"
    DEFAULT_MODEL_PROVIDER = "Gemini"  # Corrected to match ModelProvider enum and api_models.json
    # Corrected keys from ANALYST_CONFIG
    DEFAULT_ANALYSTS_STRING = "technical_analyst,fundamentals_analyst,sentiment_analyst,valuation_analyst"
    # DEFAULT_ANALYSTS_STRING = "sentiment_analyst"
    parser = argparse.ArgumentParser(description="Run the AI Hedge Fund Backtester for Weekly Predictions.")
    parser.add_argument("--ticker", type=str, required=True, help="Comma-separated list of stock tickers (e.g., AAPL,MSFT).")
    parser.add_argument("--start-date", type=str, required=True, help="Start date for the backtest (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, required=True, help="End date for the backtest (YYYY-MM-DD).")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help=f"Name of the LLM model to use. Default: {DEFAULT_MODEL_NAME}.")
    parser.add_argument("--model-provider", type=str, default=DEFAULT_MODEL_PROVIDER, choices=["OpenAI", "Groq", "Ollama", "Anthropic", "DeepSeek", "Gemini"], help=f"Provider of the LLM model. Default: {DEFAULT_MODEL_PROVIDER}.")
    parser.add_argument(
        "--selected-analysts",
        type=str,
        default=DEFAULT_ANALYSTS_STRING,
        help=f"Comma-separated list of analyst agent keys. Default: {DEFAULT_ANALYSTS_STRING}. Available: {', '.join(ANALYST_CONFIG.keys())}.",
    )
    parser.add_argument("--show-reasoning", action="store_true", help="Show reasoning from the prediction agent.")
    parser.add_argument("--ollama", action="store_true", help="Use Ollama as the model provider with a default Ollama model (llama3) if model-name is not specified.")
    parser.add_argument("--data-source", type=str, default="financialdatasets", choices=["financialdatasets", "alphavantage"], help="Data source for financial data. Default: financialdatasets.")  # Added data_source argument

    args = parser.parse_args()

    tickers_list = [ticker.strip().upper() for ticker in args.ticker.split(",")]

    selected_analysts_list = None
    if args.selected_analysts:  # This will now use the default string if not provided by user
        selected_analysts_list = [analyst.strip() for analyst in args.selected_analysts.split(",")]

    if args.ollama:
        args.model_provider = "ollama"
        # If model_name is still the general default (gemini-2.0-flash, which is not for ollama)
        # and was not explicitly set by the user for ollama, switch to an ollama-specific default.
        if args.model_name == DEFAULT_MODEL_NAME:
            args.model_name = "llama3"  # Default Ollama model

    backtester = Backtester(
        agent=run_hedge_fund,  # Pass the main agent function
        tickers=tickers_list,
        start_date=args.start_date,
        end_date=args.end_date,
        model_name=args.model_name,
        model_provider=args.model_provider,
        selected_analysts=selected_analysts_list,
        show_reasoning=args.show_reasoning,
        data_source=args.data_source,  # Pass data_source to Backtester
    )

    results = backtester.run_backtest()
    # Results are already printed by _print_accuracy_report within run_backtest
    # print("\nFinal Accuracy Results:")
    # print(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
    # for ticker, acc in results['ticker_accuracy'].items():
    #     print(f"  {ticker} Accuracy: {acc:.2f}%")
