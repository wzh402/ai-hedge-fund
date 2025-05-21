import json
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.graph.state import AgentState, show_agent_reasoning
from pydantic import BaseModel, Field
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm

# Define the prediction categories
WeeklyPredictionCategory = Literal[
    "Up >5%",
    "Up 3%-5%",
    "Within +/-3%",
    "Down 3%-5%",
    "Down >5%",
]


class WeeklyPrediction(BaseModel):
    predicted_category: WeeklyPredictionCategory = Field(description="The predicted price change category for the next week")
    confidence: float = Field(description="Confidence in the prediction, between 0.0 and 100.0")
    reasoning: str = Field(description="Reasoning for the prediction")


class WeeklyPredictionsOutput(BaseModel):
    predictions: dict[str, WeeklyPrediction] = Field(description="Dictionary of ticker to weekly prediction")


##### Portfolio Management Agent (now Prediction Aggregator Agent) #####
def portfolio_management_agent(state: AgentState):
    """Aggregates analyst signals and generates weekly price change category predictions for multiple tickers."""

    # Get the analyst signals and tickers
    analyst_signals = state["data"]["analyst_signals"]
    tickers = state["data"]["tickers"]
    # Portfolio data might be used for context if needed by signals, but not directly by this agent's LLM call for prediction
    # portfolio = state["data"]["portfolio"]

    current_prices = {}
    signals_by_ticker = {}
    for ticker in tickers:
        progress.update_status("prediction_aggregator", ticker, "Processing analyst signals for weekly prediction")

        # Get current prices for the ticker (e.g., from risk_management_agent or another source)
        # For now, we assume risk_management_agent still provides current_price,
        # which would be the price at the end of the "past week" analysis period.
        risk_data = analyst_signals.get("risk_management_agent", {}).get(ticker, {})
        current_prices[ticker] = risk_data.get("current_price", 0)
        # max_shares and position_limits are not directly relevant for the prediction task's LLM call

        # Get signals for the ticker
        ticker_signals = {}
        for agent, signals in analyst_signals.items():
            # Exclude risk_management_agent's own detailed data if it's not a "signal"
            # Or, decide if some parts of risk_data should be part of the "signal" input
            if agent != "risk_management_agent" and ticker in signals:  # Ensure signals for the ticker exist
                # Make sure the signal structure is consistent, e.g. {"signal": "some analysis", "confidence": 75.0}
                if isinstance(signals[ticker], dict) and "signal" in signals[ticker] and "confidence" in signals[ticker]:
                    ticker_signals[agent] = {"signal": signals[ticker]["signal"], "confidence": signals[ticker]["confidence"]}
                else:
                    # Fallback or logging if structure is unexpected
                    ticker_signals[agent] = {"signal": str(signals[ticker]), "confidence": 0.0}

        signals_by_ticker[ticker] = ticker_signals

    progress.update_status("prediction_aggregator", None, "Generating weekly predictions")

    # Generate the weekly predictions
    result = generate_weekly_predictions(
        tickers=tickers,
        signals_by_ticker=signals_by_ticker,
        current_prices=current_prices,  # End of past week prices
        model_name=state["metadata"]["model_name"],
        model_provider=state["metadata"]["model_provider"],
        # Removed portfolio details like cash, positions, margin as they are not in the revised prompt for prediction
    )

    # Create the prediction message
    message_content = {ticker: prediction.model_dump() for ticker, prediction in result.predictions.items()}
    message = HumanMessage(
        content=json.dumps(message_content),
        name="prediction_aggregator",  # Renamed from portfolio_manager
    )

    # Print the prediction if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(message_content, "Prediction Aggregator")

    progress.update_status("prediction_aggregator", None, "Done")

    # Store predictions in state.data for the backtester to use
    # The exact key for storing predictions can be decided based on how backtester will retrieve it.
    # For example: state["data"]["weekly_predictions"] = message_content
    if "predictions" not in state["data"]:
        state["data"]["predictions"] = {}
    state["data"]["predictions"]["prediction_aggregator"] = message_content

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],  # Ensure updated data with predictions is returned
    }


def generate_weekly_predictions(
    tickers: list[str],
    signals_by_ticker: dict[str, dict],
    current_prices: dict[str, float],  # Prices at the end of the "past week" for context
    model_name: str,
    model_provider: str,
) -> WeeklyPredictionsOutput:
    """Generates weekly price change category predictions using an LLM."""
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an experienced Senior Quantitative Strategist responsible for predicting the percentage price change category for multiple stock tickers for the upcoming week (next 5 trading days).
                Your predictions must be based on an in-depth analysis of the past week's data conducted by a team of specialized analyst agents.

                Core Tasks:
                1.  **Comprehensive Assessment**: Carefully review all signals provided by different analyst agents for each stock ticker. These signals may cover multiple dimensions such as technical analysis, fundamental summaries, market sentiment, news impact, etc.
                2.  **Quantitative Thinking**:
                    *   **Signal Weighting**: Evaluate the relative reliability and potential impact of different signal sources. Consider the confidence level associated with each signal.
                    *   **Multi-factor Integration**: Identify consistency, contradiction, or complementarity among various signals. For example, is a buy signal from technical indicators supported by positive news?
                    *   **Pattern Recognition**: From the comprehensive signals of the past week, try to identify recurring patterns or key drivers that might indicate the price trend for the upcoming week.
                    *   **Probability and Risk**: Your goal is to select the most likely price change category and provide a corresponding confidence level. Understand the inherent uncertainty of financial markets.
                3.  **Prudent Reasoning**: In the `reasoning` field, clearly, concisely, and logically explain the basis for your prediction. Explicitly state which key signals or signal combinations dominated your judgment and how they interacted. Explain why you chose a specific price change category over others.
                4.  **Dynamic Perspective**: Recognize that the market is dynamic, and your predictions are based on information available up to the "end of the past week."

                Available Prediction Categories:
                - "Up >5%"
                - "Up 3%-5%"
                - "Within +/-3%"
                - "Down 3%-5%"
                - "Down >5%"

                Input Information:
                - `signals_by_ticker`: A dictionary where keys are stock tickers and values are a collection of analysis results for that stock from different specialized analyst agents for the past week. Each signal typically includes analysis content and a confidence level.
                - `current_prices`: A dictionary where keys are stock tickers and values are the current price of that stock at the end of the past week's analysis period.

                Please strictly follow the specified JSON format for your output.
                """,
            ),
            (
                "human",
                """Based on the team's analysis of the past week, please provide the price change category prediction for the upcoming week for the following stock tickers.

                Signals for each stock ticker (based on past week's analysis):
                {signals_by_ticker}

                Current prices (at the end of the past week):
                {current_prices}

                Please strictly follow the JSON structure below for your output (ensure all fields are present, especially `predicted_category`, `confidence`, and `reasoning`):
                {{
                  "predictions": {{
                    "TICKER1": {{
                      "predicted_category": "Up >5% | Up 3%-5% | Within +/-3% | Down 3%-5% | Down >5%",
                      "confidence": float, // A float between 0.0 and 100.0
                      "reasoning": "string // Detailed explanation of the reason for predicting this price change category, based on the provided signals and current price analysis"
                    }},
                    "TICKER2": {{
                      // ... same structure as TICKER1 ...
                    }},
                    // ... other stock tickers ...
                  }}
                }}
                """,
            ),
        ]
    )

    prompt = template.invoke(
        {
            "signals_by_ticker": json.dumps(signals_by_ticker, indent=2),
            "current_prices": json.dumps(current_prices, indent=2),
            # Removed portfolio_cash, portfolio_positions, margin_requirement, total_margin_used, max_shares
        }
    )

    def create_default_prediction_output():
        return WeeklyPredictionsOutput(predictions={ticker: WeeklyPrediction(predicted_category="Within +/-3%", confidence=0.0, reasoning="Error in prediction generation, defaulting to neutral.") for ticker in tickers})

    return call_llm(prompt=prompt, model_name=model_name, model_provider=model_provider, pydantic_model=WeeklyPredictionsOutput, agent_name="prediction_aggregator", default_factory=create_default_prediction_output)  # Renamed from portfolio_manager
