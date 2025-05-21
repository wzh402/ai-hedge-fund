import datetime
import os
import pandas as pd
import requests
import time  # Added for rate limiting

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
    CompanyFactsResponse,
)

# Global cache instance
_cache = get_cache()

# Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")
FINANCIAL_DATASETS_API_KEY = os.environ.get("FINANCIAL_DATASETS_API_KEY")

# --- Alpha Vantage Specific Functions ---


# Placeholder for search_line_items if it's specific to financialdatasets
# If an Alpha Vantage equivalent is needed, it would be a new function.
def search_line_items(
    ticker: str,
    query: list[str] | str,  # This will be used as 'line_items' for financialdatasets
    data_source: str = "financialdatasets",
    period: str = "annual",  # Original default was "ttm", current is "annual"
    limit: int = 5,  # Original default was 10, current is 5
    end_date: str | None = None,
) -> list[dict]:  # Returning list of dicts, can be parsed into LineItem model by caller
    """
    Search for line items.
    For financialdatasets.ai: uses POST to /financials/search/line-items.
    For Alpha Vantage: This function is NOT directly supported as AV provides full statements.
                       It will return an empty list if data_source is 'alphavantage'.
    """
    if data_source == "alphavantage":
        print(f"Warning: search_line_items is not directly supported for Alpha Vantage for {ticker}. Financial statements should be fetched and parsed instead.")
        return []

    elif data_source == "financialdatasets":
        # Try to get data from cache first
        # The cache key generation in cache.py uses ticker, query, period, limit, end_date, data_source
        if isinstance(query, (list, str)):  # Ensure query type is suitable for cache
            cached_data = _cache.get_line_items(ticker, query, period, limit, end_date, data_source=data_source)
            if cached_data is not None:
                # print(f"Returning cached line items for {ticker}, query: {query}, period: {period}, limit: {limit}, end_date: {end_date}")
                return cached_data

        if not FINANCIAL_DATASETS_API_KEY:
            print("Error: FINANCIAL_DATASETS_API_KEY not found. Financial Datasets API calls will fail.")
            return []

        headers = {"X-API-KEY": FINANCIAL_DATASETS_API_KEY}
        url = "https://api.financialdatasets.ai/financials/search/line-items"

        # Adapt the 'query' parameter to 'line_items_list' for the API body
        line_items_list: list[str]
        if isinstance(query, str):
            line_items_list = [query]
        elif isinstance(query, list):
            line_items_list = query
        else:
            # This case should ideally not be reached if type hints are followed,
            # but as a fallback from previous version that checked isinstance(query, (list, str))
            print(f"Error: 'query' parameter must be a string or list of strings for financialdatasets, got {type(query)}")
            return []

        if not end_date:
            # The original API call example included end_date.
            # If it's critical and None is not allowed by the API, we might need to raise an error or use a default.
            # For now, let's proceed, but the API might reject if end_date is None and required.
            # The agents seem to pass a valid end_date from config.
            print(f"Warning: end_date is None for search_line_items with financialdatasets for {ticker}. API might require it.")

        body = {
            "tickers": [ticker],
            "line_items": line_items_list,
            "end_date": end_date,  # Pass None if that's what was provided
            "period": period,
            "limit": limit,
        }

        print(f"Fetching line items from financialdatasets.ai (POST): {url} with body: {body}")

        try:
            response = requests.post(url, headers=headers, json=body)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

            data = response.json()
            # Assuming the response structure matches LineItemResponse from models.py
            # search_results = data.get("search_results", [])
            # The original code used Pydantic model: LineItemResponse(**data)
            try:
                response_model = LineItemResponse(**data)
                search_results_models = response_model.search_results
                # Convert Pydantic models to dicts for consistent return type
                search_results_dicts = [item.model_dump() for item in search_results_models]

            except Exception as pydantic_error:
                print(f"Error parsing response with LineItemResponse model: {pydantic_error}")
                print(f"Raw response data: {str(data)[:500]}")  # Log part of the raw data
                return []

            if not search_results_dicts:
                print(f"No search results found for {ticker} with query {line_items_list}")
                # Cache empty result to avoid refetching if it's a valid empty response
                _cache.set_line_items(ticker, query, period, limit, end_date, [], data_source=data_source)
                return []

            # Cache the results (list of dicts)
            _cache.set_line_items(ticker, query, period, limit, end_date, search_results_dicts[:limit], data_source=data_source)
            return search_results_dicts[:limit]

        except requests.exceptions.HTTPError as e:
            error_message = f"Error fetching data for line items from financialdatasets.ai (POST): {e.response.status_code} {e.response.reason} for URL {e.request.url}"
            if e.response is not None and e.response.text:
                error_message += f". Response text: {e.response.text[:200]}"
            print(error_message)
            return []  # Return empty list on HTTP error
        except requests.exceptions.RequestException as e:
            print(f"Request error for line items (POST) ({e.request.url if e.request else 'N/A'}): {e}")
            return []
        except ValueError as e:  # JSON decoding error
            print(f"Error decoding JSON response for line items (POST): {e}")
            return []
        except Exception as e:
            print(f"Unexpected error processing data for line items (POST): {type(e).__name__} - {e}")
            return []

    # If no valid data_source, return empty
    return []


def _make_alpha_vantage_request(params):
    """Helper function to make requests to Alpha Vantage, handling rate limits."""
    if not ALPHA_VANTAGE_API_KEY:
        raise ValueError("Alpha Vantage API key not found in environment variables.")

    params["apikey"] = ALPHA_VANTAGE_API_KEY
    url = "https://www.alphavantage.co/query"

    # Basic rate limiting: wait 12 seconds between calls (5 calls per minute)
    # A more sophisticated approach might use a global rate limiter instance.
    time.sleep(12.1)  # Sleep for a bit more than 12 seconds

    response = requests.get(url, params=params)
    response.raise_for_status()  # Raise an exception for HTTP errors
    data = response.json()
    if "Error Message" in data:
        raise Exception(f"Alpha Vantage API Error: {data['Error Message']}")
    if "Information" in data and "Rate Limit" in data["Information"]:
        # This means we hit the rate limit, though the sleep should prevent this.
        # For a robust solution, implement retries with exponential backoff here.
        raise Exception(f"Alpha Vantage API rate limit hit: {data['Information']}")
    return data


def get_prices_av(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch daily price data from Alpha Vantage."""
    # Alpha Vantage TIME_SERIES_DAILY_ADJUSTED returns up to 20+ years of data
    # We might need to adjust 'outputsize' based on the date range, but 'full' is often sufficient.
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "outputsize": "full",  # compact for last 100, full for 20+ years
    }
    data = _make_alpha_vantage_request(params)

    prices = []
    time_series = data.get("Time Series (Daily)")
    if not time_series:
        return []

    for date_str, daily_data in time_series.items():
        # Filter by date range
        if start_date <= date_str <= end_date:
            prices.append(
                Price(
                    time=date_str,
                    open=float(daily_data["1. open"]),
                    high=float(daily_data["2. high"]),
                    low=float(daily_data["3. low"]),
                    close=float(daily_data["4. close"]),
                    volume=int(daily_data["6. volume"]),
                    # Alpha Vantage provides adjusted close, which is good.
                    # Other fields like vwap, transactions might not be directly available or need calculation.
                    vwap=None,  # Or calculate if possible from other data
                    transactions=None,
                )
            )
    prices.sort(key=lambda p: p.time)  # Ensure chronological order
    return prices


def get_financial_metrics_av(ticker: str, end_date: str, period: str = "ttm", limit: int = 1) -> list[FinancialMetrics]:
    """Fetch financial metrics from Alpha Vantage (Company Overview).
    Note: This is a simplified version. AV provides detailed income, balance, cash flow statements separately.
    Mapping all those to FinancialMetrics is extensive. This uses OVERVIEW for some common metrics.
    'limit' and 'period' are not directly used as OVERVIEW provides current data.
    'end_date' is also not strictly used for filtering here as OVERVIEW is latest.
    """
    params = {"function": "OVERVIEW", "symbol": ticker}
    data = _make_alpha_vantage_request(params)

    if not data or data.get("Symbol") != ticker:
        return []

    # Mapping AV Overview fields to our FinancialMetrics model. This will be partial.
    # report_period would ideally come from actual financial statement dates.
    # For OVERVIEW, we might use the current date or leave it.
    # This is a very simplified mapping.
    try:
        market_cap = float(data.get("MarketCapitalization", 0))
        pe_ratio = float(data.get("PERatio", 0)) if data.get("PERatio") and data.get("PERatio") != "None" else None
        eps = float(data.get("EPS", 0)) if data.get("EPS") and data.get("EPS") != "None" else None
        # ... many other fields would need careful mapping from OVERVIEW or other AV endpoints
        # like INCOME_STATEMENT, BALANCE_SHEET, CASH_FLOW

        # For simplicity, we'll create one FinancialMetrics object.
        # A real implementation would fetch quarterly/annual reports and create multiple objects.
        fm = FinancialMetrics(
            report_period=datetime.datetime.now().strftime("%Y-%m-%d"),  # Placeholder
            currency_symbol=data.get("Currency", "USD"),
            market_cap=market_cap,
            pe_ratio=pe_ratio,
            price_to_sales_ratio=float(data.get("PriceToSalesRatioTTM", 0)) if data.get("PriceToSalesRatioTTM") and data.get("PriceToSalesRatioTTM") != "None" else None,
            price_to_book_ratio=float(data.get("PriceToBookRatio", 0)) if data.get("PriceToBookRatio") and data.get("PriceToBookRatio") != "None" else None,
            earnings_per_share_basic=eps,
            # ... and so on for other fields. Many will be None or require fetching other reports.
            revenue_growth_quarterly=None,
            revenue_growth_annual=None,
            gross_profit_margin_ttm=float(data.get("ProfitMargin", 0)) if data.get("ProfitMargin") and data.get("ProfitMargin") != "None" else None,  # ProfitMargin is not GrossProfitMargin
            operating_margin_ttm=None,
            net_income_margin_ttm=None,
            return_on_equity_ttm=float(data.get("ReturnOnEquityTTM", 0)) if data.get("ReturnOnEquityTTM") and data.get("ReturnOnEquityTTM") != "None" else None,
            debt_to_equity_ratio_quarterly=None,
            dividend_yield=float(data.get("DividendYield", 0)) if data.get("DividendYield") and data.get("DividendYield") != "None" else None,
            beta=float(data.get("Beta", 0)) if data.get("Beta") and data.get("Beta") != "None" else None,
            shares_outstanding=float(data.get("SharesOutstanding")) if data.get("SharesOutstanding") and data.get("SharesOutstanding") != "None" else None,
            # ... fill other fields as None or from other AV calls
        )
        return [fm]
    except (ValueError, TypeError) as e:
        print(f"Error parsing financial metrics for {ticker} from Alpha Vantage: {e}")
        return []


def get_company_news_av(ticker: str, start_date: str | None = None, end_date: str | None = None, limit: int = 50) -> list[CompanyNews]:
    """Fetch company news from Alpha Vantage. AV limit is 50 for free tier news, or 1000 for others.
    Date filtering for AV news is by 'time_from' and 'time_to'.
    """
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "limit": min(limit, 50),  # Max 50 for free tier, adjust if using paid tier
    }
    # AlphaVantage date format: YYYYMMDDTHHMM
    if start_date:
        params["time_from"] = pd.to_datetime(start_date).strftime("%Y%m%dT%H%M")
    if end_date:
        params["time_to"] = (pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).strftime("%Y%m%dT%H%M")  # end of day

    data = _make_alpha_vantage_request(params)

    news_items = []
    feed = data.get("feed", [])
    for item in feed:
        news_items.append(
            CompanyNews(
                id=item.get("url"),  # Use URL as a unique ID
                title=item.get("title"),
                date=pd.to_datetime(item.get("time_published")).strftime("%Y-%m-%dT%H:%M:%S"),  # Convert AV format
                source=item.get("source"),
                text=item.get("summary"),
                url=item.get("url"),
                # Sentiment data is also available in item["ticker_sentiment"]
            )
        )
    return news_items


# --- Modified Generic Functions ---


def get_prices(ticker: str, start_date: str, end_date: str, data_source: str = "financialdatasets") -> list[Price]:
    """Fetch price data from cache or specified API."""
    # Pass start_date and end_date to cache key
    cache_key_kwargs = {"start_date": start_date, "end_date": end_date}

    if data_source == "alphavantage":
        # Attempt to get from cache first for Alpha Vantage as well
        cached_data_av = _cache.get_prices(ticker, data_source=data_source, **cache_key_kwargs)
        if cached_data_av is not None:
            return [Price(**price) for price in cached_data_av]  # Ensure Pydantic model conversion

        prices_av = get_prices_av(ticker, start_date, end_date)
        if prices_av:
            _cache.set_prices(ticker, [p.model_dump() for p in prices_av], data_source=data_source, **cache_key_kwargs)
        return prices_av

    # Default to financialdatasets
    if not FINANCIAL_DATASETS_API_KEY:
        print("Warning: FINANCIAL_DATASETS_API_KEY not found, financialdatasets source will likely fail.")

    # Check cache first (for financialdatasets)
    # Pass start_date and end_date to cache key
    if cached_data := _cache.get_prices(ticker, data_source=data_source, **cache_key_kwargs):
        # The cached data is already filtered by the more specific key, but an additional check won't harm
        # However, the primary filtering should happen due to the specific cache key.
        # For financialdatasets, the API itself filters by start/end date, so data stored under a key
        # specific to that start/end date should be correct.
        return [Price(**price) for price in cached_data]

    headers = {}
    if FINANCIAL_DATASETS_API_KEY:
        headers["X-API-KEY"] = FINANCIAL_DATASETS_API_KEY
    else:
        pass

    url = f"https://api.financialdatasets.ai/prices/?ticker={ticker}&interval=day&interval_multiplier=1&start_date={start_date}&end_date={end_date}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"FD API Error: {ticker} - {response.status_code} - {response.text}")
    price_response = PriceResponse(**response.json())
    prices = price_response.prices
    if not prices:
        _cache.set_prices(ticker, [], data_source=data_source, **cache_key_kwargs)  # Cache empty list
        return []
    _cache.set_prices(ticker, [p.model_dump() for p in prices], data_source=data_source, **cache_key_kwargs)
    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    data_source: str = "financialdatasets",
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from cache or specified API."""
    # Key for caching should include all parameters that affect the API request and data returned.
    cache_key_kwargs = {"end_date": end_date, "period": period, "limit": limit}

    if data_source == "alphavantage":
        # Try cache for Alpha Vantage first
        cached_data_av = _cache.get_financial_metrics(ticker, data_source=data_source, **cache_key_kwargs)
        if cached_data_av is not None:
            return [FinancialMetrics(**metric) for metric in cached_data_av]

        metrics_av = get_financial_metrics_av(ticker, end_date, period, limit)
        if metrics_av:
            _cache.set_financial_metrics(ticker, [m.model_dump() for m in metrics_av], data_source=data_source, **cache_key_kwargs)
        return metrics_av

    if not FINANCIAL_DATASETS_API_KEY:
        print("Warning: FINANCIAL_DATASETS_API_KEY not found, financialdatasets source will likely fail.")

    # Check cache for financialdatasets
    if cached_data := _cache.get_financial_metrics(ticker, data_source=data_source, **cache_key_kwargs):
        # Data from cache is already specific to end_date, period, limit due to the key
        return [FinancialMetrics(**metric) for metric in cached_data]

    headers = {}
    if FINANCIAL_DATASETS_API_KEY:
        headers["X-API-KEY"] = FINANCIAL_DATASETS_API_KEY

    url = f"https://api.financialdatasets.ai/financial-metrics/?ticker={ticker}&report_period_lte={end_date}&limit={limit}&period={period}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"FD API Error: {ticker} - {response.status_code} - {response.text}")
    metrics_response = FinancialMetricsResponse(**response.json())
    financial_metrics = metrics_response.financial_metrics
    if not financial_metrics:
        _cache.set_financial_metrics(ticker, [], data_source=data_source, **cache_key_kwargs)  # Cache empty list
        return []
    _cache.set_financial_metrics(ticker, [m.model_dump() for m in financial_metrics], data_source=data_source, **cache_key_kwargs)
    return financial_metrics


def get_insider_trades(
    ticker: str,
    end_date: str,
    data_source: str = "financialdatasets",
    start_date: str | None = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    """Fetch insider trades. Alpha Vantage does not provide this directly."""
    cache_key_kwargs = {"end_date": end_date, "start_date": start_date, "limit": limit}

    if data_source == "alphavantage":
        print(f"Warning: Insider trades are not available from Alpha Vantage for {ticker}.")
        # Cache that it's an empty list for AV to avoid repeated warnings if desired, though not strictly necessary
        # _cache.set_insider_trades(ticker, [], data_source=data_source, **cache_key_kwargs)
        return []

    if not FINANCIAL_DATASETS_API_KEY:
        print("Warning: FINANCIAL_DATASETS_API_KEY not found, financialdatasets source will likely fail.")

    if cached_data := _cache.get_insider_trades(ticker, data_source=data_source, **cache_key_kwargs):
        return [InsiderTrade(**trade) for trade in cached_data]

    headers = {}
    if FINANCIAL_DATASETS_API_KEY:
        headers["X-API-KEY"] = FINANCIAL_DATASETS_API_KEY
    all_trades = []
    current_end_date_param = end_date  # Use a different variable for pagination to not confuse with cache key end_date
    # ... (rest of the pagination logic needs to be careful if current_end_date_param changes the effective query for caching)
    # For simplicity, the current caching is based on the initial end_date. Pagination might fetch more data
    # than what a strict (ticker, start_date, end_date, limit) key implies if not handled carefully.
    # However, the financialdatasets API itself does the date filtering, so the initial call with specific dates is key.

    url_params = {"ticker": ticker, "filing_date_lte": current_end_date_param, "limit": limit}
    if start_date:
        url_params["filing_date_gte"] = start_date

    # The loop for pagination fetches data in chunks. The cache should ideally store the complete result
    # for the given (start_date, end_date, limit) to avoid re-doing pagination.
    # The current cache logic in api.py for insider trades fetches all pages then caches the full list.
    # This is good. The key should reflect the initial request parameters.

    # The existing pagination logic seems fine for fetching all relevant data first, then caching.
    # The key used for _cache.set_insider_trades should be based on the original start_date, end_date, limit.
    # The loop below implements pagination for financialdatasets
    page_limit_fd = 100  # financialdatasets.ai has a max limit per page, e.g. 100 or 1000
    actual_limit_for_api_call = min(limit, page_limit_fd)  # Respect API page limits

    temp_trades_for_this_request = []
    current_page_end_date = end_date

    while len(temp_trades_for_this_request) < limit:  # Fetch until desired limit is reached or no more data
        url = f"https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_page_end_date}"
        if start_date:
            url += f"&filing_date_gte={start_date}"
        url += f"&limit={actual_limit_for_api_call}"  # Use API page limit

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            # If first page fails, raise error. If subsequent page fails, might return what we have.
            if not temp_trades_for_this_request:
                raise Exception(f"FD API Error: {ticker} - {response.status_code} - {response.text}")
            else:
                print(f"FD API Error on subsequent page: {ticker} - {response.status_code} - {response.text}. Returning partial data.")
                break  # Exit loop and return what's been gathered

        data = response.json()
        response_model = InsiderTradeResponse(**data)
        insider_trades_page = response_model.insider_trades

        if not insider_trades_page:
            break  # No more data

        temp_trades_for_this_request.extend(insider_trades_page)

        if len(insider_trades_page) < actual_limit_for_api_call:  # Last page for this date range
            break

        # Prepare for next page: set end_date for next query to be just before the earliest date of current batch
        # This requires dates to be sortable. Assuming filing_date is YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS
        earliest_filing_date_on_page = min(trade.filing_date for trade in insider_trades_page)

        # Convert to datetime, subtract one second (or day if only date), then format back to string
        try:
            earliest_dt = pd.to_datetime(earliest_filing_date_on_page)
            # Ensure we don't go past start_date if provided
            if start_date and earliest_dt.strftime("%Y-%m-%d") <= start_date:
                break
            current_page_end_date = (earliest_dt - pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S")
            if current_page_end_date < start_date if start_date else False:
                break
        except Exception as e:
            print(f"Error processing date for pagination: {e}. Stopping pagination.")
            break

    all_trades = temp_trades_for_this_request[:limit]  # Ensure we don't exceed original limit

    if not all_trades:
        _cache.set_insider_trades(ticker, [], data_source=data_source, **cache_key_kwargs)
        return []

    _cache.set_insider_trades(ticker, [trade.model_dump() for trade in all_trades], data_source=data_source, **cache_key_kwargs)
    return all_trades


def get_company_news(
    ticker: str,
    end_date: str,
    data_source: str = "financialdatasets",
    start_date: str | None = None,
    limit: int = 1000,  # financialdatasets default
) -> list[CompanyNews]:
    """Fetch company news from specified API."""
    cache_key_kwargs = {"start_date": start_date, "end_date": end_date, "limit": limit}

    if data_source == "alphavantage":
        cached_data_av = _cache.get_company_news(ticker, data_source=data_source, **cache_key_kwargs)
        if cached_data_av is not None:
            return [CompanyNews(**news) for news in cached_data_av]

        news_av = get_company_news_av(ticker, start_date, end_date, limit)
        if news_av:
            _cache.set_company_news(ticker, [n.model_dump() for n in news_av], data_source=data_source, **cache_key_kwargs)
        return news_av

    if not FINANCIAL_DATASETS_API_KEY:
        print("Warning: FINANCIAL_DATASETS_API_KEY not found, financialdatasets source will likely fail.")

    if cached_data := _cache.get_company_news(ticker, data_source=data_source, **cache_key_kwargs):
        return [CompanyNews(**news) for news in cached_data]

    # Pagination logic for financialdatasets news (similar to insider trades)
    headers = {}
    if FINANCIAL_DATASETS_API_KEY:
        headers["X-API-KEY"] = FINANCIAL_DATASETS_API_KEY

    all_news_data = []
    current_page_end_date = end_date
    page_limit_fd = 20  # Example page limit for financialdatasets news API
    actual_limit_for_api_call = min(limit, page_limit_fd)

    while len(all_news_data) < limit:
        url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_page_end_date}"
        if start_date:
            url += f"&start_date={start_date}"
        url += f"&limit={actual_limit_for_api_call}"

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            if not all_news_data:
                raise Exception(f"FD API Error: {ticker} - {response.status_code} - {response.text}")
            else:
                print(f"FD API Error on subsequent page for news: {ticker} - {response.status_code} - {response.text}. Returning partial data.")
                break

        data = response.json()
        response_model = CompanyNewsResponse(**data)
        company_news_page = response_model.news

        if not company_news_page:
            break

        all_news_data.extend(company_news_page)

        if len(company_news_page) < actual_limit_for_api_call:
            break

        earliest_date_on_page = min(news.date for news in company_news_page)
        try:
            earliest_dt = pd.to_datetime(earliest_date_on_page)
            if start_date and earliest_dt.strftime("%Y-%m-%d") <= start_date:
                break
            current_page_end_date = (earliest_dt - pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S")
            if current_page_end_date < start_date if start_date else False:
                break
        except Exception as e:
            print(f"Error processing date for news pagination: {e}. Stopping pagination.")
            break

    final_news = all_news_data[:limit]

    if not final_news:
        _cache.set_company_news(ticker, [], data_source=data_source, **cache_key_kwargs)
        return []

    _cache.set_company_news(ticker, [news.model_dump() for news in final_news], data_source=data_source, **cache_key_kwargs)
    return final_news


def get_market_cap(ticker: str, end_date: str, data_source: str = "financialdatasets") -> float | None:
    """Fetch market cap. For financialdatasets, it uses either company facts (if end_date is today) or financial_metrics.
    For Alpha Vantage, it uses the OVERVIEW endpoint (via get_financial_metrics_av).
    The cache key must include end_date as it's crucial for historical market cap.
    """
    cache_key_kwargs = {"end_date": end_date}

    # Try cache first
    cached_market_cap = _cache.get_market_cap(ticker, data_source=data_source, **cache_key_kwargs)
    if cached_market_cap is not None:
        return cached_market_cap

    market_cap_value: float | None = None

    if data_source == "alphavantage":
        metrics_av = get_financial_metrics_av(ticker, end_date, limit=1)  # end_date is somewhat advisory for AV OVERVIEW
        if metrics_av and metrics_av[0].market_cap is not None:
            market_cap_value = metrics_av[0].market_cap
        else:
            print(f"Warning: Market cap not found for {ticker} using Alpha Vantage on {end_date}.")

    elif data_source == "financialdatasets":
        # For financialdatasets, the logic depends on whether end_date is current or historical.
        # If end_date is today, use company facts API (which gives current market cap).
        # Otherwise, use historical financial_metrics.
        is_today = False
        try:
            # Check if end_date is today. Handle potential date parsing issues.
            end_date_dt = pd.to_datetime(end_date).date()
            today_dt = datetime.datetime.now().date()
            is_today = end_date_dt == today_dt
        except ValueError:
            print(f"Warning: Could not parse end_date '{end_date}' to determine if it's today. Assuming not today for market cap fetch.")

        if is_today:
            headers = {}
            if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
                headers["X-API-KEY"] = api_key
            url = f"https://api.financialdatasets.ai/company/facts/?ticker={ticker}"
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                response_model = CompanyFactsResponse(**data)
                market_cap_value = response_model.company_facts.market_cap
            except requests.exceptions.RequestException as e:
                print(f"Error fetching company facts for market cap: {ticker} - {e}")
            except (ValueError, TypeError) as e:  # Pydantic or JSON error
                print(f"Error parsing company facts response for market cap: {ticker} - {e}")
        else:
            # For historical end_date, get it from financial_metrics
            # Use the already defined get_financial_metrics which handles its own caching.
            # We need to ensure that the financial_metrics call uses the correct end_date for its cache key.
            historical_metrics = get_financial_metrics(ticker, end_date, data_source=data_source, period="ttm", limit=1)
            if historical_metrics and historical_metrics[0].market_cap is not None:
                market_cap_value = historical_metrics[0].market_cap
            else:
                print(f"Warning: Market cap not found from historical metrics for {ticker} on {end_date}.")
    else:
        print(f"Unsupported data_source '{data_source}' for get_market_cap")

    # Cache the fetched market_cap_value (even if None, to avoid refetching for a known miss)
    _cache.set_market_cap(ticker, market_cap_value, data_source=data_source, **cache_key_kwargs)
    return market_cap_value


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


# Update the get_price_data function to use the new functions
def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)
