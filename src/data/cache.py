import appdirs
import diskcache
import os
from typing import List, Dict, Any

# Define a directory for the cache
CACHE_DIR = os.path.join(appdirs.user_cache_dir("ai_hedge_fund", "AICopilot"), "api_data_cache_v3")  # Incremented version
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR, exist_ok=True)

# Initialize the persistent cache
# Timeout: 1 week (604800 seconds)
_persistent_cache = diskcache.Cache(CACHE_DIR, timeout=60 * 60 * 24 * 7)


class Cache:
    """Persistent disk cache for API responses using diskcache."""

    def __init__(self):
        self.cache = _persistent_cache  # Initialize instance variable
        self.expire_time = 60 * 60 * 24 * 7  # 1 week, ensure it's defined if used in set_line_items

    def _generate_key(self, data_type: str, ticker: str, data_source: str = "financialdatasets", **kwargs) -> str:
        """Generates a unique key for caching, including the data source and any other relevant parameters.
        Sorts all kwargs to ensure consistent key order.
        Handles list/tuple kwargs by sorting their string representations.
        """
        key_parts = {"data_type": str(data_type), "ticker": str(ticker.upper()), "data_source": str(data_source)}

        # Process kwargs carefully, sorting by key for consistency before adding to key_parts
        for k, v_val in sorted(kwargs.items()):  # Sort kwargs by key
            key_name = str(k)
            if v_val is None:
                processed_val = "NONEVAL"  # Specific marker for None values
            elif isinstance(v_val, (list, tuple)):
                # Sort string representations of list/tuple items for consistency, add prefix
                processed_val = "LIST_" + "_".join(sorted([str(i).replace("_", "__").replace(":", "--") for i in v_val]))
            elif isinstance(v_val, bool):
                processed_val = "BOOL_" + str(v_val)
            elif isinstance(v_val, (int, float)):
                processed_val = "NUM_" + str(v_val)
            else:  # Assume string or can be safely converted to string
                # Sanitize strings to prevent issues with underscores or colons if they are part of values
                processed_val = str(v_val).replace("_", "__").replace(":", "--")

            key_parts[key_name] = processed_val

        # Final key construction: sort all components by key and join
        sorted_final_components = sorted(key_parts.items())
        final_key = "_".join(f"{comp_k}_{comp_v}" for comp_k, comp_v in sorted_final_components)
        # Uncomment for debugging cache key generation:
        # print(f"Generated cache key: {final_key} for data_type={data_type}, ticker={ticker}, ds={data_source}, kwargs={kwargs}")
        return final_key

    def get_prices(self, ticker: str, data_source: str = "financialdatasets", **kwargs) -> List[Dict[str, Any]] | None:
        key = self._generate_key("prices", ticker, data_source, **kwargs)
        return self.cache.get(key)

    def set_prices(self, ticker: str, data: List[Dict[str, Any]], data_source: str = "financialdatasets", **kwargs):
        key = self._generate_key("prices", ticker, data_source, **kwargs)
        self.cache.set(key, data, expire=self.expire_time)

    def get_financial_metrics(self, ticker: str, data_source: str = "financialdatasets", **kwargs) -> List[Dict[str, Any]] | None:
        key = self._generate_key("financial_metrics", ticker, data_source, **kwargs)
        return self.cache.get(key)

    def set_financial_metrics(self, ticker: str, data: List[Dict[str, Any]], data_source: str = "financialdatasets", **kwargs):
        key = self._generate_key("financial_metrics", ticker, data_source, **kwargs)
        self.cache.set(key, data, expire=self.expire_time)

    def get_line_items(self, ticker: str, query: list[str] | str, period: str, limit: int, end_date: str | None, data_source: str = "financialdatasets") -> list[dict] | None:
        # Pass all relevant params for key generation
        key_args = {"query": query, "period": period, "limit": limit, "end_date": end_date}
        key = self._generate_key("line_items", ticker, data_source, **key_args)
        cached_value = self.cache.get(key)
        return cached_value

    def set_line_items(self, ticker: str, query: list[str] | str, period: str, limit: int, end_date: str | None, data: list[dict], data_source: str = "financialdatasets"):
        key_args = {"query": query, "period": period, "limit": limit, "end_date": end_date}
        key = self._generate_key("line_items", ticker, data_source, **key_args)
        self.cache.set(key, data, expire=self.expire_time)

    def get_insider_trades(self, ticker: str, data_source: str = "financialdatasets", **kwargs) -> List[Dict[str, Any]] | None:
        key = self._generate_key("insider_trades", ticker, data_source, **kwargs)
        return self.cache.get(key)

    def set_insider_trades(self, ticker: str, data: List[Dict[str, Any]], data_source: str = "financialdatasets", **kwargs):
        key = self._generate_key("insider_trades", ticker, data_source, **kwargs)
        self.cache.set(key, data, expire=self.expire_time)

    def get_company_news(self, ticker: str, data_source: str = "financialdatasets", **kwargs) -> List[Dict[str, Any]] | None:
        key = self._generate_key("company_news", ticker, data_source, **kwargs)
        return self.cache.get(key)

    def set_company_news(self, ticker: str, data: List[Dict[str, Any]], data_source: str = "financialdatasets", **kwargs):
        key = self._generate_key("company_news", ticker, data_source, **kwargs)
        self.cache.set(key, data, expire=self.expire_time)

    def get_market_cap(self, ticker: str, data_source: str = "financialdatasets", **kwargs) -> float | None:
        # end_date should be passed in kwargs for key generation
        key = self._generate_key("market_cap", ticker, data_source, **kwargs)
        return self.cache.get(key)

    def set_market_cap(self, ticker: str, data: float | None, data_source: str = "financialdatasets", **kwargs):
        # end_date should be passed in kwargs for key generation
        key = self._generate_key("market_cap", ticker, data_source, **kwargs)
        self.cache.set(key, data, expire=self.expire_time)

    def clear_all(self):
        """Clears the entire cache."""
        self.cache.clear()  # Use self.cache
        print(f"Persistent cache cleared at {CACHE_DIR}")


# Global cache instance
_cache_instance = Cache()


def get_cache() -> Cache:
    """Get the global cache instance."""
    return _cache_instance


if __name__ == "__main__":
    # Example usage and testing
    cache = get_cache()
    # cache.clear_all() # Uncomment to clear cache before testing

    ticker_symbol = "TESTCACHEAAPL"
    ds_fd = "financialdatasets"
    ds_av = "alphavantage"

    print(f"Cache directory: {CACHE_DIR}")

    # Test prices for financialdatasets with extra params
    print(f"--- Testing Prices ({ds_fd}) for {ticker_symbol} with date ---")
    date_param = "2023-01-01"
    cached_prices_fd_dated = cache.get_prices(ticker_symbol, data_source=ds_fd, date=date_param)
    if cached_prices_fd_dated:
        print(f"Found cached prices for {ticker_symbol} ({ds_fd}, date={date_param}): {len(cached_prices_fd_dated)} items.")
    else:
        print(f"No cached prices for {ticker_symbol} ({ds_fd}, date={date_param}). Simulating set...")
        sample_prices_fd_dated = [{"time": "2023-01-01", "close": 150.0, "source": ds_fd}]
        cache.set_prices(ticker_symbol, sample_prices_fd_dated, data_source=ds_fd, date=date_param)
        print(f"Set sample prices for {ticker_symbol} ({ds_fd}, date={date_param})")
        retrieved_prices_fd_dated = cache.get_prices(ticker_symbol, data_source=ds_fd, date=date_param)
        print(f"Retrieved after set: {retrieved_prices_fd_dated}")

    # Test prices for Alpha Vantage
    print(f"--- Testing Prices ({ds_av}) for {ticker_symbol} ---")
    cached_prices_av = cache.get_prices(ticker_symbol, data_source=ds_av)
    if cached_prices_av:
        print(f"Found cached prices for {ticker_symbol} ({ds_av}): {len(cached_prices_av)} items. First item: {cached_prices_av[0]}")
    else:
        print(f"No cached prices for {ticker_symbol} ({ds_av}). Simulating set...")
        sample_prices_av = [{"time": "2024-01-01", "close": 150.0, "source": ds_av}, {"time": "2024-01-02", "close": 151.0, "source": ds_av}]
        cache.set_prices(ticker_symbol, sample_prices_av, data_source=ds_av)
        print(f"Set sample prices for {ticker_symbol} ({ds_av})")
        retrieved_prices_av = cache.get_prices(ticker_symbol, data_source=ds_av)
        print(f"Retrieved after set: {retrieved_prices_av}")

    # Test financial_metrics for financialdatasets
    print(f"--- Testing Financial Metrics ({ds_fd}) for {ticker_symbol} ---")
    cached_fm_fd = cache.get_financial_metrics(ticker_symbol, data_source=ds_fd, period="annual")
    if cached_fm_fd:
        print(f"Found cached financial metrics for {ticker_symbol} ({ds_fd}, period=annual): {len(cached_fm_fd)} items.")
    else:
        print(f"No cached financial metrics for {ticker_symbol} ({ds_fd}, period=annual). Simulating set...")
        sample_fm_fd = [{"metric": "revenue", "value": 1000000, "source": ds_fd}]
        cache.set_financial_metrics(ticker_symbol, sample_fm_fd, data_source=ds_fd, period="annual")
        print(f"Set sample financial metrics for {ticker_symbol} ({ds_fd}, period=annual)")
        retrieved_fm_fd = cache.get_financial_metrics(ticker_symbol, data_source=ds_fd, period="annual")
        print(f"Retrieved after set: {retrieved_fm_fd}")

    # Test market_cap for alphavantage
    print(f"--- Testing Market Cap ({ds_av}) for {ticker_symbol} ---")
    cached_mc_av = cache.get_market_cap(ticker_symbol, data_source=ds_av)
    if cached_mc_av:
        print(f"Found cached market cap for {ticker_symbol} ({ds_av}): {cached_mc_av}")
    else:
        print(f"No cached market cap for {ticker_symbol} ({ds_av}). Simulating set...")
        sample_mc_av = {"market_cap": "1T", "source": ds_av}
        cache.set_market_cap(ticker_symbol, sample_mc_av, data_source=ds_av)
        print(f"Set sample market cap for {ticker_symbol} ({ds_av})")
        retrieved_mc_av = cache.get_market_cap(ticker_symbol, data_source=ds_av)
        print(f"Retrieved after set: {retrieved_mc_av}")

    print("--- Cache test complete ---")
