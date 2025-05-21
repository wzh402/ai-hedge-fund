"""Helper functions for LLM"""

import json
from typing import TypeVar, Type, Optional, Any
from pydantic import BaseModel
from src.llm.models import get_model, get_model_info
from src.utils.progress import progress

T = TypeVar("T", bound=BaseModel)


def call_llm(
    prompt: Any,
    model_name: str,
    model_provider: str,
    pydantic_model: Type[T],
    agent_name: Optional[str] = None,
    max_retries: int = 3,
    default_factory=None,
) -> T:
    """
    Makes an LLM call with retry logic, handling both JSON supported and non-JSON supported models.

    Args:
        prompt: The prompt to send to the LLM
        model_name: Name of the model to use
        model_provider: Provider of the model
        pydantic_model: The Pydantic model class to structure the output
        agent_name: Optional name of the agent for progress updates
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure

    Returns:
        An instance of the specified Pydantic model
    """

    model_info = get_model_info(model_name, model_provider)
    llm = get_model(model_name, model_provider)

    if llm is None:
        error_msg = f"Error: Could not initialize LLM model '{model_name}' from provider '{model_provider}'. Please check your LLM configuration in src/llm/models.py."
        print(error_msg)
        if agent_name:
            progress.update_status(agent_name, None, f"LLM init error: {model_name} ({model_provider})")
        if default_factory:
            return default_factory()
        return create_default_response(pydantic_model)

    # If model_info is None (no specific info) or if model_info indicates JSON mode is supported,
    # attempt to use with_structured_output with json_mode.
    structured_output_applied = False
    if not model_info or model_info.has_json_mode():
        try:
            llm = llm.with_structured_output(
                pydantic_model,
                method="json_mode",
            )
            structured_output_applied = True
        except Exception as e:
            print(f"Warning: Failed to apply .with_structured_output(method='json_mode') to {model_name} ({model_provider}): {e}. Will attempt manual JSON parsing if applicable.")
            # If this fails, structured_output_applied remains False, and manual parsing might be attempted later.

    # Call the LLM with retries
    for attempt in range(max_retries):
        try:
            result = llm.invoke(prompt)

            # Case 1: Structured output was applied, result should be the Pydantic model.
            if structured_output_applied:
                if isinstance(result, pydantic_model):
                    return result
                else:
                    # This is unexpected if with_structured_output was supposed to work.
                    # Try to parse manually as a fallback.
                    response_content_fallback = str(result.content) if hasattr(result, "content") else str(result)
                    if agent_name:
                        progress.update_status(agent_name, None, f"Warning: Expected Pydantic model from {model_name} (structured_output), got {type(result)}. Attempting manual parse. Attempt {attempt + 1}/{max_retries}")
                    parsed_fallback = extract_json_from_response(response_content_fallback)
                    if parsed_fallback:
                        try:
                            return pydantic_model(**parsed_fallback)
                        except Exception as p_exc_fallback:
                            raise ValueError(f"Fallback manual parsing succeeded but Pydantic validation failed for {model_name}: {p_exc_fallback}. Parsed: {parsed_fallback}") from p_exc_fallback
                    raise ValueError(f"Expected Pydantic model from {model_name} (structured_output), but got {type(result)} and unparsable content: {response_content_fallback[:200]}")

            # Case 2: Structured output was NOT applied (either because model_info indicated no JSON mode, or .with_structured_output failed).
            # This implies model_info exists and model_info.has_json_mode() is False, or structured_output_applied is False.
            # We need to manually parse.
            elif model_info and not model_info.has_json_mode():
                response_content = str(result.content) if hasattr(result, "content") else str(result)
                parsed_result = extract_json_from_response(response_content)
                if parsed_result:
                    try:
                        return pydantic_model(**parsed_result)
                    except Exception as p_exc:
                        if agent_name:
                            progress.update_status(agent_name, None, f"Pydantic validation error for {model_name} (manual parse). Attempt {attempt + 1}/{max_retries}. Error: {str(p_exc)[:100]}")
                        if attempt == max_retries - 1:
                            print(f"Pydantic validation error for {model_name} ({model_provider}) for agent {agent_name} after manual parse: {p_exc}. Parsed content: {parsed_result}")
                        raise  # Reraise to be caught by the outer try-except for retry
                else:  # Failed to extract JSON
                    error_msg_json = f"Failed to extract JSON from non-JSON mode model response for {model_name}. Content snippet: {response_content[:200]}"
                    if agent_name:
                        progress.update_status(agent_name, None, f"{error_msg_json} Attempt {attempt + 1}/{max_retries}.")
                    raise ValueError(error_msg_json)
            else:
                # This is a catch-all for unexpected state, e.g. model_info is None and structured_output was not applied or failed.
                # Treat as raw output and try to parse.
                response_content_raw = str(result.content) if hasattr(result, "content") else str(result)
                if agent_name:
                    progress.update_status(agent_name, None, f"Warning: Unexpected LLM output state for {model_name}. Attempting manual parse. Attempt {attempt + 1}/{max_retries}")
                parsed_raw = extract_json_from_response(response_content_raw)
                if parsed_raw:
                    try:
                        return pydantic_model(**parsed_raw)
                    except Exception as p_exc_raw:
                        raise ValueError(f"Raw manual parsing succeeded but Pydantic validation failed for {model_name}: {p_exc_raw}. Parsed: {parsed_raw}") from p_exc_raw
                raise ValueError(f"Unexpected LLM output state for {model_name}, and failed to manually parse. Content: {response_content_raw[:200]}")

        except Exception as e:
            error_details = str(e)[:200]  # Get a snippet of the error
            if agent_name:
                progress.update_status(agent_name, None, f"Error - retry {attempt + 1}/{max_retries} for {model_name}: {error_details}")

            if attempt == max_retries - 1:
                final_error_msg = f"Error in LLM call to {model_name} ({model_provider}) for agent {agent_name} after {max_retries} attempts: {e}"
                print(final_error_msg)
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)

    # Fallback, should ideally be unreachable if max_retries >= 1
    final_fallback_msg = f"LLM call to {model_name} ({model_provider}) for agent {agent_name} failed after all retries and fallbacks."
    print(final_fallback_msg)
    if default_factory:
        return default_factory()
    return create_default_response(pydantic_model)


def create_default_response(model_class: Type[T]) -> T:
    """Creates a safe default response based on the model's fields."""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
            default_values[field_name] = {}
        else:
            # For other types (like Literal), try to use the first allowed value
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None

    return model_class(**default_values)


def extract_json_from_response(content: str) -> Optional[dict]:
    """Extracts JSON from markdown-formatted response."""
    try:
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7 :]  # Skip past ```json
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
                return json.loads(json_text)
    except Exception as e:
        print(f"Error extracting JSON from response: {e}")
    return None
