"""Llama.cpp server API client for making LLM requests.

Llama.cpp server runs an HTTP API that is compatible with the OpenAI Chat Completions API.
Default endpoint: http://localhost:8080/v1/chat/completions
"""

import logging
import httpx
from typing import List, Dict, Any, Optional, Union
from .config import LLAMACPP_HOST, DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)


def build_message_content(
    text: str,
    images: Optional[List[Dict[str, str]]] = None
) -> Union[str, List[Dict[str, Any]]]:
    """
    Build message content for llama.cpp API.

    For text-only messages, returns a string.
    For multimodal messages (with images), returns an array of content parts.

    Args:
        text: The text content of the message
        images: Optional list of image dicts with 'content' (base64 data URI) and 'filename'

    Returns:
        Either a string (text only) or a list of content parts (multimodal)
    """
    if not images:
        return text

    # Build multimodal content array
    content = [
        {"type": "text", "text": text}
    ]

    for image in images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": image["content"]  # base64 data URI
            }
        })

    return content


async def query_model(
    model: str,
    messages: List[Dict[str, Any]],
    timeout: float = None,
    stage: str = None
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via llama.cpp server API.

    Args:
        model: Llama.cpp model identifier (e.g., "llama3.1")
        messages: List of message dicts with 'role' and 'content'.
                  Content can be a string (text only) or an array of content parts
                  for multimodal messages (see build_message_content).
        timeout: Request timeout in seconds (defaults to DEFAULT_TIMEOUT from config)
        stage: Optional stage identifier for debugging (e.g., "STAGE1", "STAGE2", "STAGE3")

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    if timeout is None:
        timeout = DEFAULT_TIMEOUT

    if stage:
        logger.debug("[%s] Querying model: %s", stage, model)

    url = f"http://{LLAMACPP_HOST}/v1/chat/completions"

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            message = data['choices'][0]['message']

            return {
                'content': message.get('content'),
                'reasoning_details': message.get('reasoning_details')
            }

    except httpx.ConnectError as e:
        logger.error("Connection error querying model %s: Cannot connect to llama.cpp at %s. Is the server running? Error: %s", model, LLAMACPP_HOST, e)
        return {
            'error': True,
            'error_type': 'connection',
            'error_message': 'Cannot connect to llama.cpp server'
        }
    except httpx.HTTPStatusError as e:
        logger.error("HTTP error querying model %s: Status %s. Response: %s", model, e.response.status_code, e.response.text[:500])
        return {
            'error': True,
            'error_type': 'http',
            'error_message': f'HTTP {e.response.status_code}: {e.response.text[:200]}'
        }
    except httpx.TimeoutException as e:
        logger.error("Timeout error querying model %s: Request took longer than %ss. Error: %s", model, timeout, e)
        return {
            'error': True,
            'error_type': 'timeout',
            'error_message': f'Request timed out after {timeout}s'
        }
    except Exception as e:
        logger.error("Unexpected error querying model %s: %s: %s", model, type(e).__name__, e)
        return {
            'error': True,
            'error_type': 'unknown',
            'error_message': str(e)
        }


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, Any]],
    stage: str = None
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of llama.cpp model identifiers
        messages: List of message dicts to send to each model
        stage: Optional stage identifier for debugging (e.g., "STAGE1", "STAGE2")

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    import asyncio

    if stage:
        logger.debug("[%s] Querying %d models in parallel...", stage, len(models))

    # Create tasks for all models
    tasks = [query_model(model, messages, stage=stage) for model in models]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}


async def query_models_streaming(
    models: List[str],
    messages: List[Dict[str, Any]]
):
    """
    Query multiple models in parallel and yield results as they complete.

    Args:
        models: List of llama.cpp model identifiers
        messages: List of message dicts to send to each model

    Yields:
        Tuple of (model, response) as each model completes
    """
    import asyncio
    import time

    start_time = time.time()
    logger.debug("[PARALLEL] Starting %d model queries at t=0.0s", len(models))

    # Create named tasks so we can identify which model completed
    async def query_with_name(model: str):
        req_start = time.time() - start_time
        logger.debug("[PARALLEL] Starting request to %s at t=%.2fs", model, req_start)
        response = await query_model(model, messages)
        req_end = time.time() - start_time
        logger.debug("[PARALLEL] Got response from %s at t=%.2fs", model, req_end)
        return (model, response)

    # Create ALL tasks at once - they start executing immediately in parallel
    tasks = [asyncio.create_task(query_with_name(model)) for model in models]
    logger.debug("[PARALLEL] All %d tasks created and running in parallel", len(tasks))

    # Yield results as they complete (first finished = first yielded)
    for coro in asyncio.as_completed(tasks):
        model, response = await coro
        yield_time = time.time() - start_time
        logger.debug("[PARALLEL] Yielding %s at t=%.2fs", model, yield_time)
        yield (model, response)
