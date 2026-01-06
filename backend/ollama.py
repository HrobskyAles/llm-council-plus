"""Ollama API client for making LLM requests."""

import logging
import httpx
from typing import List, Dict, Any, Optional
from .config import OLLAMA_HOST, DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = None
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via Ollama API.

    Args:
        model: Ollama model identifier (e.g., "gemma3:latest")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds (defaults to DEFAULT_TIMEOUT from config)

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    
    url = f"http://{OLLAMA_HOST}/api/chat"
    
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
            message = data['message']

            return {
                'content': message.get('content'),
                'reasoning_details': None  # Ollama API doesn't provide this
            }

    except httpx.ConnectError as e:
        logger.error("Connection error querying model %s: Cannot connect to Ollama at %s. Is Ollama running? Error: %s", model, OLLAMA_HOST, e)
        return None
    except httpx.HTTPStatusError as e:
        logger.error("HTTP error querying model %s: Status %s. Response: %s", model, e.response.status_code, e.response.text)
        return None
    except httpx.TimeoutException as e:
        logger.error("Timeout error querying model %s: Request took longer than %ss. Error: %s", model, timeout, e)
        return None
    except Exception as e:
        logger.error("Unexpected error querying model %s: %s: %s", model, type(e).__name__, e)
        return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of Ollama model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    import asyncio

    # Create tasks for all models
    tasks = [query_model(model, messages) for model in models]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}


async def query_models_streaming(
    models: List[str],
    messages: List[Dict[str, str]]
):
    """
    Query multiple models in parallel and yield results as they complete.

    Args:
        models: List of Ollama model identifiers
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


async def query_models_with_stage_timeout(
    models: List[str],
    messages: List[Dict[str, str]],
    stage: str = None,
    stage_timeout: float = 90.0,
    min_results: int = 3
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel with overall stage timeout.

    Uses first-N-complete pattern: returns when either:
    - All models complete
    - stage_timeout is reached (returns completed results)
    - min_results are collected AND timeout reached

    Args:
        models: List of Ollama model identifiers
        messages: List of message dicts to send to each model
        stage: Optional stage identifier for debugging
        stage_timeout: Maximum time to wait for this stage (seconds)
        min_results: Minimum number of results to wait for before timeout applies

    Returns:
        Dict mapping model identifier to response dict
    """
    import asyncio
    import time

    start_time = time.time()
    results = {}

    if stage:
        logger.info("[%s] Starting with stage_timeout=%.1fs, min_results=%d",
                   stage, stage_timeout, min_results)

    # Create named tasks
    async def query_with_name(model: str):
        response = await query_model(model, messages)
        return (model, response)

    # Create ALL tasks at once
    tasks = {asyncio.create_task(query_with_name(model)): model for model in models}
    pending = set(tasks.keys())
    extended_wait_used = False  # Track if we've already done the extended wait

    while pending:
        elapsed = time.time() - start_time
        remaining_timeout = stage_timeout - elapsed

        if remaining_timeout <= 0:
            if len(results) >= min_results:
                logger.warning("[%s] Stage timeout reached after %.1fs with %d/%d results",
                             stage, elapsed, len(results), len(models))
                for task in pending:
                    task.cancel()
                break
            elif not extended_wait_used:
                # Not enough results, wait a bit more (up to 30s extra) - but only once
                extended_wait_used = True
                remaining_timeout = min(30.0, stage_timeout * 0.5)
                logger.warning("[%s] Only %d results, waiting %.1fs more",
                             stage, len(results), remaining_timeout)
            else:
                # Extended wait already used, give up and return what we have
                logger.warning("[%s] Extended wait exhausted after %.1fs with only %d/%d results, proceeding anyway",
                             stage, elapsed, len(results), len(models))
                for task in pending:
                    task.cancel()
                break

        try:
            done, pending = await asyncio.wait(
                pending,
                timeout=remaining_timeout,
                return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                try:
                    model, response = task.result()
                    results[model] = response
                except Exception as e:
                    model = tasks[task]
                    logger.error("[%s] Task for %s failed: %s", stage, model, e)
                    results[model] = {
                        'error': True,
                        'error_type': 'unknown',
                        'error_message': str(e)
                    }

            if not done and not pending:
                break

        except asyncio.TimeoutError:
            if len(results) >= min_results:
                logger.warning("[%s] Timeout with %d/%d results, proceeding",
                             stage, len(results), len(models))
                for task in pending:
                    task.cancel()
                break

    # Add timeout entries for missing models
    for model in models:
        if model not in results:
            results[model] = {
                'error': True,
                'error_type': 'stage_timeout',
                'error_message': f'Model did not respond within stage timeout ({stage_timeout}s)'
            }

    total_time = time.time() - start_time
    success_count = sum(1 for r in results.values() if r and not r.get('error'))
    logger.info("[%s] Completed in %.2fs: %d success, %d failed",
               stage, total_time, success_count, len(models) - success_count)

    return results
