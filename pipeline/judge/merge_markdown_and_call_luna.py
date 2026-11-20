#!/usr/bin/env python3
"""Merge two Markdown files into a prompt and save a model response.

The model receives the merged Markdown as its user input.  Optional local
figure files are attached as multimodal content parts.  No system prompt is
added.

Before running, fill in the five paths in the configuration section below, or
provide the paths with command-line options.

Example:
    python3 scionbench/judge/merge_markdown_and_call_luna.py

The API is called with the standard OpenAI-compatible HTTP interface and uses
only Python's standard library.  The default endpoint is the MaaS Kimi K3
``/v1/chat/completions`` gateway used by the judge pipeline.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
import socket
import ssl
import sys
import time
from http.client import (
    BadStatusLine,
    HTTPException,
    IncompleteRead,
    RemoteDisconnected,
)
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


class _NoRedirectHandler(HTTPRedirectHandler):
    """Leave redirect responses visible to the caller.

    urllib's default handler may turn a redirected POST into a GET for 301/302
    responses.  The judge must make that decision itself so the request body,
    method, and authorization policy are explicit.
    """

    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


_NO_REDIRECT_OPENER = build_opener(_NoRedirectHandler)


def urlopen(request: Request, timeout: int | None = None) -> Any:
    """Open a request without urllib's implicit redirect rewriting.

    Keeping this as a module-level function also makes transport behavior easy
    to replace in tests without contacting a real gateway.
    """

    return _NO_REDIRECT_OPENER.open(request, timeout=timeout)


# ---------------------------------------------------------------------------
# Fill these paths before running.  Command-line options override these values.
# ---------------------------------------------------------------------------
GOLDEN_NODES = ""
AGENT_OUTPUT_MD = ""
PROMPT_MD = ""
MERGED_MD_OUTPUT = ""
MODEL_OUTPUT_MD = ""


# ---------------------------------------------------------------------------
# API configuration.  The complete pipeline injects the key from
# ``pipeline/judge_dispatch.py``.  Direct invocations should export
# ``AIGW_API_KEY`` in the shell instead of adding a second key to this file.
# ---------------------------------------------------------------------------
MODEL = os.environ.get("JUDGE_MODEL", "")
API_MODE = "chat_completions"  # "responses" or "chat_completions"
API_BASE_URL = os.environ.get("JUDGE_API_ENDPOINT", "")
API_KEY = ""
API_KEY_ENV = "AIGW_API_KEY"
THINKING = False
TEMPERATURE = 1.0
STREAM = False
TOP_P: float | None = None
MAX_TOKENS: int | None = None
REQUEST_TIMEOUT_SECONDS = 600
MAX_RETRIES = 10
RETRY_DELAY_SECONDS = 2
# Keep the complete JSON POST body below a conservative gateway-friendly size.
# The value includes prompt text, image data URLs, and JSON framing.
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_IMAGES_PER_REQUEST = 24
MAX_IMAGE_DIMENSION = 1600
IMAGE_QUALITY = 75
MAX_REDIRECTS = 3


PLACEHOLDERS = {
    "{{GOLDEN_NODES}}": "GOLDEN_NODES",
    "{{AGENT_OUTPUT_MD}}": "AGENT_OUTPUT_MD",
}
RETRYABLE_HTTP_STATUS_CODES = {408, 409, 425, 429}
REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}


class ScriptError(Exception):
    """An expected, user-correctable script error."""


class RetryableAPIError(ScriptError):
    """An API/protocol failure which is safe to retry."""


def _find_api_error(value: Any) -> Any | None:
    """Find an error object in common top-level or wrapped response shapes."""

    if isinstance(value, dict):
        if "error" in value and value["error"]:
            return value["error"]
        for key in ("data", "result", "response", "output"):
            if key in value:
                found = _find_api_error(value[key])
                if found is not None:
                    return found
    elif isinstance(value, list):
        for item in value:
            found = _find_api_error(item)
            if found is not None:
                return found
    return None


def _read_markdown(path: Path, label: str) -> str:
    if not path.is_file():
        raise ScriptError(f"{label} does not exist or is not a file: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScriptError(f"cannot read {label} {path}: {exc}") from exc


def merge_markdown(
    golden_nodes_path: Path, agent_output_path: Path, prompt_path: Path
) -> str:
    """Replace both supported placeholders in the prompt template."""

    golden_nodes = _read_markdown(golden_nodes_path, "GOLDEN_NODES")
    agent_output = _read_markdown(agent_output_path, "AGENT_OUTPUT_MD")
    prompt = _read_markdown(prompt_path, "PROMPT_MD")

    missing = [placeholder for placeholder in PLACEHOLDERS if placeholder not in prompt]
    if missing:
        names = ", ".join(missing)
        raise ScriptError(f"prompt is missing placeholder(s): {names}")

    return prompt.replace("{{GOLDEN_NODES}}", golden_nodes).replace(
        "{{AGENT_OUTPUT_MD}}", agent_output
    )


def _write_text(path: Path, text: str, label: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise ScriptError(f"cannot write {label} {path}: {exc}") from exc


def _resolve_path(raw_path: str | None, configured_path: str, label: str) -> Path:
    value = configured_path if raw_path is None else raw_path
    if not value or not value.strip():
        raise ScriptError(
            f"{label} is empty; fill it in the configuration section or pass the "
            f"corresponding command-line option"
        )
    return Path(value).expanduser().resolve()


def _resolve_api_key(api_key: str, api_key_env: str) -> str:
    if api_key.strip():
        return api_key.strip()

    if api_key_env.strip():
        configured_key = os.getenv(api_key_env, "").strip()
        if configured_key:
            return configured_key

    if api_key_env.strip():
        raise ScriptError(
            f"API key is unavailable; set {api_key_env} in the environment"
        )
    raise ScriptError("API key is unavailable; set AIGW_API_KEY in the environment")


def _resolve_api_base_url(configured_url: str) -> str:
    url = configured_url.strip() or os.getenv("OPENAI_BASE_URL", "").strip()
    if not url:
        raise ScriptError(
            "API_BASE_URL is empty; fill it in the configuration section or set "
            "OPENAI_BASE_URL"
        )
    url = url.rstrip("/")
    # Accept either the configured /v1 root or the full endpoint shown in the
    # gateway curl example; call_model appends the endpoint exactly once.
    for suffix in ("/chat/completions", "/responses"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    return url.rstrip("/")


def _resolve_image_paths(raw_paths: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for raw_path in raw_paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise ScriptError(f"image does not exist or is not a file: {path}")
        media_type, _ = mimetypes.guess_type(path.name)
        if not media_type or not media_type.startswith("image/"):
            raise ScriptError(f"unsupported image type for {path}")
        if path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def _image_data_url(path: Path) -> str:
    media_type, _ = mimetypes.guess_type(path.name)
    if not media_type or not media_type.startswith("image/"):
        raise ScriptError(f"unsupported image type for {path}")
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError as exc:
        raise ScriptError(f"cannot read image {path}: {exc}") from exc
    return f"data:{media_type};base64,{encoded}"


def _encoded_image_candidates(path: Path) -> Iterable[tuple[str, str, bool]]:
    """Yield original and progressively smaller data URLs for one image.

    Pillow is optional.  When it is unavailable, the original image remains
    usable and the request-size budget can still omit it safely.
    """

    media_type, _ = mimetypes.guess_type(path.name)
    if not media_type or not media_type.startswith("image/"):
        raise ScriptError(f"unsupported image type for {path}")

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ScriptError(f"cannot read image {path}: {exc}") from exc

    original_url = f"data:{media_type};base64,{base64.b64encode(raw).decode('ascii')}"
    seen: set[str] = set()
    seen.add(original_url)
    yield media_type, original_url, False

    try:
        from PIL import Image
    except ImportError:
        return

    try:
        with Image.open(io.BytesIO(raw)) as source:
            source.load()
            source_mode = source.mode
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            for dimension, quality in (
                (MAX_IMAGE_DIMENSION, IMAGE_QUALITY),
                (1200, 65),
                (800, 55),
                (600, 45),
            ):
                image = source.copy()
                image.thumbnail((dimension, dimension), resampling)
                has_alpha = "A" in source_mode or "transparency" in source.info
                if has_alpha:
                    # JPEG is substantially smaller for large figures.  A
                    # white background preserves visual content for judging.
                    rgba = image.convert("RGBA")
                    background = Image.new("RGB", rgba.size, "white")
                    background.paste(rgba, mask=rgba.getchannel("A"))
                    image = background
                else:
                    image = image.convert("RGB")
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=quality, optimize=True)
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                candidate = f"data:image/jpeg;base64,{encoded}"
                if candidate not in seen:
                    seen.add(candidate)
                    yield "image/jpeg", candidate, True
    except Exception as exc:  # noqa: BLE001 - fall back to the original bytes.
        print(
            f"warning: could not resize image {path}: {exc}; using original bytes",
            file=sys.stderr,
        )


def _content_size_bytes(content: Any) -> int:
    return len(
        json.dumps(
            {"messages": [{"role": "user", "content": content}]},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _serialized_size_bytes(payload: Any) -> int:
    """Return the exact UTF-8 size used for an outgoing JSON request."""

    return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def _image_part_refs(value: Any) -> list[tuple[list[Any], int]]:
    """Find removable image content parts in a nested request payload."""

    refs: list[tuple[list[Any], int]] = []
    if isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, dict) and item.get("type") in {
                "image_url",
                "input_image",
            }:
                refs.append((value, index))
            else:
                refs.extend(_image_part_refs(item))
    elif isinstance(value, dict):
        for item in value.values():
            refs.extend(_image_part_refs(item))
    return refs


def _trim_payload_images(payload: dict[str, Any], max_request_bytes: int) -> int:
    """Drop least-recent image parts until the serialized body fits the cap."""

    removed = 0
    while _serialized_size_bytes(payload) > max_request_bytes:
        refs = _image_part_refs(payload)
        if not refs:
            raise ScriptError(
                "text and request metadata exceed the maximum request size "
                f"of {max_request_bytes} bytes"
            )
        parent, index = refs[-1]
        parent.pop(index)
        removed += 1
    if removed:
        print(
            f"image budget: dropped {removed} image part(s) to keep the "
            f"serialized request at or below {max_request_bytes} bytes",
            file=sys.stderr,
        )
    return removed


def _prepare_image_urls(
    merged_prompt: str,
    image_paths: Iterable[Path],
    *,
    max_request_bytes: int = MAX_REQUEST_BYTES,
    max_images: int = MAX_IMAGES_PER_REQUEST,
) -> list[tuple[str, str]]:
    """Fit text and images under a bounded JSON request-size budget."""

    paths = list(image_paths)
    if max_request_bytes <= 0:
        raise ScriptError("max request size must be greater than zero")
    if max_images < 0:
        raise ScriptError("max image count cannot be negative")

    text_content: Any = merged_prompt
    if _content_size_bytes(text_content) > max_request_bytes:
        raise ScriptError(
            "text prompt alone exceeds the maximum request size "
            f"({max_request_bytes} bytes)"
        )

    # The final payload is checked again after model-specific fields are added.
    # Using the complete available budget here lets small requests retain as
    # many images as possible; the final check can drop trailing images if the
    # endpoint metadata pushes the body over the exact cap.
    budget = max_request_bytes
    content: list[dict[str, Any]] = [{"type": "text", "text": merged_prompt}]
    selected: list[tuple[str, str]] = []
    resized = 0
    skipped = 0

    for index, path in enumerate(paths):
        if index >= max_images:
            skipped += 1
            continue
        chosen: tuple[str, str, bool] | None = None
        for candidate in _encoded_image_candidates(path):
            media_type, data_url, was_resized = candidate
            candidate_item = {
                "type": "image_url",
                "image_url": {"url": data_url},
            }
            if _content_size_bytes(content + [candidate_item]) <= budget:
                chosen = candidate
                break
        if chosen is None:
            skipped += 1
            continue
        media_type, data_url, was_resized = chosen
        content.append(
            {"type": "image_url", "image_url": {"url": data_url}}
        )
        selected.append((media_type, data_url))
        resized += int(was_resized)

    if paths and (resized or skipped):
        print(
            "image budget: "
            f"requested={len(paths)} attached={len(selected)} "
            f"resized={resized} skipped={skipped} "
            f"limit={max_request_bytes} bytes/{max_images} images",
            file=sys.stderr,
        )
    return selected


def _chat_user_content_from_urls(
    merged_prompt: str, image_urls: Iterable[tuple[str, str]]
) -> Any:
    urls = list(image_urls)
    if not urls:
        return merged_prompt
    content: list[dict[str, Any]] = [{"type": "text", "text": merged_prompt}]
    content.extend(
        {"type": "image_url", "image_url": {"url": data_url}}
        for _, data_url in urls
    )
    return content


def _responses_input_from_urls(
    merged_prompt: str, image_urls: Iterable[tuple[str, str]]
) -> Any:
    urls = list(image_urls)
    if not urls:
        return merged_prompt
    content: list[dict[str, Any]] = [
        {"type": "input_text", "text": merged_prompt}
    ]
    content.extend(
        {"type": "input_image", "image_url": data_url}
        for _, data_url in urls
    )
    return [{"role": "user", "content": content}]


def _chat_user_content(
    merged_prompt: str,
    image_paths: Iterable[Path],
    *,
    max_request_bytes: int = MAX_REQUEST_BYTES,
    max_images: int = MAX_IMAGES_PER_REQUEST,
) -> Any:
    image_urls = _prepare_image_urls(
        merged_prompt,
        image_paths,
        max_request_bytes=max_request_bytes,
        max_images=max_images,
    )
    return _chat_user_content_from_urls(merged_prompt, image_urls)


def _responses_input(
    merged_prompt: str,
    image_paths: Iterable[Path],
    *,
    max_request_bytes: int = MAX_REQUEST_BYTES,
    max_images: int = MAX_IMAGES_PER_REQUEST,
) -> Any:
    image_urls = _prepare_image_urls(
        merged_prompt,
        image_paths,
        max_request_bytes=max_request_bytes,
        max_images=max_images,
    )
    return _responses_input_from_urls(merged_prompt, image_urls)


def _post_headers(api_key: str, accept: str) -> dict[str, str]:
    return {
        "Accept": accept,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "scionbench-judge/1.0",
    }


def _validate_redirect(source_url: str, target_url: str) -> None:
    source = urlparse(source_url)
    target = urlparse(target_url)
    if source.scheme != "https" or target.scheme != "https":
        raise ScriptError(
            f"refusing unsafe POST redirect from {source_url} to {target_url}"
        )
    if not source.hostname or not target.hostname:
        raise ScriptError(f"refusing malformed POST redirect target: {target_url}")
    if source.username or source.password or target.username or target.password:
        raise ScriptError("refusing POST redirect URLs containing user credentials")
    try:
        source_port = source.port or 443
        target_port = target.port or 443
    except ValueError as exc:
        raise ScriptError(f"invalid redirect URL: {target_url}") from exc
    if source.hostname != target.hostname or source_port != target_port:
        raise ScriptError(
            f"refusing cross-host POST redirect from {source_url} to {target_url}"
        )


def _open_post(
    *,
    url: str,
    body: bytes,
    headers: dict[str, str],
    timeout_seconds: int,
) -> Any:
    """Open a POST while preserving body and authorization across safe redirects."""

    current_url = url
    visited = {url}
    for _ in range(MAX_REDIRECTS + 1):
        request = Request(
            current_url,
            data=body,
            headers=dict(headers),
            method="POST",
        )
        try:
            return urlopen(request, timeout=timeout_seconds)
        except HTTPError as exc:
            if exc.code not in REDIRECT_STATUS_CODES:
                raise
            location = exc.headers.get("Location") if exc.headers else None
            if not location:
                exc.close()
                raise RetryableAPIError(
                    f"API returned HTTP {exc.code} without a Location header"
                ) from exc
            target_url = urljoin(current_url, location)
            _validate_redirect(current_url, target_url)
            if target_url in visited:
                exc.close()
                raise RetryableAPIError(
                    f"API redirect loop involving {target_url}"
                ) from exc
            visited.add(target_url)
            print(
                f"warning: following safe POST redirect HTTP {exc.code} "
                f"to {target_url}",
                file=sys.stderr,
            )
            exc.close()
            current_url = target_url
    raise RetryableAPIError(
        f"API exceeded the maximum number of POST redirects ({MAX_REDIRECTS})"
    )


def _retry_sleep(delay_seconds: float, attempt: int) -> None:
    if delay_seconds > 0:
        time.sleep(delay_seconds * (attempt + 1))


def _http_error_body(exc: HTTPError) -> str:
    """Read an error body without masking the original HTTP status."""

    try:
        return exc.read().decode("utf-8", errors="replace")
    except (OSError, IncompleteRead, UnicodeDecodeError):
        return ""


def _request_json(
    *,
    url: str,
    payload: dict[str, Any],
    api_key: str,
    timeout_seconds: int,
    max_retries: int,
    retry_delay_seconds: float,
    max_request_bytes: int = MAX_REQUEST_BYTES,
) -> dict[str, Any]:
    """POST JSON with bounded retries for transport and protocol failures."""

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if len(body) > max_request_bytes:
        raise ScriptError(
            f"API request body is {len(body)} bytes, above the limit "
            f"of {max_request_bytes} bytes"
        )
    headers = _post_headers(api_key, "application/json")
    attempts = max(1, max_retries + 1)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with _open_post(
                url=url,
                body=body,
                headers=headers,
                timeout_seconds=timeout_seconds,
            ) as response:
                raw_response = response.read()
            try:
                parsed = json.loads(raw_response.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RetryableAPIError(
                    f"API returned a non-JSON response from {url}: {exc}"
                ) from exc
            if not isinstance(parsed, dict):
                raise RetryableAPIError("API response JSON must be an object")
            api_error = _find_api_error(parsed)
            if api_error is not None:
                raise RetryableAPIError(f"API returned an error: {api_error}")
            if not _text_from_content_parts(parsed):
                raise RetryableAPIError(
                    "API JSON response did not contain model output text"
                )
            return parsed
        except HTTPError as exc:
            error_body = _http_error_body(exc)
            should_retry = (
                exc.code in RETRYABLE_HTTP_STATUS_CODES or exc.code >= 500
            )
            error = ScriptError(
                f"API request failed with HTTP {exc.code}: {error_body[:2000]}"
            )
            if not should_retry or attempt == attempts - 1:
                raise error from exc
            last_error = error
        except RetryableAPIError as exc:
            if attempt == attempts - 1:
                raise ScriptError(str(exc)) from exc
            last_error = exc
        except (
            URLError,
            TimeoutError,
            socket.timeout,
            RemoteDisconnected,
            IncompleteRead,
            BadStatusLine,
            HTTPException,
            ssl.SSLError,
            ConnectionResetError,
            BrokenPipeError,
            ConnectionError,
            OSError,
            UnicodeDecodeError,
        ) as exc:
            if attempt == attempts - 1:
                raise ScriptError(f"API request failed: {exc}") from exc
            last_error = exc
        _retry_sleep(retry_delay_seconds, attempt)

    raise ScriptError(f"API request failed after retries: {last_error}")


def _text_from_content_parts(parts: Any) -> str:
    if isinstance(parts, str):
        return parts
    if isinstance(parts, dict):
        chunks: list[str] = []
        # Gateways use both OpenAI names and a handful of common aliases for
        # generated text.  Preserve field order so streamed reasoning/content
        # is deterministic when both are present.
        for key in (
            "text",
            "content",
            "value",
            "output_text",
            "answer",
            "generated_text",
            "completion",
            "reasoning_content",
            "reasoning",
        ):
            if key in parts:
                text = _text_from_content_parts(parts[key])
                if text:
                    chunks.append(text)
        if chunks:
            return "".join(chunks)
        # Nonstandard wrappers often put the actual message under one of
        # these containers (for example {"data": {"answer": "..."}}).
        for key in ("message", "delta", "output", "data", "result", "response", "choices"):
            if key in parts:
                text = _text_from_content_parts(parts[key])
                if text:
                    chunks.append(text)
        return "".join(chunks)
    if not isinstance(parts, list):
        return ""

    chunks: list[str] = []
    for part in parts:
        if isinstance(part, str):
            chunks.append(part)
            continue
        if not isinstance(part, dict):
            continue
        text = _text_from_content_parts(part)
        if text:
            chunks.append(text)
    return "".join(chunks)


def _extract_responses_text(response: dict[str, Any]) -> str:
    # Some compatible gateways expose the SDK convenience field directly.
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        return output_text

    chunks: list[str] = []
    for key in (
        "text",
        "content",
        "output_text",
        "answer",
        "generated_text",
        "completion",
        "reasoning_content",
        "reasoning",
        "response",
        "result",
        "data",
    ):
        value = response.get(key)
        text = _text_from_content_parts(value)
        if text:
            chunks.append(text)
    output = response.get("output")
    if isinstance(output, str):
        chunks.append(output)
    if isinstance(output, dict):
        output = [output]
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            text = _text_from_content_parts(item.get("content"))
            if not text:
                text = _text_from_content_parts(item.get("text"))
            if not text:
                text = _text_from_content_parts(item)
            if text:
                chunks.append(text)
    if isinstance(response.get("choices"), (list, dict)):
        chunks.append(_extract_chat_completion_text(response))
    return "".join(chunks)


def _extract_chat_completion_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if isinstance(choices, dict):
        choices = [choices]
    chunks: list[str] = []
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                for key in (
                    "content",
                    "text",
                    "output_text",
                    "generated_text",
                    "completion",
                    "reasoning_content",
                    "reasoning",
                    "answer",
                ):
                    text = _text_from_content_parts(message.get(key))
                    if text:
                        chunks.append(text)
            for key in (
                "text",
                "content",
                "output_text",
                "generated_text",
                "completion",
                "reasoning_content",
                "reasoning",
                "answer",
            ):
                text = _text_from_content_parts(choice.get(key))
                if text:
                    chunks.append(text)
            delta = choice.get("delta")
            if isinstance(delta, dict):
                for key in (
                    "content",
                    "text",
                    "output_text",
                    "generated_text",
                    "completion",
                    "reasoning_content",
                    "reasoning",
                    "answer",
                ):
                    text = _text_from_content_parts(delta.get(key))
                    if text:
                        chunks.append(text)
    if chunks:
        return "".join(chunks)
    for key in (
        "text",
        "content",
        "output_text",
        "output",
        "answer",
        "generated_text",
        "completion",
        "reasoning_content",
        "reasoning",
        "response",
        "result",
        "data",
        "message",
        "delta",
    ):
        text = _text_from_content_parts(response.get(key))
        if text:
            chunks.append(text)
    return "".join(chunks)


def _extract_chat_stream_delta_text(event: dict[str, Any]) -> str:
    """Extract assistant text from one OpenAI-compatible SSE event."""

    error = _find_api_error(event)
    if error is not None:
        raise RetryableAPIError(f"API returned an error: {error}")

    chunks: list[str] = []
    choices = event.get("choices")
    if isinstance(choices, dict):
        choices = [choices]
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            for container in (choice.get("delta"), choice.get("message"), choice):
                if not isinstance(container, dict):
                    continue
                for key in (
                    "content",
                    "text",
                    "output_text",
                    "generated_text",
                    "completion",
                    "reasoning_content",
                    "reasoning",
                ):
                    text = _text_from_content_parts(container.get(key))
                    if text:
                        chunks.append(text)
    for key in (
        "content",
        "text",
        "output_text",
        "generated_text",
        "completion",
        "reasoning_content",
        "reasoning",
        "answer",
        "message",
        "delta",
    ):
        text = _text_from_content_parts(event.get(key))
        if text:
            chunks.append(text)
    if not chunks:
        for key in ("data", "result", "response", "output"):
            text = _text_from_content_parts(event.get(key))
            if text:
                chunks.append(text)
    return "".join(chunks)


def _iter_sse_data(decoded: str) -> Iterable[str]:
    """Yield complete SSE data payloads, including multiline events."""

    data_lines: list[str] = []
    for raw_line in decoded.splitlines():
        line = raw_line.rstrip("\r")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        elif line.startswith(("{", "[")):
            # A few gateways label the response as streaming but omit the
            # ``data:`` SSE prefix.  Treat each complete JSON line as an event.
            data_lines.append(line)
    if data_lines:
        yield "\n".join(data_lines)


def _request_stream_text(
    *,
    url: str,
    payload: dict[str, Any],
    api_key: str,
    timeout_seconds: int,
    max_retries: int,
    retry_delay_seconds: float,
    max_request_bytes: int = MAX_REQUEST_BYTES,
) -> str:
    """POST JSON and collect text from an OpenAI-compatible SSE response."""

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if len(body) > max_request_bytes:
        raise ScriptError(
            f"API request body is {len(body)} bytes, above the limit "
            f"of {max_request_bytes} bytes"
        )
    headers = _post_headers(api_key, "text/event-stream")
    attempts = max(1, max_retries + 1)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with _open_post(
                url=url,
                body=body,
                headers=headers,
                timeout_seconds=timeout_seconds,
            ) as response:
                raw_response = response.read()
            decoded = raw_response.decode("utf-8")
            saw_sse_event = False
            chunks: list[str] = []
            for data in _iter_sse_data(decoded):
                saw_sse_event = True
                if not data.strip():
                    continue
                if data.strip() == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise RetryableAPIError(
                        f"API returned an invalid streaming event from {url}: {exc}"
                    ) from exc
                if not isinstance(event, dict):
                    continue
                text = _extract_chat_stream_delta_text(event)
                if text:
                    chunks.append(text)

            text = "".join(chunks)
            if text:
                return text

            # Some compatible gateways ignore stream=true and return one JSON
            # completion.  Accept it, including nonstandard text fields.
            if not saw_sse_event:
                try:
                    fallback = json.loads(decoded)
                except json.JSONDecodeError as exc:
                    raise RetryableAPIError(
                        f"API returned neither SSE nor JSON from {url}: {exc}"
                    ) from exc
                if isinstance(fallback, dict):
                    error = _find_api_error(fallback)
                    if error is not None:
                        raise RetryableAPIError(f"API returned an error: {error}")
                    text = _extract_chat_completion_text(fallback)
                    if text:
                        return text
            raise RetryableAPIError(
                "API streaming response did not contain model output text"
            )
        except HTTPError as exc:
            error_body = _http_error_body(exc)
            should_retry = (
                exc.code in RETRYABLE_HTTP_STATUS_CODES or exc.code >= 500
            )
            error = ScriptError(
                f"API request failed with HTTP {exc.code}: {error_body[:2000]}"
            )
            if not should_retry or attempt == attempts - 1:
                raise error from exc
            last_error = error
        except RetryableAPIError as exc:
            if attempt == attempts - 1:
                raise ScriptError(str(exc)) from exc
            last_error = exc
        except (
            URLError,
            TimeoutError,
            socket.timeout,
            RemoteDisconnected,
            IncompleteRead,
            BadStatusLine,
            HTTPException,
            ssl.SSLError,
            ConnectionResetError,
            BrokenPipeError,
            ConnectionError,
            OSError,
            UnicodeDecodeError,
        ) as exc:
            if attempt == attempts - 1:
                raise ScriptError(f"API request failed: {exc}") from exc
            last_error = exc
        _retry_sleep(retry_delay_seconds, attempt)

    raise ScriptError(f"API request failed after retries: {last_error}")


def call_model(
    merged_prompt: str,
    *,
    model: str,
    api_mode: str,
    api_base_url: str,
    api_key: str,
    timeout_seconds: int,
    max_retries: int,
    retry_delay_seconds: float,
    thinking: bool,
    temperature: float,
    top_p: float | None,
    max_tokens: int | None,
    stream: bool = False,
    image_paths: Iterable[Path] = (),
    max_request_bytes: int = MAX_REQUEST_BYTES,
    max_images: int = MAX_IMAGES_PER_REQUEST,
) -> str:
    """Send *merged_prompt* and optional figure images, returning model text."""

    image_paths = tuple(image_paths)
    image_urls = _prepare_image_urls(
        merged_prompt,
        image_paths,
        max_request_bytes=max_request_bytes,
        max_images=max_images,
    )

    response: dict[str, Any] = {}

    if api_mode == "responses":
        if stream:
            raise ScriptError("stream=true is only supported for chat_completions")
        endpoint = f"{api_base_url}/responses"
        payload = {
            "model": model,
            "input": _responses_input_from_urls(merged_prompt, image_urls),
        }
        # Keep the AIGW request compatible with the supplied curl contract:
        # Responses requests contain only ``model`` and ``input`` by default.
        # The temperature/top-p/max-token options remain available for the
        # legacy chat-completions branch below.
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens
        if thinking:
            payload["chat_template_kwargs"] = {"thinking": thinking}
        _trim_payload_images(payload, max_request_bytes)
        response = _request_json(
            url=endpoint,
            payload=payload,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
            max_request_bytes=max_request_bytes,
        )
        text = _extract_responses_text(response)
    elif api_mode == "chat_completions":
        endpoint = f"{api_base_url}/chat/completions"
        payload = {
            "model": model,
            "temperature": temperature,
            "stream": stream,
            "messages": [
                {
                    "role": "user",
                    "content": _chat_user_content_from_urls(merged_prompt, image_urls),
                }
            ],
        }
        if thinking:
            payload["chat_template_kwargs"] = {"thinking": thinking}
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        _trim_payload_images(payload, max_request_bytes)
        if stream:
            text = _request_stream_text(
                url=endpoint,
                payload=payload,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds,
                max_request_bytes=max_request_bytes,
            )
        else:
            response = _request_json(
                url=endpoint,
                payload=payload,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds,
                max_request_bytes=max_request_bytes,
            )
            text = _extract_chat_completion_text(response)
    else:
        raise ScriptError(
            f"unsupported API_MODE {api_mode!r}; use 'responses' or "
            "'chat_completions'"
        )

    if not text:
        error = _find_api_error(response)
        if error is not None:
            raise ScriptError(f"API returned an error: {error}")
        raise ScriptError("API response did not contain model output text")
    return text


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace GOLDEN_NODES and AGENT_OUTPUT_MD in a Markdown prompt, "
            "call the configured OpenAI-compatible judge model, and save the raw model text."
        )
    )
    parser.add_argument("--golden-nodes", help="Path to the standard-answer Markdown.")
    parser.add_argument("--agent-output-md", help="Path to the agent-output Markdown.")
    parser.add_argument("--prompt-md", help="Path to the prompt template Markdown.")
    parser.add_argument(
        "--merged-md-output",
        help="Path where the merged prompt Markdown will be written.",
    )
    parser.add_argument(
        "--model-output-md",
        help="Path where the model output will be written verbatim.",
    )
    parser.add_argument(
        "--image-path",
        action="append",
        default=[],
        help="Local figure image to attach; repeat for multiple images.",
    )
    parser.add_argument("--model", default=MODEL, help=f"Model ID (default: {MODEL}).")
    parser.add_argument(
        "--api-mode",
        choices=("responses", "chat_completions"),
        default=API_MODE,
        help=f"API endpoint style (default: {API_MODE}).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible API base URL (overrides API_BASE_URL).",
    )
    parser.add_argument(
        "--api-key-env",
        default=API_KEY_ENV,
        help=f"Environment variable containing the API key (default: {API_KEY_ENV}).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=REQUEST_TIMEOUT_SECONDS,
        help=f"HTTP timeout in seconds (default: {REQUEST_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES,
        help=f"Retries for transient API failures (default: {MAX_RETRIES}).",
    )
    parser.add_argument(
        "--max-request-bytes",
        type=int,
        default=MAX_REQUEST_BYTES,
        help=(
            "Maximum serialized JSON request size in bytes "
            f"(default: {MAX_REQUEST_BYTES})."
        ),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=MAX_IMAGES_PER_REQUEST,
        help=(
            "Maximum images attached to one request "
            f"(default: {MAX_IMAGES_PER_REQUEST})."
        ),
    )
    parser.add_argument(
        "--thinking",
        choices=("true", "false"),
        default="true" if THINKING else "false",
        help=f"Enable gateway thinking mode (default: {'true' if THINKING else 'false'}).",
    )
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument(
        "--stream",
        choices=("true", "false"),
        default="true" if STREAM else "false",
        help=f"Request an SSE response (default: {'true' if STREAM else 'false'}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        if args.max_retries < 0:
            raise ScriptError("max retries cannot be negative")
        if args.max_request_bytes <= 0:
            raise ScriptError("max request bytes must be greater than zero")
        if args.max_images < 0:
            raise ScriptError("max images cannot be negative")
        golden_nodes_path = _resolve_path(args.golden_nodes, GOLDEN_NODES, "GOLDEN_NODES")
        agent_output_path = _resolve_path(
            args.agent_output_md, AGENT_OUTPUT_MD, "AGENT_OUTPUT_MD"
        )
        prompt_path = _resolve_path(args.prompt_md, PROMPT_MD, "PROMPT_MD")
        merged_output_path = _resolve_path(
            args.merged_md_output, MERGED_MD_OUTPUT, "MERGED_MD_OUTPUT"
        )
        model_output_path = _resolve_path(
            args.model_output_md, MODEL_OUTPUT_MD, "MODEL_OUTPUT_MD"
        )

        if merged_output_path == model_output_path:
            raise ScriptError(
                "MERGED_MD_OUTPUT and MODEL_OUTPUT_MD must be different files"
            )

        merged_prompt = merge_markdown(
            golden_nodes_path, agent_output_path, prompt_path
        )
        _write_text(merged_output_path, merged_prompt, "merged Markdown")
        image_paths = _resolve_image_paths(args.image_path)

        api_key = _resolve_api_key(API_KEY, args.api_key_env)
        api_base_url = _resolve_api_base_url(args.base_url or API_BASE_URL)
        model_output = call_model(
            merged_prompt,
            model=args.model,
            api_mode=args.api_mode,
            api_base_url=api_base_url,
            api_key=api_key,
            timeout_seconds=args.timeout,
            max_retries=args.max_retries,
            retry_delay_seconds=RETRY_DELAY_SECONDS,
            thinking=args.thinking == "true",
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            stream=args.stream == "true",
            image_paths=image_paths,
            max_request_bytes=args.max_request_bytes,
            max_images=args.max_images,
        )
        _write_text(model_output_path, model_output, "model output")
    except ScriptError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"merged Markdown: {merged_output_path}")
    print(f"attached images: {len(image_paths)}")
    print(f"model output: {model_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
