"""Exclusive ARK chat-completions provider boundary for fixed paper V1."""

from __future__ import annotations

import copy
import stat
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from hermes_runtime.redaction import redact_value

from nurse_rcl.agents.retry import RetryableParsingError

ARK_PROVIDER = "ark_openai_compatible"
ARK_ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3"
ARK_MODEL = "deepseek-v4-flash-ga-260731"
ARK_API_KEY_VARIABLE = "ARK_API_KEY2"
ARK_KEY_FILE_RELATIVE = Path(".config") / "nurse" / "ark.env"
ARK_CHAT_COMPLETIONS_URL = f"{ARK_ENDPOINT}/chat/completions"
ARK_REQUEST_TIMEOUT_SECONDS = 180.0


class ArkConfigurationError(ValueError):
    """Raised when a caller attempts to violate the fixed ARK boundary."""


class MissingArkApiKeyError(ArkConfigurationError):
    """Raised when ARK_API_KEY2 is absent or empty."""


class InsecureArkKeyFileError(ArkConfigurationError):
    """Raised when the server key file is not mode 0600."""


class ArkKeyFileFormatError(ArkConfigurationError):
    """Raised when the private key file has an unauthorized assignment."""


class ArkProviderResponseError(RetryableParsingError):
    """Raised when ARK returns a non-object JSON response."""


class ArkResponseResourceError(RuntimeError):
    """A bounded response reader stopped before allocating an oversized body."""


def _bounded_response(stream: httpx.Response, limit: int) -> httpx.Response:
    """Bound expanded bytes before decompressor allocation, then decode only once."""
    coding = stream.headers.get("content-encoding", "identity").strip().lower()
    if coding not in {"identity", "gzip", "deflate"}:
        raise ArkResponseResourceError("unsupported_response_encoding")
    decoder = None if coding == "identity" else zlib.decompressobj(
        zlib.MAX_WBITS + 16 if coding == "gzip" else zlib.MAX_WBITS)
    body = bytearray()
    compressed_bytes = 0
    # Injected clients may hand back an already-buffered Response. Real HTTP
    # uses iter_raw, never httpx's unbounded per-chunk content decoder.
    chunks = (stream.content,) if stream.is_stream_consumed else stream.iter_raw()
    if stream.is_stream_consumed:
        decoder = None  # httpx already decoded this synthetic/buffered object.
    for chunk in chunks:
        compressed_bytes += len(chunk)
        if compressed_bytes > limit:
            raise ArkResponseResourceError("response_resource_limit")
        expanded = chunk if decoder is None else decoder.decompress(chunk, limit - len(body) + 1)
        if len(body) + len(expanded) > limit:
            raise ArkResponseResourceError("response_resource_limit")
        body.extend(expanded)
        if decoder is not None and (decoder.unconsumed_tail or decoder.unused_data):
            raise ArkResponseResourceError("response_resource_limit")
    if decoder is not None and not decoder.eof:
        raise ArkProviderResponseError("incomplete HTTP content encoding")
    headers = {k: v for k, v in stream.headers.items()
               if k.lower() not in {"content-encoding", "content-length"}}
    return httpx.Response(stream.status_code, headers=headers,
                          content=bytes(body), request=stream.request)


class ArkCredential:
    """Opaque in-memory credential that reveals its value only to transport code."""

    __slots__ = ("__value",)

    def __init__(self, value: str) -> None:
        normalized = value.strip()
        if not normalized:
            raise MissingArkApiKeyError(f"{ARK_API_KEY_VARIABLE} must be non-empty")
        self.__value = normalized

    def reveal_for_transport(self) -> str:
        """Return the credential solely for construction of the HTTP Authorization header."""

        return self.__value

    def __repr__(self) -> str:
        return "ArkCredential(***)"


@dataclass(frozen=True, slots=True)
class ArkSettings:
    """Fixed provider coordinates plus an opaque explicitly supplied credential."""

    credential: ArkCredential = field(repr=False)

    @property
    def provider(self) -> str:
        return ARK_PROVIDER

    @property
    def endpoint(self) -> str:
        return ARK_ENDPOINT

    @property
    def model(self) -> str:
        return ARK_MODEL

    @property
    def provider_fallback(self) -> bool:
        return False

    @property
    def model_fallback(self) -> bool:
        return False

    @property
    def request_timeout_seconds(self) -> float:
        return ARK_REQUEST_TIMEOUT_SECONDS

    def audit_record(self) -> dict[str, str | bool | float]:
        """Return the complete serializable configuration record without the credential."""

        return {
            "api_key_environment_variable": ARK_API_KEY_VARIABLE,
            "credential_configured": True,
            "endpoint": self.endpoint,
            "model": self.model,
            "model_fallback": self.model_fallback,
            "provider": self.provider,
            "provider_fallback": self.provider_fallback,
            "request_timeout_seconds": self.request_timeout_seconds,
        }

    def redact_artifact(self, value: Any) -> Any:
        """Recursively remove this credential from an artifact-shaped value."""

        return redact_value(
            value,
            secrets=(self.credential.reveal_for_transport(),),
        )


def load_ark_settings(environment: Mapping[str, str]) -> ArkSettings:
    """Load only ARK_API_KEY2 from an explicitly supplied environment mapping."""

    try:
        raw_key = environment[ARK_API_KEY_VARIABLE]
    except KeyError as error:
        raise MissingArkApiKeyError(
            f"required environment variable {ARK_API_KEY_VARIABLE} is missing"
        ) from error
    return ArkSettings(credential=ArkCredential(raw_key))


def validate_private_key_file_mode(mode: int) -> None:
    """Require the exact POSIX permission bits 0600."""

    if stat.S_IMODE(mode) != 0o600:
        raise InsecureArkKeyFileError("ARK key file must have POSIX mode 0600")


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def parse_ark_key_file_text(text: str) -> ArkCredential:
    """Parse a minimal env file containing exactly one ARK_API_KEY2 assignment."""

    raw_key: str | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped.removeprefix("export ").lstrip()
        if "=" not in stripped:
            raise ArkKeyFileFormatError(
                f"line {line_number} is not an {ARK_API_KEY_VARIABLE} assignment"
            )
        name, value = (part.strip() for part in stripped.split("=", 1))
        if name != ARK_API_KEY_VARIABLE:
            raise ArkKeyFileFormatError(
                f"unauthorized key file variable {name}; only {ARK_API_KEY_VARIABLE} is allowed"
            )
        if raw_key is not None:
            raise ArkKeyFileFormatError(f"duplicate {ARK_API_KEY_VARIABLE} assignment")
        raw_key = _unquote(value)
    if raw_key is None:
        raise MissingArkApiKeyError(f"key file does not define {ARK_API_KEY_VARIABLE}")
    return ArkCredential(raw_key)


def default_ark_key_file(*, home: Path | None = None) -> Path:
    """Return ~/.config/nurse/ark.env without reading HOME-like environment variables."""

    base = home if home is not None else Path.home()
    return base / ARK_KEY_FILE_RELATIVE


def load_ark_settings_from_key_file(path: Path) -> ArkSettings:
    """Load fixed settings from a mode-0600 server key file."""

    key_path = path.expanduser()
    observed_mode = key_path.stat().st_mode
    validate_private_key_file_mode(observed_mode)
    credential = parse_ark_key_file_text(key_path.read_text(encoding="utf-8"))
    return ArkSettings(credential=credential)


def load_server_ark_settings(*, home: Path | None = None) -> ArkSettings:
    """Load the sole supported server credential path."""

    return load_ark_settings_from_key_file(default_ark_key_file(home=home))


class ArkChatCompletionsClient:
    """Explicit synchronous client for one fixed ARK chat-completions model."""

    def __init__(
        self,
        settings: ArkSettings,
        *,
        transport: httpx.BaseTransport | None = None,
        response_limit_bytes: int | None = None,
    ) -> None:
        self._settings = settings
        if response_limit_bytes is not None and (
            type(response_limit_bytes) is not int or not 1 <= response_limit_bytes <= 4_194_304
        ):
            raise ValueError("invalid response resource envelope")
        self._response_limit_bytes = response_limit_bytes
        self._client = httpx.Client(
            timeout=ARK_REQUEST_TIMEOUT_SECONDS,
            transport=transport,
        )

    def complete(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """POST one fixed-model chat-completions request without fallback."""

        payload = copy.deepcopy(dict(request))
        if payload.get("model") != self._settings.model:
            raise ArkConfigurationError(
                f"request model must equal fixed model {self._settings.model}"
            )
        forbidden = {"api_key", "base_url", "endpoint", "provider"}.intersection(payload)
        if forbidden:
            raise ArkConfigurationError(
                f"request contains forbidden provider override fields: {sorted(forbidden)}"
            )
        headers = {
                "Authorization": (
                    f"Bearer {self._settings.credential.reveal_for_transport()}"
                ),
                "Content-Type": "application/json",
            }
        if self._response_limit_bytes is None:
            response = self._client.post(ARK_CHAT_COMPLETIONS_URL, headers=headers, json=payload)
        else:
            # Incremental HTTP reading is independent of native token defaults.
            # Retain an error response body too: failures can carry real usage.
            headers["Accept-Encoding"] = "identity"
            with self._client.stream("POST", ARK_CHAT_COMPLETIONS_URL,
                                     headers=headers, json=payload) as stream:
                response = _bounded_response(stream, self._response_limit_bytes)
        response.raise_for_status()
        try:
            value = response.json()
        except ValueError as error:
            raise ArkProviderResponseError("ARK response body is not valid JSON") from error
        if not isinstance(value, dict):
            raise ArkProviderResponseError("ARK response JSON must be an object")
        return value

    def close(self) -> None:
        """Close the fixed-timeout client owned by this boundary."""

        self._client.close()

    def __enter__(self) -> ArkChatCompletionsClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
