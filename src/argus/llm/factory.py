"""Azure OpenAI model factory (architecture AD-10, SS10).

The ONLY place model objects are constructed. Everything goes through
LangChain's Azure OpenAI integration (`langchain-openai`). Pinning discipline:
callers record deployment name + api_version + auth mode (exposed by
`model_fingerprint`) into run state and manifests (SS12.1).

Authentication (SS12.4) is a switch, not a fork — ARGUS_AZURE_AUTH:
    api_key            classic key via AZURE_OPENAI_API_KEY (default;
                       required up front, so a missing key fails at
                       construction, not on the first API call)
    default_credential Entra ID via azure.identity.DefaultAzureCredential —
                       managed identity, workload identity, `az login`,
                       environment credentials, … No key is read or stored.
                       Needs the `entra` extra (pip install -e ".[entra]").

Required environment (both modes):
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION,
    ARGUS_AZURE_EMBEDDING_DEPLOYMENT,
    ARGUS_AZURE_SYNTHESIS_DEPLOYMENT, ARGUS_AZURE_UTILITY_DEPLOYMENT (M3+)

Dev/tests may bypass Azure entirely with ARGUS_LLM=stub / ARGUS_EMBEDDER=hashing.
"""
from __future__ import annotations

import os
import threading

from argus.envfile import load_process_env

_ROLE_ENV = {
    # strong deployment: synthesis, planning (SS10)
    "synthesis": "ARGUS_AZURE_SYNTHESIS_DEPLOYMENT",
    # cheaper deployment: claim pairing, citation verification (SS10)
    "utility": "ARGUS_AZURE_UTILITY_DEPLOYMENT",
}

_AUTH_ENV = "ARGUS_AZURE_AUTH"
_AUTH_MODES = ("api_key", "default_credential")
_AAD_SCOPE = "https://cognitiveservices.azure.com/.default"

_token_provider = None  # one credential per process; the provider refreshes tokens
_token_lock = threading.Lock()


class AzureConfigError(RuntimeError):
    pass


def _require(var: str) -> str:
    value = os.environ.get(var, "")
    if not value:
        raise AzureConfigError(
            f"missing env var {var}; Azure OpenAI is the mandated LLM stack (AD-10). "
            f"For local dev without Azure, set ARGUS_LLM=stub ARGUS_EMBEDDER=hashing."
        )
    return value


def auth_mode() -> str:
    """Validated ARGUS_AZURE_AUTH, defaulting to api_key."""
    mode = (os.environ.get(_AUTH_ENV, "") or "api_key").strip().lower()
    if mode not in _AUTH_MODES:
        raise AzureConfigError(
            f"{_AUTH_ENV}={mode!r} is not valid — expected one of {list(_AUTH_MODES)}."
        )
    return mode


def _auth_kwargs() -> dict:
    """Credential kwargs shared by the chat and embeddings constructors."""
    if auth_mode() == "api_key":
        return {"api_key": _require("AZURE_OPENAI_API_KEY")}
    global _token_provider
    with _token_lock:
        if _token_provider is None:
            try:
                from azure.identity import (DefaultAzureCredential,
                                            get_bearer_token_provider)
            except ImportError as exc:
                raise AzureConfigError(
                    f"{_AUTH_ENV}=default_credential requires the azure-identity "
                    'package: pip install -e ".[entra]" (or pip install '
                    "azure-identity)."
                ) from exc
            _token_provider = get_bearer_token_provider(
                DefaultAzureCredential(), _AAD_SCOPE
            )
    return {"azure_ad_token_provider": _token_provider}


def get_chat_model(role: str = "utility", *, temperature: float = 0.0):
    """AzureChatOpenAI for a role. Temperature defaults to 0 (SS8.1)."""
    load_process_env()  # .env + console overrides feed the AZURE_* reads below
    from langchain_openai import AzureChatOpenAI

    deployment = _require(_ROLE_ENV[role])
    return AzureChatOpenAI(
        azure_endpoint=_require("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=deployment,
        api_version=_require("AZURE_OPENAI_API_VERSION"),
        temperature=temperature,
        **_auth_kwargs(),
    )


def get_azure_embeddings():
    load_process_env()
    from langchain_openai import AzureOpenAIEmbeddings

    return AzureOpenAIEmbeddings(
        azure_endpoint=_require("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=_require("ARGUS_AZURE_EMBEDDING_DEPLOYMENT"),
        api_version=_require("AZURE_OPENAI_API_VERSION"),
        **_auth_kwargs(),
    )


def model_fingerprint(role: str | None = None) -> dict[str, str]:
    """Deployment/api-version/auth identifiers for run manifests (SS12.1).
    Never raises — manifests are written even for misconfigured runs."""
    fp = {"api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "")}
    if role:
        fp["deployment"] = os.environ.get(_ROLE_ENV[role], "")
    fp["embedding_deployment"] = os.environ.get("ARGUS_AZURE_EMBEDDING_DEPLOYMENT", "")
    fp["auth"] = (os.environ.get(_AUTH_ENV, "") or "api_key").strip().lower()
    return fp
