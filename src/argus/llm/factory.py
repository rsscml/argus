"""Azure OpenAI model factory (architecture AD-10, SS10).

The ONLY place model objects are constructed. Everything goes through
LangChain's Azure OpenAI integration (`langchain-openai`). Pinning discipline:
callers record deployment name + api_version (exposed by `model_fingerprint`)
into run state and manifests (SS12.1).

Required environment:
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY (or Entra ID, SS12.4),
    AZURE_OPENAI_API_VERSION,
    ARGUS_AZURE_EMBEDDING_DEPLOYMENT,
    ARGUS_AZURE_SYNTHESIS_DEPLOYMENT, ARGUS_AZURE_UTILITY_DEPLOYMENT (M3+)

Dev/tests may bypass Azure with ARGUS_EMBEDDER=hashing (see argus.ingest.embed).
"""
from __future__ import annotations

import os

_ROLE_ENV = {
    # strong deployment: synthesis, planning (SS10)
    "synthesis": "ARGUS_AZURE_SYNTHESIS_DEPLOYMENT",
    # cheaper deployment: claim pairing, citation verification (SS10)
    "utility": "ARGUS_AZURE_UTILITY_DEPLOYMENT",
}


class AzureConfigError(RuntimeError):
    pass


def _require(var: str) -> str:
    value = os.environ.get(var, "")
    if not value:
        raise AzureConfigError(
            f"missing env var {var}; Azure OpenAI is the mandated LLM stack (AD-10). "
            f"For local dev without Azure, set ARGUS_EMBEDDER=hashing."
        )
    return value


def get_chat_model(role: str = "utility", *, temperature: float = 0.0):
    """AzureChatOpenAI for a role. Temperature defaults to 0 (SS8.1)."""
    from langchain_openai import AzureChatOpenAI

    deployment = _require(_ROLE_ENV[role])
    return AzureChatOpenAI(
        azure_endpoint=_require("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=deployment,
        api_version=_require("AZURE_OPENAI_API_VERSION"),
        temperature=temperature,
    )


def get_azure_embeddings():
    from langchain_openai import AzureOpenAIEmbeddings

    return AzureOpenAIEmbeddings(
        azure_endpoint=_require("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=_require("ARGUS_AZURE_EMBEDDING_DEPLOYMENT"),
        api_version=_require("AZURE_OPENAI_API_VERSION"),
    )


def model_fingerprint(role: str | None = None) -> dict[str, str]:
    """Deployment/api-version identifiers for run manifests (SS12.1)."""
    fp = {"api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "")}
    if role:
        fp["deployment"] = os.environ.get(_ROLE_ENV[role], "")
    fp["embedding_deployment"] = os.environ.get("ARGUS_AZURE_EMBEDDING_DEPLOYMENT", "")
    return fp
