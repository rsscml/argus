"""Tests for the Azure auth seam (llm/factory.py, SS12.4).

No network and no real azure-identity: the Entra path is exercised against a
faked ``azure.identity`` module, asserting exactly what the factory wires
into the LangChain constructors.
"""
from __future__ import annotations

import sys
import types

import pytest

from argus import envfile
from argus.llm import factory

AZ = {
    "AZURE_OPENAI_ENDPOINT": "https://unit.openai.azure.com",
    "AZURE_OPENAI_API_VERSION": "2024-10-21",
    "ARGUS_AZURE_SYNTHESIS_DEPLOYMENT": "gpt-strong",
    "ARGUS_AZURE_UTILITY_DEPLOYMENT": "gpt-mini",
    "ARGUS_AZURE_EMBEDDING_DEPLOYMENT": "embed-3",
}


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no stray .env
    envfile.reset_for_tests()
    for key in (*AZ, "AZURE_OPENAI_API_KEY", "ARGUS_AZURE_AUTH"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(factory, "_token_provider", None)
    yield
    envfile.reset_for_tests()


def test_auth_mode_validation_and_catalog(monkeypatch):
    assert factory.auth_mode() == "api_key"  # unset -> default
    monkeypatch.setenv("ARGUS_AZURE_AUTH", " Default_Credential ")
    assert factory.auth_mode() == "default_credential"  # trimmed, case-folded
    monkeypatch.setenv("ARGUS_AZURE_AUTH", "magic")
    with pytest.raises(factory.AzureConfigError, match="api_key.*default_credential"):
        factory.auth_mode()
    # the switch is exposed on the console's settings page
    from argus.webapp.config_store import SETTING_KEYS
    assert "ARGUS_AZURE_AUTH" in SETTING_KEYS


def test_api_key_mode_fails_fast_then_constructs(monkeypatch):
    for k, v in AZ.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(factory.AzureConfigError, match="AZURE_OPENAI_API_KEY"):
        factory.get_chat_model("utility")  # at construction, not first call
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k-123")
    model = factory.get_chat_model("synthesis")
    assert model.deployment_name == "gpt-strong"
    assert getattr(model, "azure_ad_token_provider", None) is None
    assert factory.model_fingerprint("synthesis")["auth"] == "api_key"


def test_default_credential_wires_provider_for_chat_and_embeddings(monkeypatch):
    calls = {"cred": 0, "scope": None}
    provider = lambda: "tok"  # noqa: E731

    fake = types.ModuleType("azure.identity")
    fake.DefaultAzureCredential = lambda: calls.__setitem__("cred", calls["cred"] + 1)
    def _gbtp(credential, scope):
        calls["scope"] = scope
        return provider
    fake.get_bearer_token_provider = _gbtp
    monkeypatch.setitem(sys.modules, "azure", types.ModuleType("azure"))
    monkeypatch.setitem(sys.modules, "azure.identity", fake)

    for k, v in AZ.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("ARGUS_AZURE_AUTH", "default_credential")
    # no AZURE_OPENAI_API_KEY anywhere

    chat = factory.get_chat_model("utility")
    emb = factory.get_azure_embeddings()
    assert chat.azure_ad_token_provider is provider
    assert emb.azure_ad_token_provider is provider
    assert calls["cred"] == 1  # one credential per process, shared by both
    assert calls["scope"] == "https://cognitiveservices.azure.com/.default"
    assert factory.model_fingerprint()["auth"] == "default_credential"


def test_missing_azure_identity_points_at_the_extra(monkeypatch):
    for k, v in AZ.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("ARGUS_AZURE_AUTH", "default_credential")
    monkeypatch.setitem(sys.modules, "azure.identity", None)  # forces ImportError
    with pytest.raises(factory.AzureConfigError, match=r"\[entra\]"):
        factory.get_chat_model("utility")
