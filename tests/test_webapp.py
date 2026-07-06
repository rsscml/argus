"""End-to-end tests for the web layer, run against a hermetic workspace.

Everything goes through the HTTP API with the stub/offline engine
(ARGUS_LLM=stub, ARGUS_EMBEDDER=hashing), so the suite needs no network and
no Azure credentials. Tests run in operator-flow order within this file and
share one workspace; the auth test mutates the password and therefore runs
last.

    pip install -e ".[webapp]" pytest httpx
    pytest tests/test_webapp.py -q
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def workspace(tmp_path_factory) -> Path:
    """A throwaway copy of the project's config + corpus, with its own git."""
    ws = tmp_path_factory.mktemp("argus-ws")
    for item in ("registry", "domains", "corpora", "tests"):
        src = REPO / item
        if src.exists():
            shutil.copytree(src, ws / item,
                            ignore=shutil.ignore_patterns("__pycache__", ".backups",
                                                          "test_*.py"))
    subprocess.run(["git", "init", "-q", str(ws)], check=True)
    subprocess.run(["git", "-C", str(ws), "-c", "user.email=t@t",
                    "-c", "user.name=t", "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(ws), "-c", "user.email=t@t",
                    "-c", "user.name=t", "commit", "-qm", "baseline"], check=True)
    return ws


@pytest.fixture(scope="session")
def client(workspace, tmp_path_factory):
    os.chdir(workspace)
    os.environ["ARGUS_LLM"] = "stub"
    os.environ["ARGUS_EMBEDDER"] = "hashing"
    os.environ.pop("ARGUS_WEB_PASSWORD", None)
    from fastapi.testclient import TestClient

    from argus.webapp.app import create_app

    with TestClient(create_app()) as c:  # context manager runs the lifespan
        yield c


def wait_job(client, job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.25)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


# ---------------------------------------------------------------- pages
def test_pages_and_overview(client):
    for page in ("/", "/admin", "/maintenance", "/static/app.css", "/static/app.js"):
        assert client.get(page).status_code == 200, page
    ov = client.get("/api/overview").json()
    assert ov["mode"] == {"llm": "stub", "embedder": "hashing"}
    assert {d["name"] for d in ov["domains"]} >= {"general_news", "commodities"}
    assert ov["registry"]["sources"] >= 6


# ---------------------------------------------------------------- runs
def test_brief_end_to_end(client):
    r = client.post("/api/actions/brief", json={"domain": "general_news"})
    assert r.status_code == 200
    job = wait_job(client, r.json()["job_id"])
    assert job["status"] == "done", job.get("error")
    run_id = job["result"]["run_id"]

    out = client.get(f"/api/runs/{run_id}/output").json()
    assert "Daily brief" in out["markdown"]
    assert '<sup class="cite">' in out["html"]  # rendered citation chips

    detail = client.get(f"/api/runs/{run_id}").json()
    manifest = detail["manifest"]
    assert manifest["workload"] == "daily_brief"
    assert len(manifest["snapshot_set"]) >= 2  # the demo corpus
    # single-voice corpus can never self-corroborate -> attribute fallback
    assert set(manifest["verdict_counts"]) == {"attributed"}

    # the shared queue refuses a duplicate brief for the same domain
    client.post("/api/actions/brief", json={"domain": "general_news"})
    dup = client.post("/api/actions/brief", json={"domain": "general_news"})
    assert dup.status_code == 409
    for j in client.get("/api/jobs?limit=5").json():
        if j["status"] in ("queued", "running"):
            wait_job(client, j["id"])


def test_health_snapshots_after_poll(client):
    h = client.get("/api/health?window_hours=24").json()
    by = {s["source_id"]: s for s in h["sources"]}
    assert by["local_demo"]["status"] == "ok"
    assert by["un_news"]["status"] in ("degraded", "failing")  # offline RSS

    snaps = client.get("/api/snapshots?limit=10&source=local_demo").json()
    assert len(snaps) >= 2
    detail = client.get(f"/api/snapshots/{snaps[0]['content_hash']}").json()
    assert detail["preview_status"] == "ok" and len(detail["preview"]) > 50


def test_research_and_replay(client):
    r = client.post("/api/actions/research", json={
        "domain": "general_news",
        "question": "What happened to tanker transits and insurance premiums?"})
    job = wait_job(client, r.json()["job_id"])
    assert job["status"] == "done", job.get("error")
    out = client.get(f"/api/runs/{job['result']['run_id']}/output").json()
    assert "Research report" in out["markdown"]

    brief = next(x for x in client.get("/api/runs?limit=20").json()
                 if x["workload"] == "daily_brief" and x["status"] == "done")
    job = wait_job(client, client.post(
        f"/api/actions/replay/{brief['run_id']}").json()["job_id"])
    assert job["status"] == "done", job.get("error")
    assert job["result"]["comparison"]["snapshot_set_identical"] is True


def test_golden(client):
    job = wait_job(client, client.post(
        "/api/actions/golden", json={"domain": "general_news"}).json()["job_id"])
    assert job["status"] == "done" and job["result"]["ok"] is True
    job = wait_job(client, client.post(
        "/api/actions/golden", json={"domain": "commodities"}).json()["job_id"])
    assert job["status"] == "failed" and "no golden set" in job["error"]


# ---------------------------------------------------------------- governance
def test_registry_crud_and_validation(client):
    r = client.post("/api/registry/sources/bbc_world/status",
                    json={"status": "paused"}).json()
    assert r["committed"] is True and r["commit"]
    reg = client.get("/api/registry").json()
    assert next(s for s in reg["sources"]
                if s["id"] == "bbc_world")["status"] == "paused"
    client.post("/api/registry/sources/bbc_world/status", json={"status": "active"})

    new = {"source": {
        "id": "test_feed", "name": "Test Feed", "language": "en",
        "fetch": {"adapter": "rss", "endpoints": ["https://example.org/f.xml"]},
        "license": "test license", "tags": {"general_news": {"tier": 3}},
        "status": "paused"}}
    assert client.post("/api/registry/sources", json=new).json()["committed"] is True
    assert client.post("/api/registry/sources", json=new).status_code == 409
    assert client.post("/api/registry/sources",
                       json={"source": {"id": "x!", "name": "x"}}).status_code == 422
    assert client.post("/api/actions/validate").json()["ok"] is True


def test_domains_profile_entities_scheduler(client):
    doms = client.get("/api/domains").json()["domains"]
    profile = next(d for d in doms if d["name"] == "general_news")["profile"]
    profile["schedule"]["brief_cron"] = "15 06 * * *"
    assert client.put("/api/domains/general_news",
                      json={"profile": profile}).json()["committed"] is True
    profile["schedule"]["brief_cron"] = "99 99 * * *"
    assert client.put("/api/domains/general_news",
                      json={"profile": profile}).status_code == 422

    row = next(d for d in client.post("/api/scheduler/general_news",
               json={"enabled": True}).json()["domains"]
               if d["domain"] == "general_news")
    assert row["enabled"] is True and row["next_run"]
    client.post("/api/scheduler/general_news", json={"enabled": False})

    assert client.post("/api/domains", json={
        "name": "demo_beat", "description": "test",
        "include_tags": ["general_news"], "min_tier": 3,
        "brief_cron": None}).json()["committed"] is True
    ent = client.get("/api/domains/demo_beat/entities").json()
    assert ent["dictionaries"][0]["rows"] == []
    client.put("/api/domains/demo_beat/entities", json={
        "path": ent["dictionaries"][0]["path"],
        "rows": [{"surface_form": "WTI", "canonical_id": "CMD:WTI",
                  "type": "commodity"}]})
    ent = client.get("/api/domains/demo_beat/entities").json()
    assert ent["dictionaries"][0]["rows"][0]["canonical_id"] == "CMD:WTI"


def test_candidates_review(client):
    from argus.snapshots.db import CandidateRow, make_engine, make_session_factory
    from argus.settings import get_settings
    session = make_session_factory(make_engine(get_settings().resolved_db_url))()
    session.add(CandidateRow(url="https://tradepress.example/story",
                             netloc="tradepress.example", title="Trade Press",
                             rationale="hit", run_id="t"))
    session.add(CandidateRow(url="https://agg.example/x", netloc="agg.example",
                             title="Agg", rationale="hit", run_id="t"))
    session.commit(); session.close()

    cands = client.get("/api/candidates?status=pending").json()
    approve = next(c for c in cands if c["netloc"] == "tradepress.example")
    r = client.post(f"/api/candidates/{approve['id']}/approve",
                    json={"add_to_registry": True, "tag": "general_news",
                          "tier": 3}).json()
    assert r["added_to_registry"] is True
    added = next(s for s in client.get("/api/registry").json()["sources"]
                 if s["id"] == r["source_id"])
    assert added["status"] == "paused" and "UNREVIEWED" in added["license"]

    reject = next(c for c in cands if c["netloc"] == "agg.example")
    assert client.post(f"/api/candidates/{reject['id']}/reject",
                       json={"reason": "aggregator"}).json()["status"] == "rejected"


# ------------------------------------------------- settings + auth (LAST)
def test_settings_and_auth_gate(client):
    r = client.put("/api/settings", json={"changes": {
        "ARGUS_BRIEF_MAX_STORIES": "12",
        "ARGUS_WEB_PASSWORD": "hunter2"}}).json()
    assert "ARGUS_WEB_PASSWORD" in r["saved"]

    assert client.get("/api/settings").status_code == 401
    state = client.get("/api/authstate").json()
    assert state == {"required": True, "ok": False}
    assert client.post("/api/login",
                       json={"password": "wrong"}).status_code == 401
    good = client.post("/api/login", json={"password": "hunter2"})
    assert good.status_code == 200 and "argus_session" in good.cookies

    fields = {f["key"]: f for f in client.get("/api/settings").json()["fields"]}
    assert fields["ARGUS_BRIEF_MAX_STORIES"]["value"] == "12"
    assert fields["ARGUS_WEB_PASSWORD"]["set"] is True
    assert "value" not in fields["ARGUS_WEB_PASSWORD"]  # secrets never echoed

    client.cookies.clear()  # prove the Bearer header alone is sufficient
    bearer = {"Authorization": "Bearer hunter2"}
    assert client.get("/api/overview", headers=bearer).status_code == 200

    client.put("/api/settings", headers=bearer,
               json={"changes": {"ARGUS_WEB_PASSWORD": None}})
    assert client.get("/api/authstate").json()["required"] is False
