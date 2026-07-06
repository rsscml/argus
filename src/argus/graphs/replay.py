"""Evidence-identical replay for any workload (architecture SS11.3).

Synthesis re-runs the full claims tail (extract -> corroborate -> synthesize
-> verify) against the FROZEN evidence set in the manifest — no polling, no
re-retrieval. Byte-identical output is not guaranteed (LLM nondeterminism);
evidence-identical is: same snapshots, same citable universe.
"""
from __future__ import annotations

import json
from pathlib import Path

from argus.claims.engine import corroborate_claims
from argus.index.retrieval import RetrievedChunk
from argus.synthesis.render import render_brief, render_report
from argus.synthesis.verify import check_body, drop_unsupported
from argus.util import utcnow


def _evidence_from_manifest(manifest: dict) -> dict[str, RetrievedChunk]:
    return {
        sid: RetrievedChunk(
            chunk_id=e["chunk_id"], content_hash=e["content_hash"],
            source_id=e["source_id"], tier=e.get("tier"), score=0.0,
            text=e["text"], title=e.get("title"), url=e.get("url", ""),
            published_at=e.get("published_at"), cluster_id=e.get("cluster_id"),
            entity_ids=e.get("entity_ids") or [],
        )
        for sid, e in manifest["evidence"].items()
    }


def replay_run(deps, manifest_path: Path) -> tuple[Path, dict]:
    manifest = json.loads(manifest_path.read_text())
    evidence = _evidence_from_manifest(manifest)

    raw = []
    for sid, chunk in evidence.items():
        group = ""
        try:
            group = deps.registry.get(chunk.source_id).independence_group or ""
        except KeyError:
            pass
        raw.extend(deps.extractor.extract(sid, chunk, group))
    quant_cfg = deps.profile.claims.quantitative
    claims = corroborate_claims(
        raw, deps.profile.corroboration, matcher=deps.matcher,
        quant_rel_tol=quant_cfg.tolerance.relative,
        quant_abs_tol=quant_cfg.tolerance.absolute,
    )
    result = deps.synthesizer.write_brief(manifest["domain"], claims, evidence)
    report = check_body(result.body_md, evidence, deps.verifier)
    body, _ = drop_unsupported(result.body_md, report)

    common = dict(
        profile=deps.profile, domains_dir=deps.settings.domains_dir,
        domain=manifest["domain"], run_id=manifest["run_id"] + "-replay",
        generated_at=utcnow().isoformat(timespec="seconds"),
        body_md=body, evidence=evidence,
        registry_commit=manifest.get("registry_commit"),
        synthesizer=deps.synthesizer.name,
        embedder=deps.ingest_deps.embedder.name,
    )
    if manifest.get("workload") == "deep_research":
        rendered = render_report(question=manifest.get("question", ""), **common)
    else:
        rendered = render_brief(window_hours=manifest.get("window_hours", 0), **common)

    out_path = manifest_path.with_name(manifest["run_id"] + "-replay.md")
    out_path.write_text(rendered)
    comparison = {
        "snapshot_set_identical": True,  # frozen by construction
        "original_cited_sids": manifest["stats"]["cited_sids"],
        "replay_cited_sids": result.cited_sids,
    }
    return out_path, comparison
