"""Shared tail nodes: corroborate -> synthesize -> verify (SS8.2, SS4 invariant 2).

Both the daily brief graph and the deep research graph attach this SAME tail,
so trust policy executes in exactly one place in the codebase — a brief can
never bypass a rule that deep research respects, and vice versa.

`deps` is duck-typed: it must expose registry, profile, extractor, matcher,
synthesizer, verifier, and settings. BriefDeps and ResearchDeps both do.
"""
from __future__ import annotations

from langgraph.graph import END

from argus.claims.engine import corroborate_claims
from argus.graphs.state import TailState
from argus.synthesis.verify import check_body, drop_unsupported


def attach_tail(graph, deps, after: str) -> None:
    def corroborate(state: TailState) -> dict:
        raw = []
        for sid, chunk in state.evidence.items():
            group = ""
            try:
                group = deps.registry.get(chunk.source_id).independence_group or ""
            except KeyError:
                pass
            raw.extend(deps.extractor.extract(sid, chunk, group))
        quant_cfg = deps.profile.claims.quantitative
        corroborated = corroborate_claims(
            raw,
            deps.profile.corroboration,
            matcher=deps.matcher,
            quant_rel_tol=quant_cfg.tolerance.relative,
            quant_abs_tol=quant_cfg.tolerance.absolute,
        )
        counts: dict[str, int] = {}
        for claim in corroborated:
            counts[claim.verdict.value] = counts.get(claim.verdict.value, 0) + 1
        return {"claims": corroborated, "verdict_counts": counts}

    def synthesize(state: TailState) -> dict:
        result = deps.synthesizer.write_brief(
            state.domain, state.claims, state.evidence,
            feedback=state.verifier_feedback or None,
        )
        return {
            "body_md": result.body_md,
            "cited_sids": result.cited_sids,
            "uncited_sentences": result.uncited_sentences,
            "invalid_citations_stripped": result.invalid_citations_stripped,
        }

    def verify(state: TailState) -> dict:
        report = check_body(state.body_md, state.evidence, deps.verifier)
        failed = [c.sentence for c in report.failed]
        if failed and state.verify_attempts < deps.settings.verify_max_retries:
            return {
                "verify_attempts": state.verify_attempts + 1,
                "verifier_feedback": failed,
                "needs_retry": True,
            }
        body, dropped = drop_unsupported(state.body_md, report)
        return {
            "body_md": body,
            "needs_retry": False,
            "verifier_feedback": [],
            "sentences_checked": report.checked,
            "sentences_dropped": dropped,
            "citation_precision": report.precision,
        }

    graph.add_node("corroborate", corroborate)
    graph.add_node("synthesize", synthesize)
    graph.add_node("verify", verify)
    graph.add_edge(after, "corroborate")
    graph.add_edge("corroborate", "synthesize")
    graph.add_edge("synthesize", "verify")
    graph.add_conditional_edges(
        "verify",
        lambda state: "synthesize" if state.needs_retry else END,
        {"synthesize": "synthesize", END: END},
    )


def tail_manifest_fields(state: TailState) -> dict:
    """Manifest fragments shared by both workloads (SS12.1)."""
    return {
        "claims": [c.model_dump(mode="json") for c in state.claims],
        "verdict_counts": state.verdict_counts,
        "verification": {
            "sentences_checked": state.sentences_checked,
            "sentences_dropped": state.sentences_dropped,
            "attempts": state.verify_attempts,
            "citation_precision": state.citation_precision,
        },
    }


def serialize_evidence(evidence) -> dict:
    return {
        sid: {
            "chunk_id": c.chunk_id, "content_hash": c.content_hash,
            "source_id": c.source_id, "tier": c.tier, "title": c.title,
            "url": c.url, "published_at": c.published_at,
            "cluster_id": c.cluster_id, "entity_ids": c.entity_ids,
            "text": c.text,
        }
        for sid, c in evidence.items()
    }
