"""Golden-set harness (architecture SS12.5, M4 exit criterion).

Cases live in tests/golden/<domain>/cases.yaml: canned evidence documents with
declared sources/tiers/clusters, plus expected verdicts. The runner executes
the offline claims pipeline (extract -> corroborate -> synthesize -> verify)
and scores expectation pass-rate and citation precision against targets.
Runs in CI on any change to prompts, profiles, or retrieval settings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from argus.claims.engine import corroborate_claims
from argus.config.profile import DomainProfile
from argus.index.retrieval import RetrievedChunk
from argus.synthesis.verify import check_body
from argus.util import utcnow


@dataclass
class GoldenOutcome:
    case: str
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    citation_precision: float = 1.0


@dataclass
class GoldenReport:
    outcomes: list[GoldenOutcome] = field(default_factory=list)
    targets: dict = field(default_factory=dict)

    @property
    def expectation_pass_rate(self) -> float:
        total = sum(len(o.passed) + len(o.failed) for o in self.outcomes)
        return 1.0 if not total else sum(len(o.passed) for o in self.outcomes) / total

    @property
    def citation_precision(self) -> float:
        vals = [o.citation_precision for o in self.outcomes]
        return min(vals) if vals else 1.0

    @property
    def ok(self) -> bool:
        return (
            self.expectation_pass_rate >= self.targets.get("expectation_pass_rate", 1.0)
            and self.citation_precision >= self.targets.get("citation_precision", 0.95)
        )


def _chunks_for(case: dict) -> dict[str, RetrievedChunk]:
    now = utcnow().isoformat(timespec="seconds")
    evidence: dict[str, RetrievedChunk] = {}
    for i, doc in enumerate(case["docs"], 1):
        sid = f"S{i}"
        content_hash = f"golden{i:02d}" + "0" * 56
        evidence[sid] = RetrievedChunk(
            chunk_id=f"{content_hash}:0", content_hash=content_hash,
            source_id=doc["source_id"], tier=doc.get("tier"), score=0.0,
            text=doc["text"], title=doc.get("title", doc["source_id"]),
            url=f"golden://{doc['source_id']}",
            published_at=doc.get("published_at", now),
            cluster_id=doc.get("cluster", f"cl-{i}"),
            entity_ids=doc.get("entity_ids", []),
        )
    return evidence


def run_golden(
    domain_profile: DomainProfile, cases_path: Path, *,
    extractor, matcher, verifier, synthesizer,
) -> GoldenReport:
    spec = yaml.safe_load(cases_path.read_text())
    report = GoldenReport(targets=spec.get("targets", {}))
    quant_cfg = domain_profile.claims.quantitative

    for case in spec.get("cases", []):
        outcome = GoldenOutcome(case=case["name"])
        evidence = _chunks_for(case)
        raw = []
        for sid, chunk in evidence.items():
            group = next(
                (d.get("group", d["source_id"]) for d in case["docs"]
                 if d["source_id"] == chunk.source_id), chunk.source_id,
            )
            raw.extend(extractor.extract(sid, chunk, group))
        claims = corroborate_claims(
            raw, domain_profile.corroboration, matcher=matcher,
            quant_rel_tol=quant_cfg.tolerance.relative,
            quant_abs_tol=quant_cfg.tolerance.absolute,
        )

        for expect in case.get("expect", []):
            needle = expect["claim_contains"].lower()
            wanted_type = expect.get("claim_type")
            matches = [
                c for c in claims
                if needle in c.text.lower()
                and (wanted_type is None or c.claim_type == wanted_type)
            ]
            label = f"{expect['claim_contains']} -> {expect['verdict']}"
            if matches and all(c.verdict.value == expect["verdict"] for c in matches):
                outcome.passed.append(label)
            else:
                got = sorted({c.verdict.value for c in matches}) or ["<no matching claim>"]
                outcome.failed.append(f"{label} (got {', '.join(got)})")

        result = synthesizer.write_brief(domain_profile.domain, claims, evidence)
        outcome.citation_precision = check_body(
            result.body_md, evidence, verifier
        ).precision
        report.outcomes.append(outcome)
    return report
