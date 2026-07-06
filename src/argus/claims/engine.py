"""Corroboration engine (architecture SS8.4).

Deterministic policy code — NOT an LLM. The only LLM-assisted step is event
pairing, injected behind the EventMatcher seam (heuristic offline / Azure at
temperature 0), and even that is advisory: independence counting, tier rules,
tolerance matching, and verdicts are pure policy.

Steps (SS8.4):
  1. Group candidate-matching claims (events via matcher; quantities by
     (entity, metric, as-of window) under the profile's value tolerance).
  2. Independence: supporters sharing a story_cluster_id OR the same registry
     independence_group collapse into one voice (union-find).
  3. Apply per-type policy (min_sources, min_tier1) -> verdict.
  4. Contradictions are OUTPUT, not resolved.
"""
from __future__ import annotations

from typing import Protocol

from argus.claims.models import Claim, ClaimEvidence, CorroboratedClaim, Verdict
from argus.config.profile import CorroborationConfig
from argus.ingest.sparse import tokenize

_NEGATION_TOKENS = frozenset({"denied", "deny", "denies", "no", "not", "never"})


class EventMatcher(Protocol):
    def same_event(self, a: Claim, b: Claim) -> bool: ...


class HeuristicEventMatcher:
    """Content-token Jaccard similarity; negation tokens excluded so a denial
    still matches the assertion it denies (that pair becomes a contradiction)."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def _tokens(self, claim: Claim) -> set[str]:
        return {t for t in tokenize(claim.text) if t not in _NEGATION_TOKENS}

    def same_event(self, a: Claim, b: Claim) -> bool:
        ta, tb = self._tokens(a), self._tokens(b)
        if not ta or not tb:
            return False
        return len(ta & tb) / len(ta | tb) >= self.threshold


class LlmEventMatcher:
    """Azure-backed pairing at temperature 0 (SS8.4 step 1). Pre-filtered by a
    cheap token-overlap gate so LLM calls stay bounded."""

    def __init__(self, chat_model=None, gate: float = 0.2) -> None:
        if chat_model is None:
            from argus.llm.factory import get_chat_model

            chat_model = get_chat_model("utility", temperature=0.0)
        self._model = chat_model
        self._gate = HeuristicEventMatcher(gate)

    def same_event(self, a: Claim, b: Claim) -> bool:
        if not self._gate.same_event(a, b):
            return False
        prompt = (
            "Do these two sentences assert (or deny) the SAME underlying event? "
            "Answer strictly YES or NO.\n"
            f"A: {a.text}\nB: {b.text}"
        )
        response = self._model.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        return "YES" in str(content).upper()


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        self.parent[self.find(i)] = self.find(j)

    def components(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            out.setdefault(self.find(i), []).append(i)
        return out


def count_voices(evidence: list[ClaimEvidence]) -> tuple[int, int]:
    """(independent_voices, tier1_voices): evidence items sharing a cluster_id
    or an independence_group are ONE voice (SS8.4 step 2)."""
    if not evidence:
        return 0, 0
    uf = _UnionFind(len(evidence))
    for i in range(len(evidence)):
        for j in range(i + 1, len(evidence)):
            a, b = evidence[i], evidence[j]
            same_cluster = a.cluster_id is not None and a.cluster_id == b.cluster_id
            same_group = a.independence_group and a.independence_group == b.independence_group
            if same_cluster or same_group:
                uf.union(i, j)
    components = uf.components().values()
    tier1 = sum(1 for c in components if any(evidence[i].tier == 1 for i in c))
    return len(components), tier1


def _cluster_claims(claims: list[Claim], matcher: EventMatcher) -> list[list[Claim]]:
    uf = _UnionFind(len(claims))
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            if matcher.same_event(claims[i], claims[j]):
                uf.union(i, j)
    return [[claims[i] for i in idxs] for idxs in uf.components().values()]


def _merged(group: list[Claim]) -> tuple[list[ClaimEvidence], list[str], list[str], str | None]:
    evidence = [e for c in group for e in c.evidence]
    sids = sorted({e.sid for e in evidence}, key=lambda s: int(s[1:]))
    sources = sorted({e.source_id for e in evidence})
    story = next((e.cluster_id for e in evidence if e.cluster_id), None)
    return evidence, sids, sources, story


def _apply_rule(
    voices: int, tier1: int, rule, fallback: str
) -> tuple[Verdict, bool]:
    """-> (verdict, dropped). Below threshold follows the domain fallback (SS5.2/R4)."""
    if voices >= rule.min_sources and tier1 >= rule.min_tier1:
        return Verdict.confirmed, False
    if fallback == "attribute":
        return Verdict.attributed, False
    if fallback == "flag":
        return Verdict.insufficient, False
    return Verdict.insufficient, True  # drop


def _corroborate_events(
    events: list[Claim], config: CorroborationConfig, matcher: EventMatcher
) -> list[CorroboratedClaim]:
    out: list[CorroboratedClaim] = []
    for group in _cluster_claims(events, matcher):
        asserted = [c for c in group if c.modality == "asserted"]
        denied = [c for c in group if c.modality == "denied"]
        evidence, sids, sources, story = _merged(group)

        if asserted and denied:
            a_srcs = sorted({e.source_id for c in asserted for e in c.evidence})
            d_srcs = sorted({e.source_id for c in denied for e in c.evidence})
            voices, tier1 = count_voices(evidence)
            out.append(CorroboratedClaim(
                text=asserted[0].text, claim_type="event",
                verdict=Verdict.contradicted, independent_voices=voices,
                tier1_voices=tier1, sids=sids, sources=sources, story_cluster=story,
                conflict=f"asserted by {', '.join(a_srcs)}; denied by {', '.join(d_srcs)}",
            ))
            continue

        voices, tier1 = count_voices(evidence)
        verdict, dropped = _apply_rule(voices, tier1, config.event, config.fallback)
        out.append(CorroboratedClaim(
            text=group[0].text, claim_type="event", verdict=verdict,
            independent_voices=voices, tier1_voices=tier1, sids=sids,
            sources=sources, story_cluster=story, dropped=dropped,
        ))
    return out


def _values_agree(a: float, b: float, rel_tol: float | None, abs_tol: float | None) -> bool:
    if abs_tol is not None and abs(a - b) <= abs_tol:
        return True
    if rel_tol is not None and abs(a - b) <= rel_tol * max(abs(a), abs(b)):
        return True
    return rel_tol is None and abs_tol is None and a == b


def _corroborate_quants(
    quants: list[Claim], config: CorroborationConfig,
    rel_tol: float | None, abs_tol: float | None,
) -> list[CorroboratedClaim]:
    rule = config.quantitative
    if rule.value_tolerance != "inherit":
        rel_tol, abs_tol = float(rule.value_tolerance), None

    by_key: dict[tuple, list[Claim]] = {}
    for claim in quants:
        key = (claim.entity_id or "?", claim.metric or "?", (claim.as_of or "")[:10])
        by_key.setdefault(key, []).append(claim)

    out: list[CorroboratedClaim] = []
    for key, group in by_key.items():
        # greedy value-grouping under tolerance
        value_groups: list[list[Claim]] = []
        for claim in sorted(group, key=lambda c: c.value or 0.0):
            for vg in value_groups:
                if _values_agree(vg[0].value or 0.0, claim.value or 0.0, rel_tol, abs_tol):
                    vg.append(claim)
                    break
            else:
                value_groups.append([claim])

        evidence, sids, sources, story = _merged(group)
        voices, tier1 = count_voices(evidence)

        live = [vg for vg in value_groups if vg]
        if len(live) > 1:
            descs = []
            for vg in live:
                srcs = sorted({e.source_id for c in vg for e in c.evidence})
                descs.append(f"{vg[0].value} {vg[0].unit or ''} ({', '.join(srcs)})".strip())
            out.append(CorroboratedClaim(
                text=live[0][0].text, claim_type="quantitative",
                verdict=Verdict.contradicted, independent_voices=voices,
                tier1_voices=tier1, sids=sids, sources=sources, story_cluster=story,
                conflict="conflicting values: " + " vs ".join(descs),
            ))
            continue

        verdict, dropped = _apply_rule(voices, tier1, rule, config.fallback)
        out.append(CorroboratedClaim(
            text=group[0].text, claim_type="quantitative", verdict=verdict,
            independent_voices=voices, tier1_voices=tier1, sids=sids,
            sources=sources, story_cluster=story, dropped=dropped,
        ))
    return out


def corroborate_claims(
    claims: list[Claim],
    config: CorroborationConfig,
    *,
    matcher: EventMatcher,
    quant_rel_tol: float | None = None,
    quant_abs_tol: float | None = None,
) -> list[CorroboratedClaim]:
    events = [c for c in claims if c.claim_type == "event"]
    quants = [c for c in claims if c.claim_type == "quantitative"]
    results = _corroborate_events(events, config, matcher)
    results += _corroborate_quants(quants, config, quant_rel_tol, quant_abs_tol)
    return results
