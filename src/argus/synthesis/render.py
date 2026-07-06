"""Brief rendering (SS8.5 output, SS12.4 provenance footer)."""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, StrictUndefined

from argus.config.profile import DomainProfile
from argus.index.retrieval import RetrievedChunk

DEFAULT_TEMPLATE = """# Daily brief — {{ domain }}
*Generated {{ generated_at }} · window {{ window_hours }}h · run `{{ run_id }}`*

{{ body_md }}

---

## Sources
{% for sid, c in evidence.items() -%}
- **[{{ sid }}]** {{ c.source_id }}{% if c.tier %} (tier {{ c.tier }}){% endif %} — {{ c.title or c.url }} · published {{ c.published_at or "n/a" }} · snapshot `{{ c.content_hash[:12] }}`
{% endfor %}

---
*Provenance: registry `{{ registry_commit or "uncommitted" }}` · synthesizer `{{ synthesizer }}` · embedder `{{ embedder }}`.
Every claim above cites an immutable snapshot; see the run manifest for the full audit trail.
Research output only — not investment, legal, or operational advice.*
"""


def render_brief(
    *,
    profile: DomainProfile,
    domains_dir: Path,
    domain: str,
    run_id: str,
    generated_at: str,
    window_hours: int,
    body_md: str,
    evidence: dict[str, RetrievedChunk],
    registry_commit: str | None,
    synthesizer: str,
    embedder: str,
) -> str:
    template_src = DEFAULT_TEMPLATE
    if profile.output.brief_template is not None:
        path = profile.output.brief_template
        if not path.is_absolute():
            path = domains_dir / domain / path
        if path.exists():
            template_src = path.read_text()
    env = Environment(undefined=StrictUndefined, keep_trailing_newline=True)
    return env.from_string(template_src).render(
        domain=domain,
        run_id=run_id,
        generated_at=generated_at,
        window_hours=window_hours,
        body_md=body_md,
        evidence=evidence,
        registry_commit=registry_commit,
        synthesizer=synthesizer,
        embedder=embedder,
    )


REPORT_TEMPLATE = DEFAULT_TEMPLATE.replace(
    "# Daily brief — {{ domain }}",
    "# Research report — {{ domain }}\n*Question: {{ question }}*",
).replace("window {{ window_hours }}h · ", "")


def render_report(
    *, profile: DomainProfile, domains_dir: Path, domain: str, run_id: str,
    generated_at: str, question: str, body_md: str, evidence, registry_commit,
    synthesizer: str, embedder: str,
) -> str:
    template_src = REPORT_TEMPLATE
    if profile.output.report_template is not None:
        path = profile.output.report_template
        if not path.is_absolute():
            path = domains_dir / domain / path
        if path.exists():
            template_src = path.read_text()
    env = Environment(undefined=StrictUndefined, keep_trailing_newline=True)
    return env.from_string(template_src).render(
        domain=domain, run_id=run_id, generated_at=generated_at,
        question=question, body_md=body_md, evidence=evidence,
        registry_commit=registry_commit, synthesizer=synthesizer, embedder=embedder,
    )
