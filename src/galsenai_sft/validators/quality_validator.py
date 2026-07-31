"""Validation qualité d'un lot de Samples.

Détecte :
  - contenus vides / trop courts / anormalement longs (seuils configurables) ;
  - **doublons exacts** (empreinte normalisée de la conversation) ;
  - réponse assistant vide ou manquante ;
  - tours assistant identiques à la question (copie), signal de bruit MT.

Renvoie un :class:`ValidationReport`. Peut aussi **filtrer** un flux en retirant
les Samples portant une erreur bloquante (utilisé par la CLI ``build``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator

from galsenai_sft.core.config import QualityConfig, get_settings
from galsenai_sft.core.schema import Role, Sample
from galsenai_sft.validators.report import Issue, Severity, ValidationReport


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def sample_fingerprint(sample: Sample) -> str:
    """Empreinte stable d'une conversation (pour la déduplication exacte)."""
    joined = "\n".join(f"{m.role.value}:{_normalize(m.content)}" for m in sample.messages)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def check_sample(sample: Sample, cfg: QualityConfig, index: int | None = None) -> list[Issue]:
    """Contrôles qualité sur un Sample isolé (hors doublons, qui sont globaux)."""
    issues: list[Issue] = []

    def add(code: str, sev: Severity, msg: str) -> None:
        issues.append(
            Issue(code=code, severity=sev, message=msg, sample_index=index, source=sample.source)
        )

    assistant_turns = [m for m in sample.messages if m.role is Role.ASSISTANT]
    if not assistant_turns:
        add("missing_assistant", Severity.ERROR, "aucun tour assistant")
    for m in assistant_turns:
        if not (m.content and m.content.strip()) and not m.tool_calls:
            add("empty_assistant", Severity.ERROR, "réponse assistant vide")

    for m in sample.messages:
        n = len(m.content)
        if m.content.strip() and n < cfg.min_chars:
            add("too_short", Severity.WARNING, f"contenu très court ({n} car)")
        if n > cfg.max_chars:
            add("too_long", Severity.WARNING, f"contenu très long ({n} car)")

    # Copie question -> réponse (bruit fréquent des corpus traduits)
    users = [_normalize(m.content) for m in sample.messages if m.role is Role.USER]
    for m in assistant_turns:
        if _normalize(m.content) and _normalize(m.content) in users:
            add("echo_answer", Severity.WARNING, "réponse identique à la question")

    return issues


def validate_quality(
    samples: Iterable[Sample], cfg: QualityConfig | None = None
) -> ValidationReport:
    """Valide un lot complet (contrôles par sample + doublons globaux)."""
    cfg = cfg or get_settings().quality
    report = ValidationReport()
    seen: dict[str, int] = {}

    for i, s in enumerate(samples):
        report.total += 1
        report.extend(check_sample(s, cfg, index=i))
        if cfg.drop_duplicates:
            fp = sample_fingerprint(s)
            if fp in seen:
                report.add(
                    Issue(
                        code="duplicate",
                        severity=Severity.ERROR,
                        message=f"doublon de l'exemple #{seen[fp]}",
                        sample_index=i,
                        source=s.source,
                    )
                )
            else:
                seen[fp] = i
    return report


def filter_quality(
    samples: Iterable[Sample],
    cfg: QualityConfig | None = None,
    seen: set[int] | None = None,
) -> Iterator[Sample]:
    """Filtre un flux : retire vides/doublons (erreurs bloquantes), garde le reste.

    ``seen`` permet de **partager l'index de déduplication entre plusieurs
    datasets** : indispensable dès que des sources se recouvrent (plusieurs
    corpus de traduction wolof agrègent les mêmes phrases OPUS/MAFAND). Sans
    index partagé, chaque dataset ne se dédupliquerait que contre lui-même et
    le dataset final accumulerait les doublons inter-sources.

    L'index ne conserve que **64 bits** d'empreinte par exemple (au lieu des 64
    caractères hexadécimaux) : ~4× moins de mémoire, soit quelques dizaines de
    Mo même sur des millions d'exemples. Risque de collision négligeable
    (< 1e-7 sur 1 M d'exemples) et sans conséquence : un doublon supposé est
    simplement écarté.
    """
    cfg = cfg or get_settings().quality
    seen = seen if seen is not None else set()
    for s in samples:
        issues = check_sample(s, cfg)
        if any(i.severity is Severity.ERROR for i in issues):
            continue
        if cfg.drop_duplicates:
            fp = int(sample_fingerprint(s)[:16], 16)
            if fp in seen:
                continue
            seen.add(fp)
        yield s
