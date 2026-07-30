"""Garde-fou mémoire — **la machine ne doit jamais geler à cause d'un build**.

Retour d'expérience : un build a saturé la RAM (accumulation de tous les Samples
en mémoire + téléchargement intégral des datasets). Le noyau s'est mis à
thrasher (``journald: Under memory pressure``) et la machine a gelé *sans* que
l'OOM-killer n'intervienne — le pire scénario Linux sur un poste de travail.

Trois lignes de défense, indépendantes :

1. **Pipeline en flux** (voir :mod:`galsenai_sft.build`) : rien n'est accumulé.
2. **Ce garde-fou** : un thread surveille la mémoire *disponible du système* et
   le *RSS du processus* ; sous le plancher, le build s'arrête **proprement**
   (fichiers fermés, manifest écrit) au lieu d'entraîner la machine avec lui.
3. **cgroup** (``make build``) : plafond dur ``MemoryMax`` — même en cas de bug,
   seul le build meurt, jamais la session graphique.

Aucune dépendance externe (lecture de ``/proc``) : utilisable partout, testable.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from galsenai_sft.core.logging import get_logger

if TYPE_CHECKING:  # évite un import circulaire à l'exécution
    from galsenai_sft.core.config import MemoryConfig

log = get_logger(__name__)

#: Mémoire système disponible en dessous de laquelle on arrête le build.
#: 1,5 Go laisse de quoi finir d'écrire les fichiers et rendre la main au bureau.
DEFAULT_MIN_AVAILABLE_MB = 1536.0

#: Période d'échantillonnage du surveillant (secondes).
DEFAULT_INTERVAL_S = 2.0

_MEMINFO = Path("/proc/meminfo")
_STATUS = Path("/proc/self/status")


class MemoryPressure(RuntimeError):
    """Levée quand la mémoire passe sous le plancher configuré."""


def _read_kb(path: Path, keys: tuple[str, ...]) -> dict[str, float]:
    """Extrait des valeurs ``clé: N kB`` d'un fichier ``/proc`` (Mo retournés)."""
    out: dict[str, float] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:  # pragma: no cover — /proc absent (macOS, conteneur exotique)
        return out
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        if name in keys:
            parts = rest.split()
            if parts and parts[0].isdigit():
                out[name] = int(parts[0]) / 1024.0
    return out


def available_mb() -> float:
    """Mémoire réellement allouable sans swapper (``MemAvailable``), en Mo.

    ``MemAvailable`` est l'indicateur juste : il tient compte du cache
    récupérable, contrairement à ``MemFree`` qui sous-estime massivement.
    """
    return _read_kb(_MEMINFO, ("MemAvailable",)).get("MemAvailable", float("inf"))


def process_rss_mb() -> float:
    """Mémoire résidente du processus courant, en Mo."""
    return _read_kb(_STATUS, ("VmRSS",)).get("VmRSS", 0.0)


def cgroup_limit_mb() -> float | None:
    """Plafond mémoire du cgroup courant (``MemoryMax``), en Mo, si défini."""
    try:
        rel = Path("/proc/self/cgroup").read_text(encoding="utf-8").strip().split(":")[-1]
        raw = Path(f"/sys/fs/cgroup{rel}/memory.max").read_text(encoding="utf-8").strip()
    except (OSError, IndexError):
        return None
    return None if raw == "max" else int(raw) / 1024 / 1024


def apply_address_space_limit(max_gb: float) -> None:
    """Plafond dur d'espace d'adressage du processus (``RLIMIT_AS``).

    Dernier filet : au lieu de dévorer la RAM de la machine, le processus lève
    ``MemoryError``. À n'utiliser que si le pipeline n'utilise pas de gros
    ``mmap`` (le mode streaming, par défaut, n'en fait pas).
    """
    import resource

    limit = int(max_gb * 1024**3)
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    log.info("limite d'espace d'adressage fixée à %.1f Go", limit / 1024**3)


class MemoryGuard:
    """Surveillant mémoire non bloquant.

    Un thread démon échantillonne la mémoire toutes les ``interval_s``. Le
    pipeline appelle :meth:`check` entre deux samples (coût : lecture d'un
    booléen) et s'arrête proprement si la pression est détectée.

    Usage::

        with MemoryGuard() as guard:
            for sample in flux:
                guard.check()          # lève MemoryPressure si plancher franchi
                ...

    ``min_available_mb=0`` désactive la surveillance système (utile en test/CI).
    """

    def __init__(
        self,
        min_available_mb: float = DEFAULT_MIN_AVAILABLE_MB,
        max_rss_mb: float | None = None,
        interval_s: float = DEFAULT_INTERVAL_S,
        raise_on_pressure: bool = True,
    ) -> None:
        self.min_available_mb = min_available_mb
        self.max_rss_mb = max_rss_mb
        self.interval_s = interval_s
        self.raise_on_pressure = raise_on_pressure
        self.peak_rss_mb = 0.0
        self.reason: str | None = None
        self._triggered = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def from_settings(cls, cfg: MemoryConfig) -> MemoryGuard:
        """Construit le garde-fou depuis ``configs/settings.yaml`` (section ``memory``)."""
        return cls(
            min_available_mb=cfg.min_available_mb,
            max_rss_mb=cfg.max_rss_mb,
            interval_s=cfg.interval_s,
        )

    # ------------------------------------------------------------------ état
    @property
    def triggered(self) -> bool:
        """Vrai si la pression mémoire a été détectée."""
        return self._triggered.is_set()

    def sample(self) -> tuple[float, float]:
        """Relève (mémoire disponible, RSS) en Mo et met à jour le pic."""
        avail, rss = available_mb(), process_rss_mb()
        self.peak_rss_mb = max(self.peak_rss_mb, rss)
        return avail, rss

    def evaluate(self) -> None:
        """Un tour de surveillance : arme le drapeau si un seuil est franchi."""
        avail, rss = self.sample()
        if self.min_available_mb and avail < self.min_available_mb:
            self._trip(
                f"mémoire système disponible {avail:.0f} Mo "
                f"< plancher {self.min_available_mb:.0f} Mo (RSS build {rss:.0f} Mo)"
            )
        elif self.max_rss_mb and rss > self.max_rss_mb:
            self._trip(f"RSS du build {rss:.0f} Mo > plafond {self.max_rss_mb:.0f} Mo")

    def _trip(self, reason: str) -> None:
        if not self._triggered.is_set():
            self.reason = reason
            self._triggered.set()
            log.error("PRESSION MÉMOIRE — arrêt propre demandé : %s", reason)

    # ------------------------------------------------------- boucle & cycle
    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.evaluate()
            if self._triggered.is_set():
                return  # inutile de continuer à surveiller

    def start(self) -> MemoryGuard:
        self.evaluate()  # contrôle immédiat : ne pas démarrer un build sur une machine déjà saturée
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="memory-guard", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 1)
            self._thread = None

    def check(self) -> None:
        """À appeler dans la boucle chaude : lève ``MemoryPressure`` si armé."""
        if self._triggered.is_set() and self.raise_on_pressure:
            raise MemoryPressure(self.reason or "pression mémoire")

    def __enter__(self) -> MemoryGuard:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def trim_heap() -> None:
    """Rend au système la mémoire libérée mais retenue par l'allocateur (glibc).

    Après avoir déchargé un gros modèle, ``free()`` ne restitue pas forcément
    les pages au noyau : ``malloc_trim`` force la restitution. Sans effet ailleurs
    que sous glibc — l'échec est silencieux et sans conséquence.
    """
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):  # pragma: no cover — plateformes non-glibc
        pass


def describe_environment() -> str:
    """Ligne de log récapitulant les moyens mémoire du processus."""
    limit = cgroup_limit_mb()
    cg = f"{limit / 1024:.1f} Go (cgroup)" if limit else "aucun plafond cgroup"
    return f"mémoire : {available_mb() / 1024:.1f} Go disponibles · {cg} · PID {os.getpid()}"
