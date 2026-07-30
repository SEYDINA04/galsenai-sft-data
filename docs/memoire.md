# 🧠 Mémoire — pourquoi la machine ne peut plus geler

> Incident du **30/07/2026, 17:15** : la machine (15 Go de RAM) a **gelé** pendant
> un build. Journal système : `systemd-journald: Under memory pressure, flushing
> caches` en boucle de 17:14:46 à 17:15:45, puis plus rien — redémarrage forcé.
> Aucun processus tué par l'OOM-killer : c'est le scénario Linux le plus
> pénible, le *thrashing*, où le noyau passe son temps à recharger des pages au
> lieu de travailler, sans jamais décider de tuer le coupable.

Ce document explique la cause et les trois protections mises en place.

---

## 1. Cause exacte

| Facteur | Détail |
|---|---|
| **Accumulation en RAM** | `build()` gardait **tous** les `Sample` dans `all_samples` + `by_task_samples` avant d'écrire. Code-170k = 344 000 conversations d'objets pydantic → plusieurs Go. |
| **Téléchargement intégral** | `load_dataset(streaming=False)` télécharge le dataset *entier* avant la 1ʳᵉ ligne (cache HF observé : **14 Go**, dont 6 Go de FLEURS et 3,6 Go de Common Voice). |
| **Modèle LID résident** | GlotLID v3 = **1,6 Go** en RAM, conservé pour toute la durée du build. |
| **Aucune borne** | Rien ne surveillait la mémoire : le build a consommé jusqu'au gel du bureau. |

---

## 2. Trois lignes de défense (indépendantes)

### Ligne 1 — Le pipeline ne stocke rien (`build.py`)

Chaque exemple est **écrit puis oublié** :

```
ligne brute → converter → qualité → LID → décontamination → SampleSink.write()
                                                            (+ stats incrémentales)
```

- `iter_entry_samples()` est un **générateur** : aucune liste intermédiaire.
- `SampleSink` écrit simultanément dans `chatml/all.jsonl`, `chatml/<tâche>.jsonl`,
  `alpaca/all.jsonl`, `sharegpt/all.jsonl`, et calcule les **checksums au vol**
  (plus de seconde passe de lecture sur des fichiers de plusieurs Go).
- `StatisticsAccumulator` agrège au fil de l'eau (compteurs, pas d'échantillons).
- Le LID est **libéré** dès le dernier dataset qui l'exige (`release_identifier()`
  + `malloc_trim`) : mesuré **2 010 Mo → 215 Mo**.

Résultat mesuré sur le build complet : **RSS stable ≈ 180 Mo**, quel que soit le
volume traité.

### Ligne 2 — Le garde-fou logiciel (`core/memory.py`)

Un thread surveille `MemAvailable` (mémoire système réellement allouable) et le
RSS du processus. Sous le plancher (`memory.min_available_mb`, 1,5 Go par
défaut), le build **s'arrête proprement** :

- fichiers fermés et checksums écrits ;
- manifest écrit avec `partial: true` et `stop_reason` ;
- les données déjà produites restent exploitables.

Un `Ctrl-C` suit exactement le même chemin. Le contrôle a lieu **avant** de
démarrer (inutile de lancer un build sur une machine déjà saturée).

### Ligne 3 — Le plafond dur cgroup (`scripts/build_guarded.sh`)

```bash
make build          # plafond = 60 % de la RAM, swap interdit
make build-smoke    # test rapide : 100 lignes/dataset, plafond 4 Go
MEM_MAX=6G scripts/build_guarded.sh --limit 1000
```

Le build tourne dans un *scope* systemd avec `MemoryMax` / `MemoryHigh` /
`MemorySwapMax=0`. Même en cas de bug, **le noyau tue le build, jamais la
session graphique**. C'est la seule protection qui reste vraie quoi qu'il arrive
dans le code Python.

> `MemorySwapMax=0` est volontaire : c'est le swap qui transforme une saturation
> en gel de plusieurs minutes.

---

## 3. Chargement des données : streaming par défaut

`HFLoader(streaming=True)` est désormais le défaut (`build.streaming` dans
`configs/settings.yaml`) : les datasets sont lus **par lots** au lieu d'être
téléchargés en entier. S'y ajoutent :

- la **projection de colonnes** parquet (`columns:` dans `build.yaml`) — on ne
  lit jamais les colonnes audio/image ;
- `select_columns()` en repli si la branche parquet n'est pas disponible.

---

## 4. Vérifier avant de lancer

```bash
make doctor          # RAM libre, plafond cgroup, plancher du garde-fou, streaming
```

Le manifest de chaque build trace `peak_rss_mb`, `partial` et `stop_reason`.

---

## 5. Durcissement système (optionnel, hors dépôt)

`systemd-oomd` est actif sur cette machine mais ne surveille pas les processus
lancés depuis un terminal de session. Deux options pour protéger *tout* le
système (pas seulement nos builds) :

```bash
sudo apt install earlyoom            # tue le plus gros gourmand avant le gel
sudo systemctl enable --now earlyoom
```

ou une règle `systemd-oomd` sur `user-1000.slice` (`ManagedOOMMemoryPressure=kill`).

Ces réglages sont **hors du dépôt** (configuration machine) : à appliquer une
fois, avec `sudo`.
