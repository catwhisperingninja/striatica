# striatica/pipeline/graph_fetch.py
"""Neuronpedia attribution-graph fetching.

Two-step fetch of public attribution graphs (record -> S3 payload), source-set
lookup for L0 dictionary identity, optional authenticated graph generation, and
hard identity validation between a Neuronpedia graph and a local dataset.

Endpoints (verified live 2026-07-28):

  GET  {NEURONPEDIA_BASE}/api/graph/{model_id}/{slug}      (keyless, public)
       -> small record {modelId, sourceSetName, slug, url, ...};
       record["url"] is an S3 URL holding the full graph
       {metadata, qParams, nodes, links}.
  GET  {NEURONPEDIA_BASE}/api/source-set/{model_id}/{name} (keyless, public)
       -> {sources: [{id, saelensSaeId, hfRepoId, ...}]} — the L0 identity
       source ("layer_12/width_16k/average_l0_6" etc.).
  POST {NEURONPEDIA_BASE}/api/graph/generate               (x-api-key required)
       rate-limited to 30 graphs per hour.

The public GET endpoints are keyless by design: the API key must NEVER be sent
to them, even when NEURONPEDIA_API_KEY is set in the environment. The key value
itself must never be printed or logged.

Identity validation is three-state (verified / contradicted / unverifiable).
The real hazard it exists for: Neuronpedia's gemma-2-2b layer-12 transcoder
source is average_l0_6, while the local gemma-2-2b-layer12-l0604 dataset was
built from average_l0_604 — a DIFFERENT dictionary whose feature indices are
not comparable. A proven contradiction is always a hard failure.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from pipeline.banner import detail, step_cached, warn
from pipeline.config import DATA_DIR, PROJECT_ROOT

try:  # standard load_dotenv behavior when python-dotenv is installed
    from dotenv import load_dotenv as _load_dotenv
except ImportError:  # pragma: no cover - depends on environment
    _load_dotenv = None

NEURONPEDIA_BASE = "https://www.neuronpedia.org"
GRAPHS_CACHE_DIR = DATA_DIR / "graphs"

# The only source-set family whose graphs are comparable to the local
# gemma-scope transcoder datasets.
REQUIRED_SOURCE_SET = "gemmascope-transcoder-16k"
REQUIRED_WIDTH = "width_16k"
REQUIRED_NUM_FEATURES = 16384

# saelensSaeId format carrying the L0 identity, e.g.
# "layer_12/width_16k/average_l0_6".
_SAE_ID_RE = re.compile(r"^layer_(\d+)/width_16k/average_l0_(\d+)$")


# ── API key ─────────────────────────────────────────────────────────


def _read_env_file(env_path: Path) -> None:
    """Minimal .env loader for when python-dotenv is unavailable.

    Mirrors load_dotenv's default semantics: existing os.environ entries win.
    Never prints or logs any value.
    """
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


def load_neuronpedia_api_key() -> str | None:
    """Load NEURONPEDIA_API_KEY from the project-root .env (os.environ wins).

    Returns the key or None. The key value is never printed or logged.
    """
    env_path = PROJECT_ROOT / ".env"
    if _load_dotenv is not None:
        _load_dotenv(env_path)  # does not override existing os.environ
    else:
        _read_env_file(env_path)
    return os.environ.get("NEURONPEDIA_API_KEY")


# ── Transport helpers ───────────────────────────────────────────────


def _get_json(url: str, timeout: int = 30) -> dict:
    """GET a keyless public endpoint and parse the JSON payload.

    No headers are attached — in particular x-api-key is never sent.
    """
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        payload = resp.read()
    return json.loads(payload.decode("utf-8"))


# ── Public graph fetching (keyless) ─────────────────────────────────


def fetch_graph_record(model_id: str, slug: str, timeout: int = 30) -> dict:
    """Fetch the small graph record (step 1 of 2) for a public graph.

    Returns {modelId, sourceSetName, slug, url, ...}; record["url"] points at
    the full graph payload on S3.
    """
    url = f"{NEURONPEDIA_BASE}/api/graph/{model_id}/{slug}"
    try:
        return _get_json(url, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RuntimeError(
                f"Graph '{slug}' not found for model '{model_id}' "
                f"(HTTP 404 from {url}). Check the slug — public graphs are "
                f"listed on the model's page at neuronpedia.org."
            ) from e
        raise RuntimeError(
            f"Neuronpedia graph record request failed with HTTP {e.code} "
            f"for '{slug}' ({url})."
        ) from e


def fetch_graph(
    model_id: str,
    slug: str,
    cache_dir: Path | str = GRAPHS_CACHE_DIR,
    force: bool = False,
) -> dict:
    """Fetch the full attribution graph {metadata, qParams, nodes, links}.

    Two-step: graph record -> record["url"] (S3) -> full graph JSON. The full
    graph is cached at {cache_dir}/{model_id}/{slug}.json; a cache hit makes
    zero network calls unless force=True. A malformed network payload raises
    and is never written to the cache.
    """
    cache_file = Path(cache_dir) / model_id / f"{slug}.json"
    if cache_file.is_file() and not force:
        step_cached(f"{model_id}/{slug}.json")
        return json.loads(cache_file.read_text())

    record = fetch_graph_record(model_id, slug)
    s3_url = record.get("url")
    if not s3_url:
        raise RuntimeError(
            f"Graph record for '{slug}' has no 'url' field — cannot fetch "
            f"the full graph payload."
        )

    try:
        graph = _get_json(s3_url)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Malformed JSON in graph payload for '{slug}' ({s3_url}); "
            f"nothing was cached."
        ) from e

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(graph))
    detail(
        f"fetched graph '{slug}' "
        f"({len(graph.get('nodes', []))} nodes, "
        f"{len(graph.get('links', []))} links) -> {cache_file}"
    )
    return graph


def fetch_source_set(model_id: str, name: str, timeout: int = 30) -> dict:
    """Fetch a source set (keyless) — the L0 dictionary-identity source.

    Returns {sources: [{id, saelensSaeId, hfRepoId, ...}, ...], ...}.
    """
    url = f"{NEURONPEDIA_BASE}/api/source-set/{model_id}/{name}"
    try:
        return _get_json(url, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Neuronpedia source-set request failed with HTTP {e.code} "
            f"for '{name}' ({url})."
        ) from e


# ── Graph generation (requires API key) ─────────────────────────────


def generate_graph(
    model_id: str,
    prompt: str,
    slug: str,
    api_key: str | None,
    **params,
) -> dict:
    """Generate a new attribution graph via POST /api/graph/generate.

    Requires an API key (x-api-key header). Extra keyword params (maxNLogits,
    desiredLogitProb, maxFeatureNodes, sourceSetName, ...) are passed through
    to the request body verbatim. Rate limit: 30 graphs per hour; a 429 is
    reported with guidance and never retried.
    """
    if not api_key:
        raise RuntimeError(
            "Graph generation requires a Neuronpedia API key, but "
            "NEURONPEDIA_API_KEY is not set. Add NEURONPEDIA_API_KEY=<key> "
            "to the .env file at the project root. Without a key striatica "
            "runs in fetch-only mode: existing public graphs can still be "
            "fetched with fetch_graph()."
        )

    url = f"{NEURONPEDIA_BASE}/api/graph/generate"
    body = {"prompt": prompt, "slug": slug, "modelId": model_id, **params}
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as resp:
            payload = resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise RuntimeError(
                "Neuronpedia rate limit reached (HTTP 429): graph generation "
                "is limited to 30 graphs per hour. Wait for the window to "
                "reset and try again — the request is not retried "
                "automatically."
            ) from e
        raise RuntimeError(
            f"Neuronpedia graph generation failed with HTTP {e.code} for "
            f"slug '{slug}'."
        ) from e
    return json.loads(payload.decode("utf-8"))


# ── Identity validation ─────────────────────────────────────────────


def _dataset_layer(dataset_metadata: dict) -> int:
    """Extract the transformer layer index from dataset metadata.

    The metadata 'layer' field is e.g. "layer12-l0604".
    """
    layer_field = str(dataset_metadata.get("layer", ""))
    m = re.search(r"layer_?(\d+)", layer_field)
    if not m:
        raise ValueError(
            f"Cannot determine the dataset layer from metadata layer field "
            f"'{layer_field}'."
        )
    return int(m.group(1))


def validate_graph_identity(
    record: dict,
    graph: dict,
    source_set: dict,
    dataset_metadata: dict,
    allow_l0_mismatch: bool = False,
) -> dict:
    """Validate that a Neuronpedia graph and a local dataset share identity.

    Checks (all hard ValueError on failure):
      - model: record["modelId"] and graph["metadata"]["scan"] must both match
        dataset_metadata["model_id"]
      - source set: must be the gemmascope-transcoder-16k family (checked on
        both record["sourceSetName"] and source_set["name"])
      - dataset dictionary shape: num_features == 16384, transcoder width
        "width_16k"
      - L0 (three-state): the layer_{L}/width_16k/average_l0_{N} source for
        the dataset layer is compared to the dataset transcoder l0_variant.
          VERIFIED      -> returns {"l0": N, "l0Verified": True}
          CONTRADICTED  -> ValueError quoting both sides; NEVER overridable
                           (allow_l0_mismatch does not rescue a proven
                           mismatch — different dictionaries mean feature
                           indices are not comparable)
          UNVERIFIABLE  -> ValueError, unless allow_l0_mismatch=True, then
                           returns {"l0": <dataset l0>, "l0Verified": False}
    """
    dataset_model = dataset_metadata["model_id"]

    record_model = record.get("modelId")
    if record_model != dataset_model:
        raise ValueError(
            f"Model mismatch: graph record modelId is '{record_model}' but "
            f"the local dataset is '{dataset_model}'."
        )

    scan = graph.get("metadata", {}).get("scan")
    if scan != dataset_model:
        raise ValueError(
            f"Model mismatch: graph metadata.scan is '{scan}' but the local "
            f"dataset is '{dataset_model}'."
        )

    for label, name in (
        ("graph record sourceSetName", record.get("sourceSetName")),
        ("source set name", source_set.get("name")),
    ):
        if name != REQUIRED_SOURCE_SET:
            raise ValueError(
                f"Source set mismatch: {label} is '{name}', expected the "
                f"'{REQUIRED_SOURCE_SET}' transcoder family."
            )

    width = dataset_metadata.get("transcoder", {}).get("width")
    if width != REQUIRED_WIDTH:
        raise ValueError(
            f"Dataset transcoder width is '{width}', expected "
            f"'{REQUIRED_WIDTH}'."
        )

    num_features = dataset_metadata.get("num_features")
    if num_features != REQUIRED_NUM_FEATURES:
        raise ValueError(
            f"Dataset num_features is {num_features}, expected "
            f"{REQUIRED_NUM_FEATURES}."
        )

    layer = _dataset_layer(dataset_metadata)
    dataset_l0 = int(dataset_metadata["transcoder"]["l0_variant"])

    neuronpedia_l0: int | None = None
    for source in source_set.get("sources", []):
        m = _SAE_ID_RE.match(source.get("saelensSaeId", ""))
        if m and int(m.group(1)) == layer:
            neuronpedia_l0 = int(m.group(2))
            break

    if neuronpedia_l0 is None:
        if allow_l0_mismatch:
            warn(
                f"L0 identity unverifiable: no layer_{layer} source in "
                f"source set '{source_set.get('name')}' — proceeding with "
                f"l0Verified=False (dataset l0_variant={dataset_l0})."
            )
            return {"l0": dataset_l0, "l0Verified": False}
        raise ValueError(
            f"L0 identity unverifiable: no layer_{layer} source found in "
            f"source set '{source_set.get('name')}', so the Neuronpedia "
            f"graph cannot be confirmed to use the same dictionary as the "
            f"local dataset (l0_variant={dataset_l0}). Pass "
            f"allow_l0_mismatch=True to proceed with l0Verified=False."
        )

    if neuronpedia_l0 != dataset_l0:
        raise ValueError(
            f"L0 identity contradiction for layer {layer}: the local dataset "
            f"transcoder is average_l0_{dataset_l0} but Neuronpedia's "
            f"layer-{layer} source is average_l0_{neuronpedia_l0} — these "
            f"are different dictionaries, so feature indices are not "
            f"comparable. This is a hard failure and cannot be overridden."
        )

    return {"l0": dataset_l0, "l0Verified": True}
