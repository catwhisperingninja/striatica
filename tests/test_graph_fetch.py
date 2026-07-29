# striatica/tests/test_graph_fetch.py
"""Tests for pipeline/graph_fetch.py — Neuronpedia attribution-graph fetching.

TARGET API (frozen by this test file; implementers make these pass, they do
not change them):

    # pipeline/graph_fetch.py
    NEURONPEDIA_BASE = "https://www.neuronpedia.org"
    GRAPHS_CACHE_DIR = <project>/data/graphs        # == pipeline.config.DATA_DIR / "graphs"

    def load_neuronpedia_api_key() -> str | None
        # dotenv from project root .env (os.environ wins over .env, the
        # standard load_dotenv behavior). NEVER prints/logs the key value.

    def fetch_graph_record(model_id: str, slug: str, timeout: int = 30) -> dict
        # GET {NEURONPEDIA_BASE}/api/graph/{model_id}/{slug}  (keyless — the
        # endpoint is public; x-api-key must NEVER be sent even if set in env).
        # HTTP 404 -> RuntimeError whose message names the slug.

    def fetch_graph(model_id: str, slug: str, cache_dir=GRAPHS_CACHE_DIR,
                    force: bool = False) -> dict
        # Two-step: record -> record["url"] (S3) -> full graph JSON.
        # Caches the FULL graph at {cache_dir}/{model_id}/{slug}.json.
        # Cache hit (file exists, not force) -> zero network calls.
        # force=True -> refetch both steps and rewrite the cache.
        # Malformed JSON from the network -> raise (RuntimeError or
        # json.JSONDecodeError); nothing is written to the cache.

    def fetch_source_set(model_id: str, name: str, timeout: int = 30) -> dict
        # GET {NEURONPEDIA_BASE}/api/source-set/{model_id}/{name}  (keyless).

    def generate_graph(model_id: str, prompt: str, slug: str,
                       api_key: str | None, **params) -> dict
        # POST {NEURONPEDIA_BASE}/api/graph/generate with x-api-key header.
        # Body: prompt, slug, model id, plus **params verbatim (maxNLogits,
        # desiredLogitProb, maxFeatureNodes, sourceSetName, ...).
        # api_key missing/None -> RuntimeError BEFORE any network call,
        # explaining fetch-only mode and .env NEURONPEDIA_API_KEY.
        # HTTP 429 -> RuntimeError with 30-graphs-per-hour guidance, NO retry.

    def validate_graph_identity(record: dict, graph: dict, source_set: dict,
                                dataset_metadata: dict,
                                allow_l0_mismatch: bool = False) -> dict
        # Three-state identity checks (verified / contradicted / unverifiable):
        #   - model: record["modelId"] AND graph["metadata"]["scan"] must match
        #     dataset_metadata["model_id"]  -> ValueError on mismatch
        #   - source set: must be the gemmascope-transcoder-16k family
        #     -> ValueError otherwise
        #   - dataset: num_features must be 16384 and transcoder width
        #     "width_16k" -> ValueError otherwise
        #   - L0: parse layer_{L}/width_16k/average_l0_{N} from source_set
        #     sources for the dataset layer; compare N to
        #     dataset_metadata["transcoder"]["l0_variant"].
        #       CONTRADICTED -> ValueError quoting both sides, ALWAYS (even
        #         with allow_l0_mismatch=True). The real Neuronpedia layer-12
        #         source is average_l0_6 while the local dataset is l0_604 —
        #         a different dictionary; this pairing must hard-fail.
        #       UNVERIFIABLE (no source for the layer) -> ValueError unless
        #         allow_l0_mismatch=True, then return {"l0Verified": False}.
        #       VERIFIED -> return dict with {"l0": N, "l0Verified": True}.
        # On success returns a dict carrying at least "l0" (int, the dataset
        # l0_variant) and "l0Verified" (bool) for the frozen circuit schema.

Transport is monkeypatched at urllib.request.urlopen (repo convention:
pipeline/download.py, pipeline/discovery.py use `import urllib.request` +
`urllib.request.urlopen(...)`; the implementation must call it module-qualified
so these tests can intercept it). No test touches the real network — the
fixture record's S3 url is on example.invalid on purpose.

Fixtures (real Neuronpedia data, labels neutralized, clerp blanked):
  tests/fixtures/neuronpedia_graph_record_gemma.json   — graph record (step 1)
  tests/fixtures/neuronpedia_graph_gemma.json          — full graph (step 2)
  tests/fixtures/neuronpedia_sourceset_gemma.json      — source set (L0 identity)
  frontend/public/data/gemma-2-2b-layer12-l0604-metadata.json — local dataset
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections import namedtuple
from pathlib import Path

import pytest

from pipeline import graph_fetch
from pipeline.config import DATA_DIR

# ---------------------------------------------------------------------------
# Fixture data (loaded fresh per call so tests can mutate copies safely)
# ---------------------------------------------------------------------------

TESTS_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
PROJECT_ROOT = TESTS_DIR.parent
DATASET_METADATA_PATH = (
    PROJECT_ROOT / "frontend" / "public" / "data"
    / "gemma-2-2b-layer12-l0604-metadata.json"
)

MODEL = "gemma-2-2b"
SLUG = "gemma-fact-dallas-austin"
SOURCE_SET_NAME = "gemmascope-transcoder-16k"

RECORD_URL = f"https://www.neuronpedia.org/api/graph/{MODEL}/{SLUG}"
# record["url"] in the fixture — deliberately on example.invalid so a bug that
# escapes the monkeypatch can never fetch anything real.
S3_URL = "https://example.invalid/user-graphs/test/gemma-fact-dallas-austin.json"
SOURCE_SET_URL = (
    f"https://www.neuronpedia.org/api/source-set/{MODEL}/{SOURCE_SET_NAME}"
)
GENERATE_URL = "https://www.neuronpedia.org/api/graph/generate"


def _fixture_bytes(name: str) -> bytes:
    return (FIXTURES_DIR / name).read_bytes()


def _fixture(name: str) -> dict:
    return json.loads(_fixture_bytes(name))


def _record() -> dict:
    return _fixture("neuronpedia_graph_record_gemma.json")


def _graph() -> dict:
    return _fixture("neuronpedia_graph_gemma.json")


def _source_set() -> dict:
    return _fixture("neuronpedia_sourceset_gemma.json")


def _dataset_metadata() -> dict:
    return json.loads(DATASET_METADATA_PATH.read_text())


def _source_set_l0_604() -> dict:
    """Hypothetical source set whose layer-12 dictionary matches the local
    dataset (l0=604). Derived from the real fixture by substituting only the
    l0 suffix of the layer_12 source — everything else stays real."""
    ss = _source_set()
    for src in ss["sources"]:
        if src["saelensSaeId"].startswith("layer_12/"):
            src["saelensSaeId"] = "layer_12/width_16k/average_l0_604"
            src["hfFolderId"] = "layer_12/width_16k/average_l0_604"
    return ss


def _source_set_without_layer12() -> dict:
    """Source set with no layer-12 source at all -> L0 is unverifiable."""
    ss = _source_set()
    ss["sources"] = [
        s for s in ss["sources"]
        if not s["saelensSaeId"].startswith("layer_12/")
    ]
    return ss


# ---------------------------------------------------------------------------
# Fake transport
# ---------------------------------------------------------------------------

_Call = namedtuple("_Call", "url method headers body")


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload
        self.status = 200

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeTransport:
    """Stands in for urllib.request.urlopen.

    routes: url -> bytes payload to return, or an Exception instance to raise.
    Every call is recorded as _Call(url, method, lowercased-headers, body).
    Any URL not in routes raises AssertionError (unexpected network call).
    """

    def __init__(self, routes: dict | None = None):
        self.routes = dict(routes or {})
        self.calls: list[_Call] = []

    def __call__(self, url_or_request, timeout=None, **kwargs):
        if isinstance(url_or_request, urllib.request.Request):
            req = url_or_request
            url = req.full_url
            method = req.get_method()
            headers = {k.lower(): v for k, v in req.header_items()}
            body = req.data
        else:
            url = url_or_request
            method = "GET"
            headers = {}
            body = None
        self.calls.append(_Call(url, method, headers, body))
        if url not in self.routes:
            raise AssertionError(f"Unexpected network call to {url}")
        result = self.routes[url]
        if isinstance(result, Exception):
            raise result
        return _FakeResponse(result)


def _install(monkeypatch, routes: dict | None = None) -> _FakeTransport:
    transport = _FakeTransport(routes)
    monkeypatch.setattr("urllib.request.urlopen", transport)
    return transport


def _http_error(url: str, code: int, msg: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, msg, hdrs=None, fp=None)


def _assert_no_api_key_sent(transport: _FakeTransport, sentinel: str) -> None:
    assert transport.calls, "expected at least one network call to inspect"
    for call in transport.calls:
        assert "x-api-key" not in call.headers, (
            f"x-api-key header sent on keyless endpoint {call.url}"
        )
        assert sentinel not in str(call.headers)
        assert call.body is None or sentinel not in call.body.decode(
            "utf-8", errors="replace"
        )


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_neuronpedia_base_constant(self):
        assert graph_fetch.NEURONPEDIA_BASE == "https://www.neuronpedia.org"

    def test_graphs_cache_dir_constant(self):
        assert Path(graph_fetch.GRAPHS_CACHE_DIR) == DATA_DIR / "graphs"


# ---------------------------------------------------------------------------
# load_neuronpedia_api_key
# ---------------------------------------------------------------------------

class TestLoadNeuronpediaApiKey:
    def test_env_var_wins_over_dotenv(self, monkeypatch):
        """os.environ takes precedence (standard load_dotenv semantics), so a
        set env var must be returned verbatim regardless of .env contents."""
        monkeypatch.setenv("NEURONPEDIA_API_KEY", "test-key-from-env-123")
        assert graph_fetch.load_neuronpedia_api_key() == "test-key-from-env-123"

    def test_key_value_never_printed(self, monkeypatch, capsys):
        """The key value must never hit stdout/stderr."""
        sentinel = "sk-SENTINEL-must-never-be-logged-9f2a"
        monkeypatch.setenv("NEURONPEDIA_API_KEY", sentinel)
        graph_fetch.load_neuronpedia_api_key()
        captured = capsys.readouterr()
        assert sentinel not in captured.out
        assert sentinel not in captured.err


# ---------------------------------------------------------------------------
# fetch_graph_record
# ---------------------------------------------------------------------------

class TestFetchGraphRecord:
    def test_returns_record_dict_from_api(self, monkeypatch):
        transport = _install(monkeypatch, {
            RECORD_URL: _fixture_bytes("neuronpedia_graph_record_gemma.json"),
        })
        record = graph_fetch.fetch_graph_record(MODEL, SLUG)
        assert record == _record()
        assert record["modelId"] == "gemma-2-2b"
        assert record["sourceSetName"] == "gemmascope-transcoder-16k"
        assert record["url"] == S3_URL
        assert len(transport.calls) == 1
        assert transport.calls[0].url == RECORD_URL

    def test_404_error_names_slug(self, monkeypatch):
        _install(monkeypatch, {
            RECORD_URL: _http_error(RECORD_URL, 404, "Not Found"),
        })
        with pytest.raises(RuntimeError) as excinfo:
            graph_fetch.fetch_graph_record(MODEL, SLUG)
        # Actionable: the message must name the slug that was not found.
        assert SLUG in str(excinfo.value)

    def test_never_sends_api_key_even_when_env_set(self, monkeypatch):
        sentinel = "sk-SENTINEL-do-not-send-1b7c"
        monkeypatch.setenv("NEURONPEDIA_API_KEY", sentinel)
        transport = _install(monkeypatch, {
            RECORD_URL: _fixture_bytes("neuronpedia_graph_record_gemma.json"),
        })
        graph_fetch.fetch_graph_record(MODEL, SLUG)
        _assert_no_api_key_sent(transport, sentinel)


# ---------------------------------------------------------------------------
# fetch_graph (two-step + cache)
# ---------------------------------------------------------------------------

class TestFetchGraph:
    def test_two_step_fetch_returns_full_graph_and_caches(
        self, monkeypatch, tmp_path
    ):
        transport = _install(monkeypatch, {
            RECORD_URL: _fixture_bytes("neuronpedia_graph_record_gemma.json"),
            S3_URL: _fixture_bytes("neuronpedia_graph_gemma.json"),
        })
        graph = graph_fetch.fetch_graph(MODEL, SLUG, cache_dir=tmp_path)

        expected = _graph()
        assert graph == expected
        assert graph["metadata"]["scan"] == "gemma-2-2b"
        assert graph["metadata"]["slug"] == SLUG
        assert len(graph["nodes"]) == 53
        assert len(graph["links"]) == 250

        # Exactly two calls, record first then the S3 url from the record.
        assert [c.url for c in transport.calls] == [RECORD_URL, S3_URL]

        # Full graph cached at {cache_dir}/{model_id}/{slug}.json
        cache_file = tmp_path / MODEL / f"{SLUG}.json"
        assert cache_file.is_file()
        assert json.loads(cache_file.read_text()) == expected

    def test_cache_hit_makes_no_network_calls(self, monkeypatch, tmp_path):
        cache_file = tmp_path / MODEL / f"{SLUG}.json"
        cache_file.parent.mkdir(parents=True)
        cache_file.write_bytes(_fixture_bytes("neuronpedia_graph_gemma.json"))

        # Empty routes: ANY network call raises AssertionError.
        transport = _install(monkeypatch, {})
        graph = graph_fetch.fetch_graph(MODEL, SLUG, cache_dir=tmp_path)

        assert graph == _graph()
        assert transport.calls == []

    def test_force_refetches_over_stale_cache(self, monkeypatch, tmp_path):
        # Seed the cache with valid-but-wrong real JSON (the record fixture),
        # so we can prove the returned graph came from the network.
        cache_file = tmp_path / MODEL / f"{SLUG}.json"
        cache_file.parent.mkdir(parents=True)
        cache_file.write_bytes(
            _fixture_bytes("neuronpedia_graph_record_gemma.json")
        )

        transport = _install(monkeypatch, {
            RECORD_URL: _fixture_bytes("neuronpedia_graph_record_gemma.json"),
            S3_URL: _fixture_bytes("neuronpedia_graph_gemma.json"),
        })
        graph = graph_fetch.fetch_graph(
            MODEL, SLUG, cache_dir=tmp_path, force=True
        )

        expected = _graph()
        assert graph == expected  # network content, not the stale cache
        assert [c.url for c in transport.calls] == [RECORD_URL, S3_URL]
        # Cache rewritten with the fresh full graph.
        assert json.loads(cache_file.read_text()) == expected

    def test_malformed_json_is_not_cached(self, monkeypatch, tmp_path):
        # Truncated real payload — guaranteed-invalid JSON.
        truncated = _fixture_bytes("neuronpedia_graph_gemma.json")[:200]
        _install(monkeypatch, {
            RECORD_URL: _fixture_bytes("neuronpedia_graph_record_gemma.json"),
            S3_URL: truncated,
        })
        # json.JSONDecodeError subclasses ValueError; a wrapped RuntimeError
        # is equally acceptable.
        with pytest.raises((RuntimeError, ValueError)):
            graph_fetch.fetch_graph(MODEL, SLUG, cache_dir=tmp_path)

        cache_file = tmp_path / MODEL / f"{SLUG}.json"
        assert not cache_file.exists(), "malformed payload must not be cached"

    def test_never_sends_api_key_even_when_env_set(self, monkeypatch, tmp_path):
        sentinel = "sk-SENTINEL-do-not-send-44e0"
        monkeypatch.setenv("NEURONPEDIA_API_KEY", sentinel)
        transport = _install(monkeypatch, {
            RECORD_URL: _fixture_bytes("neuronpedia_graph_record_gemma.json"),
            S3_URL: _fixture_bytes("neuronpedia_graph_gemma.json"),
        })
        graph_fetch.fetch_graph(MODEL, SLUG, cache_dir=tmp_path)
        _assert_no_api_key_sent(transport, sentinel)


# ---------------------------------------------------------------------------
# fetch_source_set
# ---------------------------------------------------------------------------

class TestFetchSourceSet:
    def test_returns_source_set_with_l0_identity(self, monkeypatch):
        transport = _install(monkeypatch, {
            SOURCE_SET_URL: _fixture_bytes("neuronpedia_sourceset_gemma.json"),
        })
        ss = graph_fetch.fetch_source_set(MODEL, SOURCE_SET_NAME)

        assert ss == _source_set()
        assert transport.calls[0].url == SOURCE_SET_URL
        # The L0-identity payload this whole unit exists for:
        by_id = {s["id"]: s for s in ss["sources"]}
        assert (
            by_id["12-gemmascope-transcoder-16k"]["saelensSaeId"]
            == "layer_12/width_16k/average_l0_6"
        )
        assert (
            by_id["12-gemmascope-transcoder-16k"]["hfRepoId"]
            == "google/gemma-scope-2b-pt-transcoders"
        )

    def test_never_sends_api_key(self, monkeypatch):
        sentinel = "sk-SENTINEL-do-not-send-77aa"
        monkeypatch.setenv("NEURONPEDIA_API_KEY", sentinel)
        transport = _install(monkeypatch, {
            SOURCE_SET_URL: _fixture_bytes("neuronpedia_sourceset_gemma.json"),
        })
        graph_fetch.fetch_source_set(MODEL, SOURCE_SET_NAME)
        _assert_no_api_key_sent(transport, sentinel)


# ---------------------------------------------------------------------------
# generate_graph
# ---------------------------------------------------------------------------

class TestGenerateGraph:
    PROMPT = "Fact: The capital of the state containing Dallas is"

    def test_missing_api_key_explains_fetch_only_mode(self, monkeypatch):
        transport = _install(monkeypatch, {})  # any network call would raise
        with pytest.raises(RuntimeError) as excinfo:
            graph_fetch.generate_graph(
                MODEL, prompt=self.PROMPT, slug="my-new-graph", api_key=None
            )
        msg = str(excinfo.value)
        assert "NEURONPEDIA_API_KEY" in msg
        assert ".env" in msg
        assert "fetch" in msg.lower()  # explains fetch-only mode
        assert transport.calls == []  # failed BEFORE any network call

    def test_429_returns_rate_limit_guidance_without_retry(self, monkeypatch):
        transport = _install(monkeypatch, {
            GENERATE_URL: _http_error(GENERATE_URL, 429, "Too Many Requests"),
        })
        with pytest.raises(RuntimeError) as excinfo:
            graph_fetch.generate_graph(
                MODEL,
                prompt=self.PROMPT,
                slug="my-new-graph",
                api_key="test-key-abc",
            )
        msg = str(excinfo.value)
        assert "30" in msg  # 30 graphs / hour rate limit guidance
        assert "hour" in msg.lower() or "hr" in msg.lower()
        assert len(transport.calls) == 1, "429 must not be retried"

    def test_posts_with_api_key_and_params(self, monkeypatch):
        transport = _install(monkeypatch, {
            GENERATE_URL: _fixture_bytes(
                "neuronpedia_graph_record_gemma.json"
            ),
        })
        result = graph_fetch.generate_graph(
            MODEL,
            prompt=self.PROMPT,
            slug="my-new-graph",
            api_key="test-key-abc",
            maxNLogits=10,
        )
        assert result == _record()

        assert len(transport.calls) == 1
        call = transport.calls[0]
        assert call.url == GENERATE_URL
        assert call.method == "POST"
        assert call.headers.get("x-api-key") == "test-key-abc"

        body = json.loads(call.body)
        assert body["prompt"] == self.PROMPT
        assert body["slug"] == "my-new-graph"
        assert MODEL in body.values()  # model id included in the POST body
        assert body["maxNLogits"] == 10  # **params passed through verbatim


# ---------------------------------------------------------------------------
# validate_graph_identity
# ---------------------------------------------------------------------------

class TestValidateGraphIdentity:
    def test_l0_match_passes_and_reports_verified(self):
        """Real dataset metadata + hypothetical l0=604-matching source set."""
        result = graph_fetch.validate_graph_identity(
            _record(), _graph(), _source_set_l0_604(), _dataset_metadata()
        )
        assert result["l0"] == 604
        assert result["l0Verified"] is True

    def test_l0_contradiction_604_vs_6_hard_fails(self):
        """Fixtures as-is: Neuronpedia layer-12 = average_l0_6, the local
        dataset = l0_604. Different dictionaries — must hard-fail."""
        with pytest.raises(ValueError) as excinfo:
            graph_fetch.validate_graph_identity(
                _record(), _graph(), _source_set(), _dataset_metadata()
            )
        msg = str(excinfo.value)
        # Must quote both sides: the dataset's 604 ...
        assert "604" in msg
        # ... and Neuronpedia's 6 as a standalone number (not the 6 in 604,
        # 16k, or 16384).
        assert re.search(r"(?<!\d)6(?!\d)", msg), msg

    def test_l0_contradiction_not_rescued_by_allow_flag(self):
        """CONTRADICTED is always a hard error — allow_l0_mismatch only covers
        the unverifiable case, never a proven mismatch."""
        with pytest.raises(ValueError):
            graph_fetch.validate_graph_identity(
                _record(),
                _graph(),
                _source_set(),
                _dataset_metadata(),
                allow_l0_mismatch=True,
            )

    def test_l0_unverifiable_hard_fails_without_flag(self):
        with pytest.raises(ValueError):
            graph_fetch.validate_graph_identity(
                _record(),
                _graph(),
                _source_set_without_layer12(),
                _dataset_metadata(),
            )

    def test_l0_unverifiable_allowed_reports_unverified(self):
        result = graph_fetch.validate_graph_identity(
            _record(),
            _graph(),
            _source_set_without_layer12(),
            _dataset_metadata(),
            allow_l0_mismatch=True,
        )
        assert result["l0Verified"] is False
        # Frozen schema requires metadata.l0:int — the dataset side is the
        # only known value when Neuronpedia's is unverifiable.
        assert result["l0"] == 604

    def test_wrong_record_model_fails(self):
        record = _record()
        record["modelId"] = "gpt2-small"
        with pytest.raises(ValueError) as excinfo:
            graph_fetch.validate_graph_identity(
                record, _graph(), _source_set_l0_604(), _dataset_metadata()
            )
        assert "gpt2-small" in str(excinfo.value)

    def test_wrong_graph_scan_fails(self):
        graph = _graph()
        graph["metadata"]["scan"] = "gpt2-small"
        with pytest.raises(ValueError) as excinfo:
            graph_fetch.validate_graph_identity(
                _record(), graph, _source_set_l0_604(), _dataset_metadata()
            )
        assert "gpt2-small" in str(excinfo.value)

    def test_wrong_source_set_family_fails(self):
        # Perturb both places the name appears so the test holds whichever
        # field the implementation checks.
        record = _record()
        record["sourceSetName"] = "gemmascope-res-16k"
        source_set = _source_set_l0_604()
        source_set["name"] = "gemmascope-res-16k"
        with pytest.raises(ValueError) as excinfo:
            graph_fetch.validate_graph_identity(
                record, _graph(), source_set, _dataset_metadata()
            )
        assert "gemmascope-res-16k" in str(excinfo.value)

    def test_wrong_dataset_width_fails(self):
        dataset = _dataset_metadata()
        dataset["transcoder"]["width"] = "width_65k"
        with pytest.raises(ValueError) as excinfo:
            graph_fetch.validate_graph_identity(
                _record(), _graph(), _source_set_l0_604(), dataset
            )
        assert "width" in str(excinfo.value).lower()

    def test_wrong_dataset_num_features_fails(self):
        dataset = _dataset_metadata()
        dataset["num_features"] = 65536
        with pytest.raises(ValueError) as excinfo:
            graph_fetch.validate_graph_identity(
                _record(), _graph(), _source_set_l0_604(), dataset
            )
        msg = str(excinfo.value)
        assert "16384" in msg or "65536" in msg
