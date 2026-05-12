import pytest
from uuid import UUID, uuid4

from src.data.load import DepBody
from src.env.id_mapping import IDMapper, MatchResult, normalize
from src.env.search_client import SearchResult


def _dep_body(uid: UUID, body: str) -> DepBody:
    return DepBody(statement_id=uid, body=body, kind="theorem", paper_id="paper-1")


def _search_result(body: str) -> SearchResult:
    return SearchResult(
        theorem_id=1, slogan_id=1, name="test", body=body,
        slogan="", theorem_type="theorem", link="", similarity=0.9, paper="paper-1",
    )


# --- Normalizer ---

def test_normalize_strips_label():
    assert r"\label{foo}" not in normalize(r"\label{foo} some body")


def test_normalize_collapses_whitespace():
    assert normalize("a  b\t\nc") == "a b c"


def test_normalize_strips_trailing_period():
    assert normalize("theorem.") == "theorem"


def test_normalize_unescape_html():
    assert normalize("a &amp; b") == "a & b"


def test_normalize_no_lowercase():
    assert normalize("\\Theta") != normalize("\\theta")


def test_normalize_idempotent():
    samples = [
        r"\label{thm:main} Let $G$ be a group.",
        r"Every finite group of order $p^2$ is abelian.",
        r"$\Theta \neq \theta$ &amp; more text.  ",
        "",
        "a" * 500,
    ]
    for s in samples:
        assert normalize(normalize(s)) == normalize(s), f"Not idempotent on: {s!r}"


# --- Mapper ---

def test_no_match_synthetic():
    uid = uuid4()
    dep_bodies = {uid: _dep_body(uid, "Every compact Hausdorff space is normal.")}
    mapper = IDMapper(dep_bodies, threshold=85.0)
    result = mapper.map_int_to_uuid(_search_result("this is not a real theorem xyz"))
    assert result.uuid is None


def test_exact_match():
    uid = uuid4()
    body = "Every finite group of order $p^2$ is abelian."
    dep_bodies = {uid: _dep_body(uid, body)}
    mapper = IDMapper(dep_bodies, threshold=85.0)
    result = mapper.map_int_to_uuid(_search_result(body))
    assert result.uuid == uid
    assert result.score >= 99.0


def test_near_match_above_threshold():
    uid = uuid4()
    body = "Every finite group of order $p^2$ is abelian."
    near = "Every finite group of order $p^2$ is abelian"  # missing period
    dep_bodies = {uid: _dep_body(uid, body)}
    mapper = IDMapper(dep_bodies, threshold=85.0)
    result = mapper.map_int_to_uuid(_search_result(near))
    assert result.uuid == uid


def test_below_threshold_returns_none():
    uid = uuid4()
    dep_bodies = {uid: _dep_body(uid, "Every compact Hausdorff space is normal.")}
    mapper = IDMapper(dep_bodies, threshold=85.0)
    # Only 50% similar body
    result = mapper.map_int_to_uuid(_search_result("The snake lemma holds in abelian categories."))
    assert result.uuid is None


def test_correct_uuid_among_many():
    uids = [uuid4() for _ in range(20)]
    target_uid = uids[7]
    target_body = "The center of a non-trivial p-group is non-trivial."
    dep_bodies = {
        uid: _dep_body(uid, f"Some unrelated statement number {i}.")
        for i, uid in enumerate(uids)
    }
    dep_bodies[target_uid] = _dep_body(target_uid, target_body)
    mapper = IDMapper(dep_bodies, threshold=85.0)
    result = mapper.map_int_to_uuid(_search_result(target_body))
    assert result.uuid == target_uid


def test_empty_dep_universe():
    mapper = IDMapper({}, threshold=85.0)
    result = mapper.map_int_to_uuid(_search_result("any body"))
    assert result.uuid is None


def test_low_confidence_gap_flagged():
    uid1, uid2 = uuid4(), uuid4()
    # Two very similar bodies — gap will be small
    body1 = "Every finite group of order $p^2$ is abelian."
    body2 = "Every finite group of order $p^2$ is abelian!"
    dep_bodies = {
        uid1: _dep_body(uid1, body1),
        uid2: _dep_body(uid2, body2),
    }
    mapper = IDMapper(dep_bodies, threshold=85.0, low_confidence_gap=10.0)
    result = mapper.map_int_to_uuid(_search_result(body1))
    # Should match but with a small gap (both bodies are nearly identical)
    assert result.uuid is not None
    assert result.second_best_gap < 10.0
