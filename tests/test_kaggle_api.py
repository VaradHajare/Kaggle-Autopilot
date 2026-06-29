"""Kaggle client wrapper against the fake API."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.tools.kaggle_api import (
    KaggleClient,
    _as_list,
    _attr,
    _parse_reward,
    parse_competition_slug,
)
from tests.conftest import FakeKaggleApi


class _Kaggle2xApi:
    """Stand-in for the kaggle>=2.0 client: list calls return wrapped response
    objects (payload under an attribute) and competition fields are snake_case."""

    def __init__(self, slug: str = "titanic") -> None:
        self._comp = SimpleNamespace(
            ref=f"https://www.kaggle.com/competitions/{slug}",
            evaluation_metric="Categorization Accuracy",
            category="Getting Started",
            reward="Knowledge",
            max_daily_submissions=10,
            max_team_size=10,
            deadline=None,
            user_has_entered=False,  # unreliable for Getting-Started comps
        )
        self._files = SimpleNamespace(
            files=[SimpleNamespace(name="train.csv", total_bytes=61194)]
        )
        self.submitted: list[tuple] = []

    def authenticate(self) -> None: ...

    def competitions_list(self, *args, **kwargs):
        return SimpleNamespace(competitions=[self._comp])

    def competition_list_files(self, competition):
        return self._files

    def competition_download_files(self, competition, path): ...

    def competition_submissions(self, competition):
        return SimpleNamespace(
            submissions=[SimpleNamespace(public_score="0.77")]
        )

    def competition_submit(self, file_name, message, competition):
        self.submitted.append((file_name, message, competition))
        return SimpleNamespace(status="complete")


@pytest.mark.parametrize(
    "value,expected",
    [
        ("https://www.kaggle.com/competitions/titanic", "titanic"),
        ("https://www.kaggle.com/c/titanic/", "titanic"),
        ("https://www.kaggle.com/competitions/spaceship-titanic?foo=1", "spaceship-titanic"),
        ("titanic", "titanic"),
    ],
)
def test_parse_slug(value, expected):
    assert parse_competition_slug(value) == expected


def test_parse_slug_rejects_garbage():
    with pytest.raises(ValueError):
        parse_competition_slug("not a url with spaces/and/slashes")


def test_check_credentials_ok(kaggle_client):
    assert kaggle_client.check_credentials() is True


def test_check_credentials_failure():
    client = KaggleClient(api=FakeKaggleApi(auth_ok=False))
    assert client.check_credentials() is False


def test_get_metadata(kaggle_client):
    meta = kaggle_client.get_competition_metadata("demo-comp")
    assert meta.eval_metric == "AUC"
    assert meta.daily_submission_limit == 5


def test_rules_accepted_true(kaggle_client):
    assert kaggle_client.check_rules_accepted("demo-comp") is True


def test_rules_accepted_false():
    client = KaggleClient(api=FakeKaggleApi(rules_accepted=False))
    assert client.check_rules_accepted("demo-comp") is False


def test_total_file_size(kaggle_client):
    assert kaggle_client.get_total_file_size("demo-comp") == 1_000


def test_submit_records_call():
    api = FakeKaggleApi()
    client = KaggleClient(api=api)
    client.submit("demo-comp", "sub.csv", "msg")
    assert api.submitted == [("sub.csv", "msg", "demo-comp")]


@pytest.mark.parametrize("reward,expected", [("$25,000", 25000.0), ("Knowledge", 0.0), (None, 0.0)])
def test_parse_reward(reward, expected):
    assert _parse_reward(reward) == expected


# --- kaggle >=2.0 compatibility -------------------------------------------------

def test_as_list_unwraps_response_object_and_bare_list():
    assert _as_list([1, 2]) == [1, 2]
    assert _as_list(None) == []
    wrapped = SimpleNamespace(competitions=["a", "b"])
    assert _as_list(wrapped) == ["a", "b"]


def test_attr_prefers_first_present_name():
    obj = SimpleNamespace(evaluation_metric="acc")
    assert _attr(obj, "evaluation_metric", "evaluationMetric") == "acc"
    legacy = SimpleNamespace(evaluationMetric="auc")
    assert _attr(legacy, "evaluation_metric", "evaluationMetric") == "auc"
    assert _attr(legacy, "missing", default="x") == "x"


def test_metadata_reads_kaggle_2x_snake_case_fields():
    client = KaggleClient(api=_Kaggle2xApi())
    meta = client.get_competition_metadata("titanic")
    assert meta.eval_metric == "Categorization Accuracy"
    assert meta.daily_submission_limit == 10
    assert meta.team_size_limit == 10


def test_total_file_size_unwraps_2x_files_response():
    client = KaggleClient(api=_Kaggle2xApi())
    assert client.get_total_file_size("titanic") == 61194


def test_rules_accepted_falls_back_to_file_probe_when_flag_false():
    # 2.x reports user_has_entered=False for Getting-Started comps, but the
    # file listing succeeds -> rules are effectively accepted.
    client = KaggleClient(api=_Kaggle2xApi())
    assert client.check_rules_accepted("titanic") is True


def test_latest_score_reads_2x_public_score():
    api = _Kaggle2xApi()
    client = KaggleClient(api=api)
    assert client.latest_submission_score("titanic") == pytest.approx(0.77)
