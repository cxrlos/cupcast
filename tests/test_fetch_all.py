import pytest

from cupcast.v1.fetch import all as fetch_all


def test_run_all_isolates_failures_and_flags_required(monkeypatch, capsys):
    calls = []

    def ok(force):
        calls.append(("ok_step", force))

    def boom(force):
        raise OSError("network said no")

    steps = (
        fetch_all.Step("good_required", True, ok, None),
        fetch_all.Step("bad_optional", False, boom, None),
        fetch_all.Step("also_good", True, ok, None),
    )
    monkeypatch.setattr(fetch_all, "STEPS", steps)
    exit_code = fetch_all.run_all(force=True)
    out = capsys.readouterr().out
    assert exit_code == 0  # only an optional step failed
    assert calls == [("ok_step", True), ("ok_step", True)]
    assert "bad_optional" in out and "network said no" in out


def test_run_all_nonzero_when_required_step_fails(monkeypatch):
    def boom(force):
        raise OSError("blocked")

    steps = (fetch_all.Step("critical", True, boom, None),)
    monkeypatch.setattr(fetch_all, "STEPS", steps)
    assert fetch_all.run_all(force=False) == 1


def test_every_real_step_accepts_force():
    for step in fetch_all.STEPS:
        assert callable(step.run)
    names = [s.name for s in fetch_all.STEPS]
    assert names.index("understat") < names.index("football_data")
    required = {s.name for s in fetch_all.STEPS if s.required}
    assert required == {"api_football", "martj42", "squads", "understat"}


def test_doctor_runs_without_network_assumptions(monkeypatch, capsys):
    class FakeResponse:
        status_code = 200
        history = []
        url = "https://example.test/"

        def iter_content(self, n):
            yield b"Div,Date,HomeTeam"

        def close(self):
            pass

    monkeypatch.setattr(
        fetch_all.requests, "get", lambda *a, **k: FakeResponse(), raising=True
    )
    assert fetch_all.doctor() == 0
    out = capsys.readouterr().out
    assert "reachability" in out and "cache inventory" in out


@pytest.mark.parametrize("flag", ["--force", "--doctor"])
def test_cli_flags_parse(flag, monkeypatch):
    monkeypatch.setattr(fetch_all, "run_all", lambda force: 0)
    monkeypatch.setattr(fetch_all, "doctor", lambda: 0)
    monkeypatch.setattr("sys.argv", ["cupcast-fetch-all", flag])
    with pytest.raises(SystemExit) as excinfo:
        fetch_all.main()
    assert excinfo.value.code == 0
