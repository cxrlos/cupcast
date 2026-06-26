# Re-scaffold + v1 Freeze Implementation Plan


**Goal:** Freeze the existing Cupcast code under the `cupcast.v1` namespace (still fully runnable) and establish empty `cupcast.v2` and `cupcast.compare` namespaces with a clean-room guard, with the test suite staying green throughout.

**Architecture:** Pure mechanical refactor under green tests. All current modules move from `src/cupcast/<pkg>` to `src/cupcast/v1/<pkg>`; every internal `cupcast.X` import is rewritten to `cupcast.v1.X`; the Makefile's code targets are repointed. Then a minimal `cupcast.v2` package is created with a test enforcing that nothing under it ever imports `cupcast.v1`.

**Tech Stack:** Python ≥3.12, `uv`, pytest, ruff, hatchling.

## Global Constraints

- **NEVER run `git commit` or `git push`.** (Repo + user hard rule.) Each task ends by staging with `git add`; the user owns all commits. Wherever this plan says "stage", it means `git add` only.
- The package, paths, and pyproject name stay lowercase `cupcast`. ("Cupcast" capitalized only in prose.)
- `data/` is never committed; nothing personal in committed files.
- v2 code must **never import v1** (clean-room rule; enforced by a test in this plan).
- The existing test suite (60 tests) must remain green after every task.
- This plan does **not** move `docs/` — v1 keeps writing to `docs/tex/...`; the `docs/v1` relocation happens in the reports/papers plan (Plan 8) together with updating v1 report output paths, to avoid editing frozen v1 path constants prematurely.

---

### Task 1: Freeze existing code under `cupcast.v1`

**Files:**
- Move: `src/cupcast/run.py` → `src/cupcast/v1/run.py`
- Move: `src/cupcast/{features,fetch,model,ratings,report,sim,validate}/` → `src/cupcast/v1/{...}/`
- Create: `src/cupcast/v1/__init__.py`
- Modify: every `*.py` under `src/cupcast/v1/` and `tests/` (import rewrite)
- Modify: `Makefile` (code targets only)
- Unchanged: `src/cupcast/__init__.py` (empty top-level package marker)

**Interfaces:**
- Consumes: nothing.
- Produces: the `cupcast.v1.*` import namespace — every former `cupcast.X.Y` module is now importable as `cupcast.v1.X.Y` (e.g. `cupcast.v1.model.dixon_coles`, `cupcast.v1.sim.monte_carlo`, `cupcast.v1.run`). Later tasks/plans reference v1 only via `cupcast.v1.*`.

- [ ] **Step 1: Establish the green baseline**

Run: `uv run pytest -q`
Expected: `60 passed`. (If not 60, stop and reconcile before moving anything.)

- [ ] **Step 2: Move all modules under `src/cupcast/v1/`**

```bash
cd "$(git rev-parse --show-toplevel)"
mkdir -p src/cupcast/v1
git mv src/cupcast/run.py src/cupcast/v1/run.py
for d in features fetch model ratings report sim validate; do
  git mv "src/cupcast/$d" "src/cupcast/v1/$d"
done
```

- [ ] **Step 3: Add the v1 package marker**

Create `src/cupcast/v1/__init__.py`:

```python
"""Cupcast v1 — frozen baseline model. Kept runnable for the v1-vs-v2 comparison."""
```

- [ ] **Step 4: Rewrite internal imports to the `cupcast.v1` namespace**

Rewrites `from cupcast.` → `from cupcast.v1.` and `import cupcast.` → `import cupcast.v1.` across the moved code and the tests. The bare `import cupcast` (no dot) in `tests/test_smoke.py` is intentionally left untouched — the top-level package still resolves.

```bash
cd "$(git rev-parse --show-toplevel)"
grep -rlE '(from|import) cupcast\.' --include='*.py' src/cupcast/v1 tests \
  | xargs sed -i -E 's/(from |import )cupcast\./\1cupcast.v1./g'
```

- [ ] **Step 5: Repoint the Makefile code targets**

In `Makefile`, change the four module references in the `all-data`, `all-data-force`, `doctor`, and `forecast` targets from `cupcast.<x>` to `cupcast.v1.<x>`:

```make
all-data:
	uv run python -m cupcast.v1.fetch.all

all-data-force:
	uv run python -m cupcast.v1.fetch.all --force

doctor:
	uv run python -m cupcast.v1.fetch.all --doctor

forecast:
	uv run python -c 'from cupcast.v1.fetch.martj42 import fetch_results_csv; fetch_results_csv(refresh=True)'
	-uv run python -m cupcast.v1.fetch.odds
	uv run python -m cupcast.v1.report.build
	$(MAKE) docs
```

(The `docs` target is unchanged — it still points at `docs/tex/...`.)

- [ ] **Step 6: Verify the suite is still green after the move**

Run: `uv run pytest -q`
Expected: `60 passed`. (Same count as Step 1. A failure here means an import was missed — inspect the traceback's module path.)

- [ ] **Step 7: Verify v1 still imports and lints**

Run: `uv run python -c "import cupcast.v1.run; import cupcast.v1.sim.monte_carlo; import cupcast.v1.model.dixon_coles; print('v1 import OK')"`
Expected: `v1 import OK`

Run: `uv run ruff check src tests`
Expected: no errors (import sorting `I` is in the ruff selection; the rewrite preserves order, but if ruff reports `I001`, run `uv run ruff check --fix src tests` and re-run pytest).

- [ ] **Step 8: Stage (do NOT commit)**

```bash
cd "$(git rev-parse --show-toplevel)"
git add -A src/cupcast tests Makefile
git status
```

Leave the change staged for the user to commit. Do not run `git commit`.

---

### Task 2: Establish `cupcast.v2` and `cupcast.compare` namespaces with a clean-room guard

**Files:**
- Create: `src/cupcast/v2/__init__.py`
- Create: `src/cupcast/compare/__init__.py`
- Test: `tests/test_v2_namespace.py`

**Interfaces:**
- Consumes: `cupcast.v1.*` (only the guard test, which scans for forbidden imports).
- Produces: the importable `cupcast.v2` and `cupcast.compare` packages; the invariant test `tests/test_v2_namespace.py` that fails if any module under `src/cupcast/v2` imports `cupcast.v1`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_v2_namespace.py`:

```python
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parents[1] / "src" / "cupcast" / "v2"


def test_v2_package_imports():
    import cupcast.v2  # noqa: F401
    import cupcast.compare  # noqa: F401


def test_v2_is_clean_room():
    """v2 must never import v1 — the comparison baseline stays honest."""
    offenders = [
        str(path.relative_to(V2_ROOT))
        for path in V2_ROOT.rglob("*.py")
        if "cupcast.v1" in path.read_text()
    ]
    assert offenders == [], f"v2 modules import v1: {offenders}"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_v2_namespace.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cupcast.v2'`.

- [ ] **Step 3: Create the v2 and compare package markers**

Create `src/cupcast/v2/__init__.py`:

```python
"""Cupcast v2 — dynamic state-space model. Clean-room: never imports cupcast.v1."""
```

Create `src/cupcast/compare/__init__.py`:

```python
"""Shared v1-vs-v2 evaluation harness."""
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_v2_namespace.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Verify the full suite is green**

Run: `uv run pytest -q`
Expected: `61 passed` (60 prior + the new namespace file contributes 2 tests, so the count rises to 62 — confirm it is `62 passed`).

- [ ] **Step 6: Stage (do NOT commit)**

```bash
cd "$(git rev-parse --show-toplevel)"
git add -A src/cupcast tests
git status
```

Leave staged for the user. Do not run `git commit`.

---

## Self-Review

- **Spec coverage (§4 repo re-scaffold):** `src/cupcast/v1` (Task 1), `src/cupcast/v2` + `src/cupcast/compare` (Task 2), v2-never-imports-v1 guard (Task 2). The `docs/v1` + `docs/v2` split is intentionally deferred to Plan 8 (noted in Global Constraints) since relocating v1 docs requires touching v1 report output paths — done there alongside the v2 papers. All other §4 items are covered here.
- **Placeholder scan:** none — every step has exact commands/code.
- **Type/name consistency:** the produced namespace `cupcast.v1.*` is referenced consistently; the guard test path `src/cupcast/v2` matches the package created in Task 2 Step 3.
- **Count check:** Task 1 keeps the suite at 60; Task 2 adds one file with 2 tests → 62. Step 5 of Task 2 asserts `62 passed`.
