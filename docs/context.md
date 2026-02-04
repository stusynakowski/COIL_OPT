# Project Context (Agent Entry Point)

## Goal
Build this project using a phased workflow: Idea → Specs → Dev Plan → Tests → Implementation.

## Current phase
Set this manually before using an agent:
- Phase: IDEA | SPEC | DEV_PLAN | TESTS | IMPL | STATUS

## Phase edit rules (hard guardrails)
- IDEA: may edit only `docs/introduction.md` and `docs/context.md`
- SPEC: may edit only `docs/spec/**`, `docs/adr/**`, and `docs/context.md`
- DEV_PLAN: may edit only `docs/dev_plan/**`, `docs/dev_notes/**`, and `docs/context.md`
- TESTS: may edit only `tests/**`, `docs/test_plan/**`, and `docs/context.md`
- IMPL: may edit only `src/**`, `docs/status_update/**` and `docs/context.md` (plus explicitly allowed config files if needed)
- STATUS: may edit only `docs/status_update/**` and `docs/context.md`

## Canonical docs
- Introduction: `docs/introduction.md`
- Specs: `docs/spec/`
- Dev Plan: `docs/dev_plan/`
- Dev Notes: `docs/dev_notes/`
- Status: `docs/status_update/`
- Test plans: `docs/test_plan/`
- ADRs: `docs/adr/`

## Commands
- Install (dev): `python -m pip install -e ".[dev]"`
- Run tests: `pytest -q`
- Run guardrails: `PHASE=spec ./scripts/verify_phase.sh`

## Conventions
- Every requirement in specs MUST have an ID like `REQ-CORE-001`
- Every test references at least one REQ id in a comment/docstring
- Avoid adding new dependencies unless required by the spec

