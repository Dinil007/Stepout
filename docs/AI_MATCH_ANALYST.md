# StepOut AI Match Analyst

## Scope

The AI Match Analyst consumes structured analytics exports only. It does not read
raw video frames and does not modify the existing computer vision pipeline.

Primary package:

```text
app/ai/
  __init__.py
  aggregator.py
  insight_engine.py
  match_analyst.py
  prompt_builder.py
  recommendations.py
  report_generator.py
  schemas.py
```

## Provider Configuration

The analyst uses a provider abstraction so analytics, API, and dashboard code do
not depend on a specific LLM vendor.

Environment variables:

```text
STEPOUT_LLM_PROVIDER=openai | gemini
OPENAI_API_KEY=<key>
OPENAI_MODEL=gpt-4o-mini
GEMINI_API_KEY=<key>
GEMINI_MODEL=gemini-1.5-flash
STEPOUT_MATCH_ID=<optional-match-id>
```

If no remote key is configured, the platform uses the deterministic offline
provider. This keeps tests, demos, FastAPI, and Streamlit usable without network
access.

## FastAPI Endpoints

```text
GET  /ai/match-summary
GET  /ai/team-analysis
GET  /ai/player/{player_id}
GET  /ai/coach-report
POST /ai/query
GET  /ai/recommendations
```

Swagger updates automatically through `app/api/main.py`.

Example query:

```json
{
  "question": "Who should be Player of the Match?"
}
```

## Streamlit

The main `streamlit_app.py` navigation includes `AI Match Analyst` with:

- Match Summary
- Team Analysis
- Player Reports with player dropdown
- Coach Report
- Opposition Analysis
- Ask AI
- Download buttons

## Generated Reports

Reports are exported to `outputs/`:

```text
ai_match_summary.md
ai_match_summary.pdf
ai_team_report.json
ai_player_reports.json
coach_report.md
opposition_report.md
recommendations.json
ai_validation_report.json
ai_performance_report.json
```

## Validation Guarantees

`ai_validation_report.json` checks:

- AI uses generated analytics only.
- Raw video analysis is not used.
- Player IDs are constrained to detected/exported IDs.
- Ratings exist for every detected player in the aggregated context.
- FastAPI and Streamlit use the same `MatchAnalyst` implementation.

## Regression Note

The implementation is additive except for API/dashboard integration. No existing
computer vision modules were modified:

```text
app/detection/
app/tracking/
app/homography/
app/analytics/
app/pose/
run_pipeline.py
```

Focused verification:

```text
python -m unittest discover -s tests -p "test_ai*.py"
python -m py_compile app/ai/*.py app/api/main.py streamlit_app.py tests/test_ai*.py
```
