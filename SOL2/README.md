# Pravaah (CAIR-CD)

Causal Analysis and Interactive Reasoning over Conversational Data.

Pravaah is a conversational intelligence system that combines:

- deterministic feature engineering,
- interpretable causal signal mining,
- evidence-grounded reasoning,
- outcome prediction with a trained neural model,
- and counterfactual intervention simulation.

It is packaged as a Streamlit app with modular Python pipelines.

## What This Project Does

Given customer-agent transcripts, Pravaah can:

1. Explain why an outcome event occurred (causal factors + evidence).
2. Predict likely intent/outcome label from a conversation (ML model).
3. Handle follow-up analytical questions with context.
4. Propose counterfactual actions that could have prevented the outcome.

## Key Features

- End-to-end processing from raw transcripts to interactive analysis.
- Traceable evidence index tied to concrete conversation turn spans.
- Lightweight, interpretable causal signals (question ratio, repeat ratio, etc.).
- Multi-turn query handling with intent routing.
- Counterfactual rule engine for preventive interventions.

## Repository Structure

```text
.
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- schema.md
|-- preprocessing/
|-- features/
|-- casual_analysis/
|-- evidence_index/
|-- ml_pipeline/
|-- query_engine/
|-- context_memory/
|-- counterfactual/
|-- streamlit_app/
|   |-- Home.py
|   `-- pages/
|-- artifacts/
|-- EDA/
`-- notebooks/
```

## End-to-End Data Flow

1. Validate raw transcripts  
   `preprocessing/ingest_validate.py`
2. Normalize to canonical schema  
   `preprocessing/normalize.py`
3. Extract turn-level features  
   `features/turn_features.py`
4. Extract conversation-level features  
   `features/conversation_features.py`
5. Mine candidate causal signals  
   `casual_analysis/signal_miner.py`
6. Score signals (lift-based)  
   `casual_analysis/casual_scoring.py`
7. Build evidence index  
   `evidence_index/build_index.py`
8. Serve query + prediction + follow-up + counterfactual  
   `query_engine/task1_explainer.py`

## Core Components

### 1. Preprocessing

- `preprocessing/ingest_validate.py`
  - loads `data/raw/Conversational_Transcript_Dataset.json`
  - validates required fields and speaker/text constraints
- `preprocessing/normalize.py`
  - maps raw records to canonical conversation object
  - derives `outcome_event` from intent heuristics
  - writes `data/processed/conversations.json`

### 2. Feature Engineering

- `features/turn_features.py`
  - `token_count`
  - `is_question`
  - `is_repeat`
- `features/conversation_features.py`
  - `total_turns`
  - `customer_turns` and `agent_turns`
  - `question_ratio`
  - `repeat_ratio`
  - `customer_last_turn`

### 3. Causal Analysis

- `casual_analysis/signal_miner.py`
  - mines candidate signals conditioned on outcome events
- `casual_analysis/casual_scoring.py`
  - scores signals using lift:
    - `P(signal | outcome) / P(signal overall)`

### 4. Evidence Index

- `evidence_index/build_index.py`
  - selects strong signals (`lift >= 1.5`)
  - stores traceable evidence records:
    - conversation id
    - domain
    - outcome event
    - signal
    - turn span

### 5. ML Outcome Prediction

- `ml_pipeline/build_single_feature.py`
  - deterministic features + MPNet embedding
- `ml_pipeline/predict_outcome.py`
  - loads `artifacts/outcome_predictor.keras`
  - loads `artifacts/label_mapping.json`
  - returns predicted label + confidence

### 6. Query and Reasoning

- `query_engine/task1_explainer.py`
  - routes user query to:
    - fresh causal explanation,
    - follow-up handler,
    - or counterfactual engine
- `query_engine/intent_classifier.py`
  - semantic intent detection for follow-up questions
- `query_engine/followup_handler.py`
  - context-aware responses for supported analytical intents
- `context_memory/analysis_context.py`
  - stores last query/result for multi-turn reasoning

### 7. Counterfactual Simulation

- `counterfactual/counterfactual_rules.py`
  - rule base of interventions
- `counterfactual/counterfactual_engine.py`
  - maps top causal factors to preventive actions

## Streamlit App Pages

- `streamlit_app/Home.py`  
  Unified dashboard for analysis, evidence, follow-up, and counterfactuals.
- `streamlit_app/pages/1_Overview.py`  
  High-level system summary.
- `streamlit_app/pages/2_Conversation_Explorer.py`  
  Browse transcripts with highlighted evidence spans.
- `streamlit_app/pages/3_Causal_Analysis.py`  
  Standalone analysis runner.
- `streamlit_app/pages/4_Multi_Turn_Query.py`  
  Chat-style context-aware querying.
- `streamlit_app/pages/5_Counterfactual_Simulator.py`  
  Standalone counterfactual interface.

## Installation

### Prerequisites

- Python 3.10 or 3.11 recommended
- pip

### Setup

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the App

From repository root:

```bash
streamlit run streamlit_app/Home.py
```

Then open the URL shown in terminal (usually `http://localhost:8501`).

## Rebuilding Pipeline Outputs

Run these scripts in order from repository root:

```bash
python preprocessing/ingest_validate.py
python preprocessing/normalize.py
python features/turn_features.py
python features/conversation_features.py
python casual_analysis/signal_miner.py
python casual_analysis/casual_scoring.py
python evidence_index/build_index.py
```

This regenerates processed data and causal artifacts in:

- `data/processed/`
- `artifacts/`

## ML Utility Scripts

- `python ml_pipeline/test_load_model.py`  
  Verifies model can be loaded.
- `python ml_pipeline/test_predict.py`  
  Runs dummy-shape prediction smoke test.
- `python ml_pipeline/test_real_prediction.py`  
  Runs prediction on a real sample conversation.
- `python ml_pipeline/rebuild_label_mapping.py`  
  Recreates `artifacts/label_mapping.json` from train split labels.

## Important Data and Artifacts

- Raw dataset: `data/raw/Conversational_Transcript_Dataset.json`
- Normalized dataset: `data/processed/conversations.json`
- Feature-enriched dataset: `data/processed/conversations_with_conv_features.json`
- Candidate signals: `artifacts/candidate_signals.json`
- Scored signals: `artifacts/scored_causal_signals.json`
- Evidence index: `artifacts/evidence_index.json`
- Trained model: `artifacts/outcome_predictor.keras`
- Label map: `artifacts/label_mapping.json`

## Current Constraints and Notes

- Outcome event mapping in preprocessing is heuristic and intent-string based.
- Causal signals are intentionally simple and interpretable, not full causal graphs.
- Counterfactual outputs are rule-driven templates, not policy-optimized actions.
- Repository currently tracks some generated files (`__pycache__`, `.pytouch`, temp script).

## Troubleshooting

- `ModuleNotFoundError` from script execution:
  - run scripts from repository root,
  - ensure virtual environment is activated.
- Slow first prediction:
  - sentence-transformer model download/initialization can take time initially.
- Streamlit page errors on missing artifacts:
  - regenerate pipeline outputs using the scripts listed above.

## License

This project is licensed under GNU GPL v3. See `LICENSE`.
