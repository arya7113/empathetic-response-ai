# Empathetic Response AI

An end-to-end pipeline that detects emotion in user text and generates an empathetic, context-aware response — combining a fine-tuned classifier with a fine-tuned generative model via LoRA.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arya7113/empathetic-response-ai/blob/main/notebooks/empatheticResponseAI.ipynb)

## Overview

This project implements a two-stage empathetic response system, similar in spirit to customer-support-style empathetic AI (e.g. Zendesk-style tooling):

**User input → Emotion Detection (DistilBERT) → Empathetic Response Generation (DialoGPT + LoRA)**

Built as a hands-on exploration of fine-tuning transformer models end-to-end: multi-label classification, parameter-efficient fine-tuning (PEFT/LoRA), synthetic dataset construction, and systematic debugging of training pipelines.

## Architecture

| Stage | Model | Task |
|---|---|---|
| Emotion Detection | DistilBERT (fine-tuned) | Multi-label classification across 6 emotions |
| Response Generation | DialoGPT-medium + LoRA | Emotion-conditioned response generation |

**Emotions:** sadness, joy, anger, fear, surprise, neutral

## Results

**Emotion Classifier (Phase 2)**
- Fine-tuned on GoEmotions, multi-label via `BCEWithLogitsLoss`
- Final F1 score: **0.713** (validation)
- Correctly handles inputs expressing multiple simultaneous emotions

**Response Generator (Phase 3)**
- Base model: `microsoft/DialoGPT-medium`, fine-tuned with LoRA (r=8, alpha=32, target modules: `c_attn`, `c_proj`)
- ~2.16M trainable parameters (0.6% of total)
- Trained on a synthetic dataset of ~4,550 examples, balanced across 6 emotions × 10 life domains
- Final validation loss: **2.88**
- Generation improved via best-of-5 sampling with a heuristic reranker (filters trailing questions, ungrounded pronouns, repetition, and generation artifacts)

## Key Engineering Decisions & Debugging

A few notable problems identified and resolved during development:

- **Label masking bug:** Initial training computed loss over the entire input sequence (prompt + response), diluting the learning signal. Fixed by masking prompt tokens with `-100` so loss is computed only on the response.
- **Dataset quality issues:** An early dataset (`facebook/empathetic_dialogues`) produced poor generations because responses were generic conversational turns, not true empathetic replies. Replaced with a purpose-built synthetic dataset.
- **Trailing-question artifact:** The synthetic dataset originally had *every* response ending in a question — appropriate for a multi-turn chat, but structurally wrong for this single-turn system. Cleaned via regex post-processing.
- **Systematic ablation:** Tested and ruled out several hypotheses for a training loss plateau (label masking, LoRA rank, base model choice) before identifying dataset scale as the dominant factor — validation loss improved from 2.95 → 2.88 primarily through dataset expansion (2,550 → 4,550 examples), not architecture changes.

## Known Limitations

Being upfront about where this system currently falls short:

- **Occasional hallucination:** The generator sometimes introduces details (people, relationships, specifics) not present in the input. Best-of-n reranking reduces but doesn't eliminate this.
- **Domain drift on some emotions:** Anger responses occasionally default to relationship-based framing even for non-relationship inputs (e.g., workplace conflict), likely reflecting a phrasing bias learned from training data distribution.
- **Single-turn only:** The system has no conversation memory — each response is generated independently. Multi-turn support would require both context-aware training data and prompt restructuring.

## Tech Stack

- `transformers`, `peft`, `accelerate`, `datasets`, `torch`
- Training: Google Colab, T4 GPU
- Models: `distilbert-base-uncased`, `microsoft/DialoGPT-medium`

## Repository Structure

\`\`\`
empathetic-response-ai/
├── notebooks/
│   └── empatheticResponseAI.ipynb
├── data/
│   ├── empathy_dataset_v2.csv
│   └── empathy_dataset_v3.csv
├── requirements.txt
└── README.md
\`\`\`

## Next Steps

- Expand dataset further to address remaining hallucination/domain-drift issues
- Add a lightweight fact-grounding or retrieval mechanism to reduce ungrounded generation
- Extend to multi-turn conversation support
- Deploy as a Gradio demo on Hugging Face Spaces
