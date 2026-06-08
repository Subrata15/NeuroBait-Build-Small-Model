# Evaluation Plan

NeuroBait needs both project-specific evaluation and a more common chat-model
evaluation pattern. The goal is to avoid relying on loss alone, because prior
runs showed eval loss is a weak proxy for this style task.

## Eval 1: NeuroBait-Specific Eval

Purpose: test whether the fine-tune learned the intended persona and multi-turn
momentum behavior.

Core checks:

- no label leakage such as `Micro-action`, `Hook`, `Stakes`
- short, warm, natural prose
- no guilt framing
- preserves user agency
- asks one useful question when context is insufficient
- gives one tiny concrete action when context is sufficient
- response length remains compact
- conversation arc builds momentum over turns
- English inputs do not degrade or trigger unwanted code-switching

Inputs:

- held-out `data/eval.jsonl`
- novel ID prompts
- novel EN prompts
- app-initiated opener continuation cases

Outputs:

- JSON generations and scores
- Markdown report
- qualitative examples for reviewer inspection

## Eval 2: General Chat-Model Pairwise Judge Eval

Purpose: add a standardized second signal commonly used for instruction/chat
models.

Approach:

- Generate responses from base and fine-tuned model for the same prompt set.
- Hide model identity from the judge.
- Ask a judge model to choose winner/tie under a fixed rubric.
- Report win/tie/loss rate and category-level notes.

This is similar in spirit to MT-Bench / AlpacaEval-style pairwise judging, but
with a NeuroBait-specific rubric added.

Suggested categories:

- Helpfulness: does it move the user toward starting?
- Persona fit: warm, agency-preserving, non-judgmental.
- Instruction following: follows the system prompt without exposing structure.
- Specificity: avoids generic motivation and uses context.
- Concision: compact enough for ADHD-friendly interaction.
- Safety: avoids medical overclaiming, shame, coercion, or crisis mishandling.
- Language consistency: responds naturally in the user's language.

Prompt set:

- ID single-turn prompts
- EN single-turn prompts
- ID multi-turn continuations
- EN multi-turn continuations
- sparse-context prompts requiring one question
- sufficient-context prompts requiring one micro-action
- edge cases: shame, avoidance spiral, deadline panic, boredom, overwhelm

Reporting:

- overall fine-tuned win rate vs base
- ID win/tie/loss
- EN win/tie/loss
- single-turn vs multi-turn split
- safety failures
- representative wins and losses

Guardrails:

- The judge prompt must forbid rewarding visible labels or long coaching plans.
- Pairwise judge results are advisory, not final truth.
- Human review remains required before upload/deploy.
