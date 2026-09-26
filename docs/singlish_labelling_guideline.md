# Singlish Labelling Guideline

## Labels

### 0 = NON-TOXIC

Examples of meaning:

- Normal conversation
- Neutral statements
- Friendly comments
- Constructive criticism
- Disagreement without personal abuse

### 1 = TOXIC

Examples of meaning:

- Insults
- Abusive language
- Harassment
- Threats
- Hateful or degrading attacks
- Offensive personal attacks

## Annotation Rules

- Judge the meaning and context, not individual keywords alone.
- Profanity is not automatically toxic if the context is non-abusive.
- Criticism is not automatically toxic.
- Do not guess unclear comments.
- If context is too unclear, flag it for discussion rather than forcing a label.

Two human reviewers should label independently.

## Agreement

When `reviewer_1_label == reviewer_2_label`, `final_label` can later be accepted.

When `reviewer_1_label != reviewer_2_label`, the comment must be manually reviewed and resolved. Disagreements must not be resolved automatically.
