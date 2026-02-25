# Performance Report: eurobert_210m_finetune
**Report ID:** 200226_eurobert_210m_finetune_report_001
**Generated:** 2026-02-20 16:09:17

---

## Model Information
| Property | Value |
|---|---|
| Model Name | eurobert_210m_finetune |
| Model Type | fine-tuned |
| HuggingFace ID | EuroBERT/EuroBERT-210m |
| Language | Multilingual (inkl. Deutsch) |
| Max Tokens | 2048 |
| Parameters | 210M |
| Notes | EuroBERT-210m, fine-tuned for single-label classification. Mixed Precision. |

## Model Architecture
| Property | Value |
|---|---|
| Architecture | EuroBertForSequenceClassification |
| Model Type | eurobert |
| Hidden Size | 768 |
| Num Layers | 12 |
| Attention Heads | 12 |
| Vocab Size | 128256 |
| Max Position Embeddings | 8192 |

## Classification Config
| Property | Value |
|---|---|
| Hypothesis Template | `Dieser Text handelt von {}.` |
| Num Labels | 13 |

## Candidate Labels & NLI Phrases
| # | Candidate Label | NLI Phrase |
|---|---|---|
| 1 | Klima / Energie | Dieser Text handelt von Klima / Energie. |
| 2 | Zuwanderung | Dieser Text handelt von Zuwanderung. |
| 3 | Renten | Dieser Text handelt von Renten. |
| 4 | Soziales Gefälle | Dieser Text handelt von Soziales Gefälle. |
| 5 | AfD/Rechte | Dieser Text handelt von AfD/Rechte. |
| 6 | Arbeitslosigkeit | Dieser Text handelt von Arbeitslosigkeit. |
| 7 | Wirtschaftslage | Dieser Text handelt von Wirtschaftslage. |
| 8 | Politikverdruss | Dieser Text handelt von Politikverdruss. |
| 9 | Gesundheitswesen, Pflege | Dieser Text handelt von Gesundheitswesen, Pflege. |
| 10 | Kosten/Löhne/Preise | Dieser Text handelt von Kosten/Löhne/Preise. |
| 11 | Ukraine/Krieg/Russland | Dieser Text handelt von Ukraine/Krieg/Russland. |
| 12 | Bundeswehr/Verteidigung | Dieser Text handelt von Bundeswehr/Verteidigung. |
| 13 | Andere | Dieser Text handelt von Andere. |

## Dataset Information
| Property | Value |
|---|---|
| Dataset | Zorryy/news_articles_2025_elections_germany |
| Evaluated On | test (370 articles) |
| N Train | 1348 |
| N Eval | 337 |
| N Test | 370 |
| N Raw | 0 |
| N Total | 2055 |
| Split Mode | custom_finetune |
| Random Seed | 42 |

## Runtime
| Property | Value |
|---|---|
| Duration | 7m 29s |
| Articles/Second | 0.82 |
| GPU | NVIDIA L4 (23.7 GB) |
| CUDA | 12.8 |
| Est. Cost / 1000 Articles | $0.08 (L4, estimated) |

## Training Parameters
| Parameter | Value |
|---|---|
| num_epochs | 3 |
| learning_rate | 2e-05 |
| batch_size_train | 4 |
| batch_size_eval | 16 |
| gradient_accumulation_steps | 4 |
| effective_batch_size | 16 |
| warmup_ratio | 0.06 |
| weight_decay | 0.01 |
| max_length | 2048 |
| fp16 | True |
| early_stopping_patience | 2 |
| best_checkpoint | /content/eurobert_finetune_output/checkpoint-170 |
| best_metric | 0.7684 |

## Aggregate Metrics
| Metric | Value |
|---|---|
| **F1 Macro** | **0.7647** |
| F1 Weighted | 0.8060 |
| Precision Macro | 0.7629 |
| Precision Weighted | 0.8041 |
| Recall Macro | 0.7744 |
| Recall Weighted | 0.8162 |
| Accuracy | 0.8162 |

## Per-Class Metrics
| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Klima / Energie | 0.8400 | 0.7000 | 0.7636 | 30 |
| Zuwanderung | 0.7073 | 0.9667 | 0.8169 | 30 |
| Renten | 0.9630 | 0.8667 | 0.9123 | 30 |
| Soziales Gefälle | 0.6970 | 0.7667 | 0.7302 | 30 |
| AfD/Rechte | 0.8889 | 0.8000 | 0.8421 | 30 |
| Arbeitslosigkeit | 0.8788 | 0.9667 | 0.9206 | 30 |
| Wirtschaftslage | 0.8519 | 0.7667 | 0.8070 | 30 |
| Politikverdruss | 0.0000 | 0.0000 | 0.0000 | 10 |
| Gesundheitswesen, Pflege | 0.9231 | 0.8000 | 0.8571 | 30 |
| Kosten/Löhne/Preise | 0.6471 | 0.7333 | 0.6875 | 30 |
| Ukraine/Krieg/Russland | 0.9310 | 0.9000 | 0.9153 | 30 |
| Bundeswehr/Verteidigung | 0.8182 | 0.9000 | 0.8571 | 30 |
| Andere | 0.7714 | 0.9000 | 0.8308 | 30 |

## Confusion Matrix (Counts)
| | Klima / Energie | Zuwanderung | Renten | Soziales Gefäll | AfD/Rechte | Arbeitslosigkei | Wirtschaftslage | Politikverdruss | Gesundheitswese | Kosten/Löhne/Pr | Ukraine/Krieg/R | Bundeswehr/Vert | Andere |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Klima / Energie** | 21 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 4 | 0 | 0 | 2 |
| **Zuwanderung** | 0 | 29 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Renten** | 0 | 0 | 26 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 |
| **Soziales Gefäll** | 0 | 2 | 1 | 23 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 2 |
| **AfD/Rechte** | 0 | 5 | 0 | 0 | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| **Arbeitslosigkei** | 0 | 1 | 0 | 0 | 0 | 29 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Wirtschaftslage** | 1 | 0 | 0 | 0 | 0 | 2 | 23 | 0 | 0 | 2 | 0 | 2 | 0 |
| **Politikverdruss** | 0 | 3 | 0 | 3 | 2 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| **Gesundheitswese** | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 24 | 2 | 0 | 0 | 1 |
| **Kosten/Löhne/Pr** | 1 | 0 | 0 | 3 | 1 | 0 | 0 | 0 | 1 | 22 | 0 | 1 | 1 |
| **Ukraine/Krieg/R** | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 27 | 2 | 0 |
| **Bundeswehr/Vert** | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 27 | 0 |
| **Andere** | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 27 |

## Notes
Fine-Tuned EuroBERT-210m auf 1348 Trainingsartikeln. Max Length 2048, FP16, EarlyStoppingCallback(patience=2). Custom Split: 30 Test/Klasse, Rest 80/20 Train/Val.

---
*Generated by pipeline_utils.generate_report()*