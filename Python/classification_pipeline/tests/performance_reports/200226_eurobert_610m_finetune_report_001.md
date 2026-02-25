# Performance Report: eurobert_610m_finetune
**Report ID:** 200226_eurobert_610m_finetune_report_001
**Generated:** 2026-02-20 17:25:11

---

## Model Information
| Property | Value |
|---|---|
| Model Name | eurobert_610m_finetune |
| Model Type | fine-tuned |
| HuggingFace ID | EuroBERT/EuroBERT-610m |
| Language | Multilingual (inkl. Deutsch) |
| Max Tokens | 2048 |
| Parameters | 610M |
| Notes | EuroBERT-610m, fine-tuned for single-label classification. Mixed Precision. |

## Model Architecture
| Property | Value |
|---|---|
| Architecture | EuroBertForSequenceClassification |
| Model Type | eurobert |
| Hidden Size | 1152 |
| Num Layers | 26 |
| Attention Heads | 18 |
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
| Duration | 61m 59s |
| Articles/Second | 0.10 |
| GPU | NVIDIA L4 (23.7 GB) |
| CUDA | 12.8 |
| Est. Cost / 1000 Articles | $0.70 (L4, estimated) |

## Training Parameters
| Parameter | Value |
|---|---|
| num_epochs | 8 |
| learning_rate | 2e-05 |
| batch_size_train | 2 |
| batch_size_eval | 8 |
| gradient_accumulation_steps | 8 |
| effective_batch_size | 16 |
| warmup_ratio | 0.06 |
| weight_decay | 0.01 |
| max_length | 2048 |
| fp16 | True |
| early_stopping_patience | 2 |
| best_checkpoint | /content/eurobert_610m_finetune_output/checkpoint-680 |
| best_metric | 0.7995 |

## Aggregate Metrics
| Metric | Value |
|---|---|
| **F1 Macro** | **0.7485** |
| F1 Weighted | 0.7889 |
| Precision Macro | 0.7452 |
| Precision Weighted | 0.7855 |
| Recall Macro | 0.7590 |
| Recall Weighted | 0.8000 |
| Accuracy | 0.8000 |

## Per-Class Metrics
| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Klima / Energie | 0.8182 | 0.9000 | 0.8571 | 30 |
| Zuwanderung | 0.8182 | 0.9000 | 0.8571 | 30 |
| Renten | 0.9286 | 0.8667 | 0.8966 | 30 |
| Soziales Gefälle | 0.7727 | 0.5667 | 0.6538 | 30 |
| AfD/Rechte | 0.7179 | 0.9333 | 0.8116 | 30 |
| Arbeitslosigkeit | 0.9286 | 0.8667 | 0.8966 | 30 |
| Wirtschaftslage | 0.7188 | 0.7667 | 0.7419 | 30 |
| Politikverdruss | 0.0000 | 0.0000 | 0.0000 | 10 |
| Gesundheitswesen, Pflege | 0.8571 | 0.8000 | 0.8276 | 30 |
| Kosten/Löhne/Preise | 0.7143 | 0.6667 | 0.6897 | 30 |
| Ukraine/Krieg/Russland | 0.8710 | 0.9000 | 0.8852 | 30 |
| Bundeswehr/Verteidigung | 0.8667 | 0.8667 | 0.8667 | 30 |
| Andere | 0.6757 | 0.8333 | 0.7463 | 30 |

## Confusion Matrix (Counts)
| | Klima / Energie | Zuwanderung | Renten | Soziales Gefäll | AfD/Rechte | Arbeitslosigkei | Wirtschaftslage | Politikverdruss | Gesundheitswese | Kosten/Löhne/Pr | Ukraine/Krieg/R | Bundeswehr/Vert | Andere |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Klima / Energie** | 27 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| **Zuwanderung** | 0 | 27 | 0 | 0 | 2 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Renten** | 1 | 0 | 26 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| **Soziales Gefäll** | 1 | 0 | 1 | 17 | 2 | 0 | 1 | 0 | 2 | 1 | 0 | 2 | 3 |
| **AfD/Rechte** | 0 | 1 | 0 | 0 | 28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| **Arbeitslosigkei** | 0 | 1 | 0 | 0 | 0 | 26 | 2 | 0 | 0 | 0 | 0 | 0 | 1 |
| **Wirtschaftslage** | 1 | 0 | 0 | 0 | 0 | 2 | 23 | 0 | 0 | 4 | 0 | 0 | 0 |
| **Politikverdruss** | 1 | 2 | 0 | 1 | 4 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| **Gesundheitswese** | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 24 | 2 | 1 | 0 | 2 |
| **Kosten/Löhne/Pr** | 1 | 1 | 0 | 3 | 1 | 0 | 1 | 0 | 0 | 20 | 0 | 0 | 3 |
| **Ukraine/Krieg/R** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 27 | 2 | 1 |
| **Bundeswehr/Vert** | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 3 | 26 | 0 |
| **Andere** | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 25 |

## Notes
Fine-Tuned EuroBERT-610m auf 1348 Trainingsartikeln. Max Length 2048, FP16, EarlyStoppingCallback(patience=2). Custom Split: 30 Test/Klasse, Rest 80/20 Train/Val.

---
*Generated by pipeline_utils.generate_report()*