# Performance Report: eurobert_210m_finetune
**Report ID:** 200226_eurobert_210m_finetune_report_002
**Generated:** 2026-02-20 16:31:18

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
| Duration | 17m 31s |
| Articles/Second | 0.35 |
| GPU | NVIDIA L4 (23.7 GB) |
| CUDA | 12.8 |
| Est. Cost / 1000 Articles | $0.20 (L4, estimated) |

## Training Parameters
| Parameter | Value |
|---|---|
| num_epochs | 8 |
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
| best_checkpoint | /content/eurobert_finetune_output/checkpoint-425 |
| best_metric | 0.7795 |

## Aggregate Metrics
| Metric | Value |
|---|---|
| **F1 Macro** | **0.7590** |
| F1 Weighted | 0.7834 |
| Precision Macro | 0.7860 |
| Precision Weighted | 0.7924 |
| Recall Macro | 0.7590 |
| Recall Weighted | 0.7892 |
| Accuracy | 0.7892 |

## Per-Class Metrics
| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Klima / Energie | 0.8438 | 0.9000 | 0.8710 | 30 |
| Zuwanderung | 0.7931 | 0.7667 | 0.7797 | 30 |
| Renten | 0.8966 | 0.8667 | 0.8814 | 30 |
| Soziales Gefälle | 0.7143 | 0.5000 | 0.5882 | 30 |
| AfD/Rechte | 0.7297 | 0.9000 | 0.8060 | 30 |
| Arbeitslosigkeit | 0.9259 | 0.8333 | 0.8772 | 30 |
| Wirtschaftslage | 0.6667 | 0.7333 | 0.6984 | 30 |
| Politikverdruss | 0.6667 | 0.2000 | 0.3077 | 10 |
| Gesundheitswesen, Pflege | 0.8889 | 0.8000 | 0.8421 | 30 |
| Kosten/Löhne/Preise | 0.6286 | 0.7333 | 0.6769 | 30 |
| Ukraine/Krieg/Russland | 0.8485 | 0.9333 | 0.8889 | 30 |
| Bundeswehr/Verteidigung | 0.8929 | 0.8333 | 0.8621 | 30 |
| Andere | 0.7222 | 0.8667 | 0.7879 | 30 |

## Confusion Matrix (Counts)
| | Klima / Energie | Zuwanderung | Renten | Soziales Gefäll | AfD/Rechte | Arbeitslosigkei | Wirtschaftslage | Politikverdruss | Gesundheitswese | Kosten/Löhne/Pr | Ukraine/Krieg/R | Bundeswehr/Vert | Andere |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Klima / Energie** | 27 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Zuwanderung** | 0 | 23 | 0 | 0 | 5 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| **Renten** | 0 | 0 | 26 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 2 |
| **Soziales Gefäll** | 1 | 0 | 1 | 15 | 1 | 0 | 3 | 0 | 0 | 6 | 0 | 1 | 2 |
| **AfD/Rechte** | 1 | 1 | 0 | 0 | 27 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| **Arbeitslosigkei** | 0 | 0 | 1 | 0 | 0 | 25 | 3 | 0 | 1 | 0 | 0 | 0 | 0 |
| **Wirtschaftslage** | 2 | 0 | 1 | 1 | 0 | 1 | 22 | 0 | 0 | 3 | 0 | 0 | 0 |
| **Politikverdruss** | 0 | 2 | 0 | 1 | 2 | 0 | 0 | 2 | 0 | 0 | 1 | 1 | 1 |
| **Gesundheitswese** | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 24 | 3 | 0 | 0 | 2 |
| **Kosten/Löhne/Pr** | 1 | 0 | 0 | 3 | 1 | 0 | 1 | 0 | 0 | 22 | 0 | 0 | 2 |
| **Ukraine/Krieg/R** | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 28 | 1 | 0 |
| **Bundeswehr/Vert** | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 3 | 25 | 0 |
| **Andere** | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 26 |

## Notes
Fine-Tuned EuroBERT-210m auf 1348 Trainingsartikeln. Max Length 2048, FP16, EarlyStoppingCallback(patience=2). Custom Split: 30 Test/Klasse, Rest 80/20 Train/Val.

---
*Generated by pipeline_utils.generate_report()*