# Performance Report: deberta_v3_large_zeroshot
**Report ID:** 190226_deberta_v3_large_zeroshot_report_002
**Generated:** 2026-02-19 16:41:12

---

## Model Information
| Property | Value |
|---|---|
| Model Name | deberta_v3_large_zeroshot |
| Model Type | zero-shot |
| HuggingFace ID | MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33 |
| Language | English |
| Max Tokens | 512 |
| Parameters | 304M |
| Notes | DeBERTa-v3-large, trained on 33 NLI datasets. Supports FP16. English-only. |

## Classification Config
| Property | Value |
|---|---|
| Hypothesis Template | `Dieser Text handelt {}.` |
| Num Labels | 13 |

## Candidate Labels & NLI Phrases
| # | Original | Candidate Label | NLI Phrase |
|---|---|---|---|
| 1 | Klima / Energie | vom Klima, Klimawandel oder der Energieversorgung | Dieser Text handelt vom Klima, Klimawandel oder der Energieversorgung. |
| 2 | Zuwanderung | von Zuwanderung oder Migration | Dieser Text handelt von Zuwanderung oder Migration. |
| 3 | Renten | von der Rente oder dem Rentensystem | Dieser Text handelt von der Rente oder dem Rentensystem. |
| 4 | Soziales Gefälle | vom Sozialen Gefälle oder von sozialer Ungleicheit | Dieser Text handelt vom Sozialen Gefälle oder von sozialer Ungleicheit. |
| 5 | AfD/Rechte | von der Brandmauer, der AfD oder dem Rechtsextremismus | Dieser Text handelt von der Brandmauer, der AfD oder dem Rechtsextremismus. |
| 6 | Arbeitslosigkeit | von Arbeitslosigkeit | Dieser Text handelt von Arbeitslosigkeit. |
| 7 | Wirtschaftslage | von der Wirtschaftslage oder der Zukunft der Deutschen Wirtschaft | Dieser Text handelt von der Wirtschaftslage oder der Zukunft der Deutschen Wirtschaft. |
| 8 | Politikverdruss | von Politikverdruss, dem Vertrauen in die Demokraite oder der Interesse für Politik bei den Bürgern | Dieser Text handelt von Politikverdruss, dem Vertrauen in die Demokraite oder der Interesse für Politik bei den Bürgern. |
| 9 | Gesundheitswesen, Pflege | vom Gesundheitswesen, der Pflege oder Krankenversicherungen | Dieser Text handelt vom Gesundheitswesen, der Pflege oder Krankenversicherungen. |
| 10 | Kosten/Löhne/Preise | von steigenden Preisen und Lebenshaltungskosten oder von Löhnen | Dieser Text handelt von steigenden Preisen und Lebenshaltungskosten oder von Löhnen. |
| 11 | Ukraine/Krieg/Russland | vom Ukraine Krieg oder von Russland | Dieser Text handelt vom Ukraine Krieg oder von Russland. |
| 12 | Bundeswehr/Verteidigung | von der Bundeswehr, der Verteidigung Deutschlands oder Investitionen in die Rüstung | Dieser Text handelt von der Bundeswehr, der Verteidigung Deutschlands oder Investitionen in die Rüstung. |
| 13 | — | Andere | Dieser Text handelt Andere. |

## Dataset Information
| Property | Value |
|---|---|
| Dataset | Zorryy/news_articles_2025_elections_germany |
| Evaluated On | test (617 articles) |
| N Train | 1043 |
| N Eval | 261 |
| N Test | 617 |
| N Raw | 0 |
| N Total | 1921 |
| Split Mode | percentage |
| Random Seed | 42 |

## Runtime
| Property | Value |
|---|---|
| Duration | 7m 51s |
| Articles/Second | 1.31 |
| GPU | Tesla T4 (15.6 GB) |
| CUDA | 12.8 |
| Est. Cost / 1000 Articles | $0.03 (T4, estimated) |

## Aggregate Metrics
| Metric | Value |
|---|---|
| **F1 Macro** | **0.7027** |
| F1 Weighted | 0.7147 |
| Precision Macro | 0.7247 |
| Precision Weighted | 0.7431 |
| Recall Macro | 0.7175 |
| Recall Weighted | 0.7212 |
| Accuracy | 0.7212 |

## Per-Class Metrics
| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| vom Klima, Klimawandel oder der Energieversorgung | 0.7547 | 0.8000 | 0.7767 | 50 |
| von Zuwanderung oder Migration | 0.7407 | 0.8000 | 0.7692 | 50 |
| von der Rente oder dem Rentensystem | 0.9783 | 0.9000 | 0.9375 | 50 |
| vom Sozialen Gefälle oder von sozialer Ungleicheit | 0.7059 | 0.4800 | 0.5714 | 50 |
| von der Brandmauer, der AfD oder dem Rechtsextremismus | 0.8718 | 0.6800 | 0.7640 | 50 |
| von Arbeitslosigkeit | 0.8302 | 0.8800 | 0.8544 | 50 |
| von der Wirtschaftslage oder der Zukunft der Deutschen Wirtschaft | 0.5085 | 0.6000 | 0.5505 | 50 |
| von Politikverdruss, dem Vertrauen in die Demokraite oder der Interesse für Politik bei den Bürgern | 0.3793 | 0.6471 | 0.4783 | 17 |
| vom Gesundheitswesen, der Pflege oder Krankenversicherungen | 0.7903 | 0.9800 | 0.8750 | 50 |
| von steigenden Preisen und Lebenshaltungskosten oder von Löhnen | 0.7333 | 0.2200 | 0.3385 | 50 |
| vom Ukraine Krieg oder von Russland | 0.8846 | 0.9200 | 0.9020 | 50 |
| von der Bundeswehr, der Verteidigung Deutschlands oder Investitionen in die Rüstung | 0.8431 | 0.8600 | 0.8515 | 50 |
| Andere | 0.4000 | 0.5600 | 0.4667 | 50 |

## Confusion Matrix (Counts)
| | vom Klima, Klim | von Zuwanderung | von der Rente o | vom Sozialen Ge | von der Brandma | von Arbeitslosi | von der Wirtsch | von Politikverd | vom Gesundheits | von steigenden  | vom Ukraine Kri | von der Bundesw | Andere |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **vom Klima, Klim** | 40 | 1 | 0 | 0 | 0 | 0 | 1 | 2 | 0 | 0 | 0 | 0 | 6 |
| **von Zuwanderung** | 0 | 40 | 0 | 0 | 2 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 5 |
| **von der Rente o** | 0 | 0 | 45 | 2 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 1 |
| **vom Sozialen Ge** | 1 | 1 | 1 | 24 | 0 | 8 | 1 | 1 | 1 | 4 | 0 | 1 | 7 |
| **von der Brandma** | 0 | 9 | 0 | 1 | 34 | 0 | 0 | 3 | 1 | 0 | 0 | 1 | 1 |
| **von Arbeitslosi** | 0 | 0 | 0 | 1 | 0 | 44 | 3 | 1 | 0 | 0 | 0 | 1 | 0 |
| **von der Wirtsch** | 2 | 0 | 0 | 0 | 0 | 0 | 30 | 0 | 2 | 0 | 0 | 1 | 15 |
| **von Politikverd** | 1 | 3 | 0 | 1 | 1 | 0 | 0 | 11 | 0 | 0 | 0 | 0 | 0 |
| **vom Gesundheits** | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 49 | 0 | 0 | 0 | 0 |
| **von steigenden ** | 5 | 0 | 0 | 4 | 0 | 0 | 23 | 1 | 0 | 11 | 0 | 0 | 6 |
| **vom Ukraine Kri** | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 46 | 1 | 0 |
| **von der Bundesw** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 43 | 1 |
| **Andere** | 3 | 0 | 0 | 0 | 1 | 0 | 0 | 7 | 8 | 0 | 0 | 3 | 28 |

## Notes
Zero-Shot NLI-Klassifikation mit DeBERTa-v3-large auf Volltext. Texte werden automatisch auf 512 Tokens gekürzt (inverted pyramid). 'Andere' per Confidence-Threshold (0.4) zugewiesen.

---
*Generated by pipeline_utils.generate_report()*