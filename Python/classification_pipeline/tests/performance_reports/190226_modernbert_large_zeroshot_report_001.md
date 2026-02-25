# Performance Report: modernbert_large_zeroshot
**Report ID:** 190226_modernbert_large_zeroshot_report_001
**Generated:** 2026-02-19 16:40:46

---

## Model Information
| Property | Value |
|---|---|
| Model Name | modernbert_large_zeroshot |
| Model Type | zero-shot |
| HuggingFace ID | MoritzLaurer/ModernBERT-large-zeroshot-v2.0 |
| Language | English |
| Max Tokens | 8192 |
| Parameters | 0.4B |
| Notes | ModernBERT-large, sehr schnell und speichereffizient. Unterstützt BF16/FP16. 8K Context Window. English-only. |

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
| Duration | 5m 26s |
| Articles/Second | 1.89 |
| GPU | Tesla T4 (15.6 GB) |
| CUDA | 12.8 |
| Est. Cost / 1000 Articles | $0.02 (T4, estimated) |

## Aggregate Metrics
| Metric | Value |
|---|---|
| **F1 Macro** | **0.3244** |
| F1 Weighted | 0.3369 |
| Precision Macro | 0.5911 |
| Precision Weighted | 0.6120 |
| Recall Macro | 0.3368 |
| Recall Weighted | 0.3517 |
| Accuracy | 0.3517 |

## Per-Class Metrics
| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| vom Klima, Klimawandel oder der Energieversorgung | 0.8000 | 0.0800 | 0.1455 | 50 |
| von Zuwanderung oder Migration | 0.7273 | 0.4800 | 0.5783 | 50 |
| von der Rente oder dem Rentensystem | 0.8936 | 0.8400 | 0.8660 | 50 |
| vom Sozialen Gefälle oder von sozialer Ungleicheit | 0.0000 | 0.0000 | 0.0000 | 50 |
| von der Brandmauer, der AfD oder dem Rechtsextremismus | 1.0000 | 0.0200 | 0.0392 | 50 |
| von Arbeitslosigkeit | 0.9556 | 0.8600 | 0.9053 | 50 |
| von der Wirtschaftslage oder der Zukunft der Deutschen Wirtschaft | 0.4706 | 0.1600 | 0.2388 | 50 |
| von Politikverdruss, dem Vertrauen in die Demokraite oder der Interesse für Politik bei den Bürgern | 0.2000 | 0.0588 | 0.0909 | 17 |
| vom Gesundheitswesen, der Pflege oder Krankenversicherungen | 1.0000 | 0.0800 | 0.1481 | 50 |
| von steigenden Preisen und Lebenshaltungskosten oder von Löhnen | 0.0000 | 0.0000 | 0.0000 | 50 |
| vom Ukraine Krieg oder von Russland | 0.7907 | 0.6800 | 0.7312 | 50 |
| von der Bundeswehr, der Verteidigung Deutschlands oder Investitionen in die Rüstung | 0.7273 | 0.1600 | 0.2623 | 50 |
| Andere | 0.1191 | 0.9600 | 0.2119 | 50 |

## Confusion Matrix (Counts)
| | vom Klima, Klim | von Zuwanderung | von der Rente o | vom Sozialen Ge | von der Brandma | von Arbeitslosi | von der Wirtsch | von Politikverd | vom Gesundheits | von steigenden  | vom Ukraine Kri | von der Bundesw | Andere |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **vom Klima, Klim** | 4 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 45 |
| **von Zuwanderung** | 0 | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 26 |
| **von der Rente o** | 0 | 0 | 42 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 |
| **vom Sozialen Ge** | 0 | 2 | 1 | 0 | 0 | 2 | 2 | 0 | 0 | 1 | 0 | 1 | 41 |
| **von der Brandma** | 0 | 6 | 0 | 0 | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 39 |
| **von Arbeitslosi** | 0 | 0 | 0 | 0 | 0 | 43 | 1 | 0 | 0 | 0 | 0 | 0 | 6 |
| **von der Wirtsch** | 0 | 1 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 1 | 0 | 40 |
| **von Politikverd** | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 14 |
| **vom Gesundheits** | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 45 |
| **von steigenden ** | 0 | 0 | 1 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 2 | 0 | 42 |
| **vom Ukraine Kri** | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 34 | 0 | 15 |
| **von der Bundesw** | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 6 | 8 | 34 |
| **Andere** | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 48 |

## Notes
Zero-Shot NLI-Klassifikation auf Volltext mit ModernBERT-large. 8K Context Window — Texte werden kaum gekürzt. FP16 aktiviert. 'Andere' per Confidence-Threshold (0.4) zugewiesen.

---
*Generated by pipeline_utils.generate_report()*