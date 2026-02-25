# Performance Report: bge_m3_v2_zeroshot
**Report ID:** 190226_bge_m3_v2_zeroshot_report_002
**Generated:** 2026-02-19 16:14:45

---

## Model Information
| Property | Value |
|---|---|
| Model Name | bge_m3_v2_zeroshot |
| Model Type | zero-shot |
| HuggingFace ID | MoritzLaurer/bge-m3-zeroshot-v2.0 |
| Language | Multilingual (100+ Sprachen, inkl. Deutsch) |
| Max Tokens | 8192 |
| Parameters | 0.6B |
| Notes | BGE-M3 basiert auf XLM-RoBERTa. Unterstützt FP16. 8K Context Window ideal für lange Artikel. |

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
| 7 | Wirtschaftslage | der Wirtschaftslage oder der Zukunft der Deutschen Wirtschaft | Dieser Text handelt der Wirtschaftslage oder der Zukunft der Deutschen Wirtschaft. |
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
| Duration | 3m 12s |
| Articles/Second | 3.21 |
| GPU | Tesla T4 (15.6 GB) |
| CUDA | 12.8 |
| Est. Cost / 1000 Articles | $0.01 (T4, estimated) |

## Aggregate Metrics
| Metric | Value |
|---|---|
| **F1 Macro** | **0.6468** |
| F1 Weighted | 0.6672 |
| Precision Macro | 0.7515 |
| Precision Weighted | 0.7752 |
| Recall Macro | 0.6443 |
| Recall Weighted | 0.6661 |
| Accuracy | 0.6661 |

## Per-Class Metrics
| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| vom Klima, Klimawandel oder der Energieversorgung | 0.8788 | 0.5800 | 0.6988 | 50 |
| von Zuwanderung oder Migration | 0.6716 | 0.9000 | 0.7692 | 50 |
| von der Rente oder dem Rentensystem | 0.9756 | 0.8000 | 0.8791 | 50 |
| vom Sozialen Gefälle oder von sozialer Ungleicheit | 0.7619 | 0.6400 | 0.6957 | 50 |
| von der Brandmauer, der AfD oder dem Rechtsextremismus | 0.8889 | 0.1600 | 0.2712 | 50 |
| von Arbeitslosigkeit | 0.9773 | 0.8600 | 0.9149 | 50 |
| der Wirtschaftslage oder der Zukunft der Deutschen Wirtschaft | 0.5122 | 0.4200 | 0.4615 | 50 |
| von Politikverdruss, dem Vertrauen in die Demokraite oder der Interesse für Politik bei den Bürgern | 0.3077 | 0.2353 | 0.2667 | 17 |
| vom Gesundheitswesen, der Pflege oder Krankenversicherungen | 0.8571 | 0.9600 | 0.9057 | 50 |
| von steigenden Preisen und Lebenshaltungskosten oder von Löhnen | 0.8571 | 0.2400 | 0.3750 | 50 |
| vom Ukraine Krieg oder von Russland | 0.8600 | 0.8600 | 0.8600 | 50 |
| von der Bundeswehr, der Verteidigung Deutschlands oder Investitionen in die Rüstung | 0.9556 | 0.8600 | 0.9053 | 50 |
| Andere | 0.2654 | 0.8600 | 0.4057 | 50 |

## Confusion Matrix (Counts)
| | vom Klima, Klim | von Zuwanderung | von der Rente o | vom Sozialen Ge | von der Brandma | von Arbeitslosi | der Wirtschafts | von Politikverd | vom Gesundheits | von steigenden  | vom Ukraine Kri | von der Bundesw | Andere |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **vom Klima, Klim** | 29 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 19 |
| **von Zuwanderung** | 0 | 45 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 4 |
| **von der Rente o** | 0 | 1 | 40 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 6 |
| **vom Sozialen Ge** | 1 | 2 | 1 | 32 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 12 |
| **von der Brandma** | 0 | 13 | 0 | 0 | 8 | 0 | 0 | 5 | 1 | 0 | 0 | 0 | 23 |
| **von Arbeitslosi** | 0 | 0 | 0 | 0 | 0 | 43 | 2 | 1 | 0 | 0 | 0 | 0 | 4 |
| **der Wirtschafts** | 0 | 1 | 0 | 1 | 0 | 0 | 21 | 1 | 3 | 1 | 1 | 0 | 21 |
| **von Politikverd** | 0 | 2 | 0 | 0 | 1 | 0 | 0 | 4 | 1 | 0 | 0 | 0 | 9 |
| **vom Gesundheits** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 48 | 0 | 0 | 0 | 2 |
| **von steigenden ** | 2 | 0 | 0 | 6 | 0 | 0 | 16 | 0 | 1 | 12 | 0 | 0 | 13 |
| **vom Ukraine Kri** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 43 | 2 | 4 |
| **von der Bundesw** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 43 | 2 |
| **Andere** | 1 | 2 | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 43 |

## Notes
Zero-Shot NLI-Klassifikation auf Volltext mit BGE-M3 v2. 8K Context Window — Texte werden kaum gekürzt. FP16 aktiviert. 'Andere' per Confidence-Threshold (0.4) zugewiesen.

---
*Generated by pipeline_utils.generate_report()*