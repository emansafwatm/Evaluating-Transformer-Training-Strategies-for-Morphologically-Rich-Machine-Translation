# Evaluating Transformer Training Strategies for Arabic–English Machine Translation

This repository contains the experiments, prediction outputs, evaluation artifacts, and manuscript materials for the study:

**Evaluating Transformer Training Strategies for Arabic-English Machine Translation: Effects of Pretraining, Tokenization, and Evaluation**

The study compares scratch Transformer systems with pretrained multilingual and translation-specialized models for Arabic↔English machine translation using the OPUS-100 Arabic–English benchmark.
The manuscript is currently under revision/submission and is not included in the public repository prior to publication.
## Scope

The revised study is limited to **Arabic–English machine translation**. It does not claim direct generalization to other morphologically rich languages.

Systems evaluated include:

- Scratch Transformer with character-level tokenization
- Scratch Transformer with word-level tokenization
- Scratch Transformer with 32k SentencePiece tokenization
- MarianMT
- mT5-base
- M2M-100
- NLLB-200

Evaluation includes:

- BLEU
- chrF
- SBERT
- COMET, where available
- paired bootstrap significance testing
- manual adequacy and fluency evaluation
- morphology-oriented error analysis

## Dataset

OPUS-100 Arabic–English:

- Training: 1,000,000 sentence pairs
- Development: 2,000 sentence pairs
- Test: 2,000 sentence pairs

The test set contains 1,997 unique source-reference pairs and three repeated pairs. New prediction files preserve all 2,000 rows. Legacy sentence-level outputs contain the corresponding 1,997 unique pairs.

## Main Results

### Arabic → English

| System | Setting | BLEU | chrF | SBERT | COMET |
|---|---|---:|---:|---:|---:|
| Scratch Transformer | Character | 36.52 | 57.98 | 0.814 | 0.802 |
| Scratch Transformer | Word | 23.50 | 48.50 | 0.705 | 0.710 |
| Scratch Transformer | SentencePiece 32k | 8.57 | 25.39 | 0.529 | — |
| MarianMT | Unadapted | 44.44 | 64.34 | 0.837 | 0.831 |
| MarianMT | Fine-tuned | 43.47 | 63.37 | 0.835 | 0.828 |
| mT5-base | Fine-tuned | 25.16 | 47.11 | 0.730 | — |
| M2M-100 | Unadapted | 25.33 | 50.38 | 0.749 | 0.757 |
| M2M-100 | Fine-tuned | 32.86 | 55.55 | 0.789 | 0.790 |
| NLLB-200 | Unadapted | 35.80 | 58.45 | 0.808 | 0.810 |
| NLLB-200 | Fine-tuned | 38.51 | 60.40 | 0.821 | 0.821 |

### English → Arabic

| System | Setting | BLEU | chrF | SBERT | COMET |
|---|---|---:|---:|---:|---:|
| Scratch Transformer | Character | 21.89 | 49.06 | 0.847 | 0.791 |
| Scratch Transformer | Word | 0.63 | 11.28 | 0.566 | — |
| Scratch Transformer | SentencePiece 32k | 4.78 | 22.10 | 0.637 | — |
| MarianMT | Unadapted | 26.39 | 52.41 | 0.843 | 0.795 |
| MarianMT | Fine-tuned | 26.08 | 52.49 | 0.851 | 0.806 |
| mT5-base | Fine-tuned | 9.85 | 33.55 | 0.781 | — |
| M2M-100 | Unadapted | 13.02 | 41.24 | 0.803 | 0.755 |
| M2M-100 | Fine-tuned | 18.82 | 45.99 | 0.829 | 0.781 |
| NLLB-200 | Unadapted | 16.26 | 45.86 | 0.843 | 0.801 |
| NLLB-200 | Fine-tuned | 20.90 | 48.74 | 0.847 | 0.804 |

## Tokenization Ablation

Under the evaluated scratch-Transformer settings and training budgets, character-level modeling performs substantially better than word-level and 32k SentencePiece tokenization in both directions.

This result is specific to the evaluated setup and should not be interpreted as a universal claim that character-level tokenization is always superior.

## Statistical Significance

Paired bootstrap resampling was performed using:

- 1,000 bootstrap samples
- seed 42
- corpus BLEU and chrF
- 95% percentile confidence intervals
- finite-sample-corrected two-sided empirical p-values

Character vs. word and character vs. SentencePiece differences are statistically significant in both translation directions.

Fine-tuning effects are model-dependent:

- M2M-100 improves significantly in both directions.
- NLLB-200 improves significantly for Arabic→English.
- MarianMT slightly but significantly decreases for Arabic→English.
- MarianMT shows no statistically significant change for English→Arabic.

The complete significance results are available in:

`analysis/significance/final_paired_bootstrap_significance.csv`

## Manual Evaluation

A manual analysis was performed on 100 sampled translations:

- 50 Arabic→English
- 50 English→Arabic
- 5 representative systems

Overall adequacy / fluency scores:

| System | Adequacy | Fluency |
|---|---:|---:|
| Character scratch | 4.12 | 4.29 |
| SentencePiece scratch | 2.29 | 3.21 |
| MarianMT fine-tuned | 4.31 | 4.36 |
| mT5 fine-tuned | 3.42 | 3.99 |
| NLLB-200 fine-tuned | 4.18 | 4.25 |

The manual study uses a single Arabic-speaking evaluator. It is reported as a focused manual analysis rather than a fully blinded multi-annotator preference study.

The final annotation workbook is available in:

`analysis/human_evaluation/human_morphology_error_analysis_100_final.xlsx`

## Error Analysis

The annotation protocol includes:

`GENDER`, `NUMBER`, `CLITIC`, `VERB`, `DEFINITENESS`, `NOMINAL_AGR`, `MORPH_FORM`, `OMISSION`, `ADDITION`, `LEXICAL`, `WORD_ORDER`, `NAMED_ENTITY`, `IDIOM`, and `NO_MAJOR_ERROR`.

Lexical and omission errors are the most frequent observed failure types. Morphology-specific errors are less frequent in the sampled outputs and mainly involve verb form, number, gender, and nominal agreement.

## Repository Structure

```text
.
.
├── README.md
├── REPRODUCIBILITY.md
├── REPRO_ARTIFACT_CHECKLIST.md
├── LICENSE
├── configs/
│   └── final_experiment_config.yaml
├── analysis/
│   ├── significance/
│   │   └── final_paired_bootstrap_significance.csv
│   └── human_evaluation/
│       └── human_morphology_error_analysis_100_final.xlsx
├── scripts/
│   ├── character/
│   │   └── char_transformer_opus_full_experiment.py
│   ├── main/
│   │   └── Evaluating Transformer.ipynb
│   └── sentencepiece/
│       └── scratch SentencePiece baseline.ipynb
└── predictions/
    ├── ar_en/
    └── en_ar/
```

## Prediction Files

`predictions/ar_en/` contains sentence-level outputs for:

- character scratch Transformer
- word scratch Transformer
- SentencePiece scratch Transformer
- MarianMT unadapted and fine-tuned
- mT5 fine-tuned
- M2M-100 unadapted and fine-tuned
- NLLB-200 unadapted and fine-tuned

`predictions/en_ar/` contains sentence-level outputs for:

- character scratch Transformer
- word scratch Transformer
- SentencePiece scratch Transformer
- MarianMT unadapted and fine-tuned
- mT5 fine-tuned
- M2M-100 unadapted and fine-tuned
- NLLB-200 fine-tuned

The sentence-level unadapted NLLB-200 English→Arabic output was not available and is therefore not included. This is also why no paired unadapted-vs-fine-tuned NLLB-200 significance test is reported for that direction.

## Experiment Sources

The repository contains the original saved experiment sources that were available at the time of archival:

- `scripts/main/Evaluating Transformer.ipynb`
- `scripts/sentencepiece/scratch SentencePiece baseline.ipynb`
- `scripts/character/char_transformer_opus_full_experiment.py`

Some revision experiments were run interactively in Colab and were not preserved as standalone scripts. Their final configurations are documented in:

`configs/final_experiment_config.yaml`

No reconstructed scripts are presented as original experiment code.

## Checkpoints

Large model checkpoints are not committed to this GitHub repository.

The repository instead provides:

- experiment source files where available
- exact final configurations
- sentence-level predictions
- statistical analysis outputs
- human-evaluation annotations

This keeps the repository lightweight while preserving the artifacts required to inspect and reproduce the reported evaluation.

## Important Experimental Notes

- The original near-zero mT5 results were discarded after identifying an invalid training/evaluation run and were replaced with stable reruns.
- mT5 is treated as a pretrained multilingual text-to-text model, not as a translation-specialized zero-shot MT system.
- Pretrained-model comparisons are system-level comparisons. Model size, pretraining corpus size, and pretraining objective are not independently controlled.
- The current results are not based on three independent training runs; unsupported three-run averaging claims from the earlier manuscript were removed.
- The NLLB-200 English→Arabic unadapted sentence-level output was unavailable for paired significance testing.
- New prediction files preserve all 2,000 test rows, while legacy sentence-level outputs contain the 1,997 unique source-reference pairs.

## Reproducibility

See:

- `REPRODUCIBILITY.md` for alignment and evaluation details.
- `REPRO_ARTIFACT_CHECKLIST.md` for the artifact inventory.
- `configs/final_experiment_config.yaml` for final experiment settings.

## License

See [LICENSE](LICENSE).
