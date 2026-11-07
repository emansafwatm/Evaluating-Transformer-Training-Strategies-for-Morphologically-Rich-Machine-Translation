# Reproducibility Notes

## Evaluation alignment

New prediction files retain all 2,000 OPUS-100 test rows.

Three source-reference pairs are duplicated in the test set, giving 1,997 unique pairs. Legacy sentence-level outputs retained one occurrence per unique pair.

Therefore:

- new-vs-new comparisons may use all 2,000 aligned rows when row order is preserved;
- comparisons involving legacy outputs use the common set of 1,997 unique aligned pairs.

## Paired bootstrap significance

The significance analysis uses:

- SacreBLEU BLEU
- chrF
- 1,000 paired bootstrap samples
- random seed 42
- 95% percentile confidence intervals
- finite-sample-corrected two-sided empirical p-values

The minimum resolvable corrected p-value with 1,000 samples is reported as `p = 0.002`.

## Human evaluation

The manual evaluation contains 100 sampled examples:

- 50 Arabic→English
- 50 English→Arabic

Systems:

- Character scratch Transformer
- SentencePiece scratch Transformer
- MarianMT fine-tuned
- mT5 fine-tuned
- NLLB-200 fine-tuned

Adequacy is judged primarily against the source sentence. The reference translation is used only as secondary context.

Because model identities became known during part of the review workflow, the analysis should not be described as a fully blinded study.

## Known limitations

- single evaluator
- selected systems evaluated manually rather than every experimental system
- no multi-seed training study
- model size and pretraining scale are not independently controlled
- no paired NLLB-200 English→Arabic unadapted-vs-fine-tuned test because the unadapted sentence-level output is unavailable
