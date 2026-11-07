# Reproducibility Artifact Checklist

## Required in GitHub

### Predictions
Place final sentence-level outputs under `predictions/`.

#### Arabic -> English
- `mt5_ar_en_final_predictions.csv`
- `scratch_sp_ar_en_final_predictions.csv`
- character scratch sentence-level predictions
- word scratch sentence-level predictions
- MarianMT unadapted sentence-level predictions
- MarianMT fine-tuned sentence-level predictions
- M2M-100 unadapted sentence-level predictions
- M2M-100 fine-tuned sentence-level predictions
- NLLB-200 unadapted sentence-level predictions
- NLLB-200 fine-tuned sentence-level predictions

#### English -> Arabic
- `mt5_en_ar_final_predictions.csv`
- `scratch_sp_en_ar_final_predictions.csv`
- `scratch_word_en_ar_final_predictions.csv`
- character scratch sentence-level predictions
- MarianMT unadapted sentence-level predictions
- MarianMT fine-tuned sentence-level predictions
- M2M-100 unadapted sentence-level predictions
- M2M-100 fine-tuned sentence-level predictions
- NLLB-200 fine-tuned sentence-level predictions

Note: the unadapted NLLB-200 EN->AR sentence-level file is unavailable. This is documented as a limitation and is why the paired NLLB EN->AR significance test is not reported.

### Analysis
- `analysis/significance/final_paired_bootstrap_significance.csv`
- `analysis/human_evaluation/human_morphology_error_analysis_100_final.xlsx`
- paired-bootstrap analysis script
- manual-evaluation codebook/protocol or documentation

### Configuration
- `configs/final_experiment_config.yaml`

## Optional / external storage

Do not commit large model checkpoints directly to Git unless they are small enough for normal GitHub limits.

Recommended external storage:
- Google Drive
- Hugging Face Hub
- Zenodo

If checkpoints are archived externally, add stable links and hashes to the repository README.

## Do not publish
- broken/invalid original mT5 outputs
- temporary checkpoints
- intermediate debug files
- reviewer correspondence
- private mapping files that are not needed for reproducibility
