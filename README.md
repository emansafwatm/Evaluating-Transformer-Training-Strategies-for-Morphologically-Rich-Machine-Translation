# Evaluating-Transformer-Training-Strategies-for-Morphologically-Rich-Machine-Translation
This study examines the impact of training strategy and evaluation methodology on machine translation for 
morphologically rich languages, using Arabic↔English translation as a primary case study. We compare transformer 
models trained from scratch with fine-tuned multilingual models—MarianMT, mT5, M2M-100, and NLLB-200—
using the OPUS-100 benchmark. Performance is assessed with both surface-level (BLEU, chrF) and semantic-level 
(SBERT, COMET) metrics. Results reveal that fine-tuned M2M-100 and NLLB-200 models achieve the strongest 
overall performance, while models trained from scratch exhibit lower lexical scores but competitive semantic 
similarity. These findings demonstrate that traditional n-gram metrics undervalue translations in morphologically 
complex languages and highlight the need for semantically aware evaluation frameworks. Beyond Arabic, our analysis 
provides broader insights into how morphology, tokenization, and pretraining interact across low- and mid-resource 
languages in neural MT. 
