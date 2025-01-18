# Model Comparison Metrics

## Performance Analysis

The following metrics compare the base Llama 3.2-3B-Instruct model with our fine-tuned version across key metrics:

| Metric | Base Model | Fine-tuned Model | Improvement |
|:-------|:----------:|:----------------:|:-----------:|
| ROUGE-1 (F1) | 0.224 ± 0.000 | 0.520 ± 0.000 | **+132.3%** |
| ROUGE-2 (F1) | 0.147 ± 0.000 | 0.204 ± 0.000 | **+39.1%** |
| ROUGE-L (F1) | 0.190 ± 0.000 | 0.360 ± 0.000 | **+89.7%** |
| Length Ratio | 5.705 ± 0.000 | 0.623 ± 0.000 | **-89.1%** |


## Metric Descriptions

- **ROUGE-1**: Word-level overlap between model output and ground truth
- **ROUGE-2**: Bigram overlap between model output and ground truth
- **ROUGE-L**: Longest common subsequence between model output and ground truth
- **Length Ratio**: Ratio of model response length to ground truth length
