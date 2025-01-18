# Model Comparison Metrics

## Performance Analysis

The following metrics compare the base Llama 3.2-3B-Instruct model with our fine-tuned version across key metrics:

| Metric | Base Model | Fine-tuned Model | Improvement |
|:-------|:----------:|:----------------:|:-----------:|
| ROUGE-1 (F1) | 0.210 ± 0.081 | 0.554 ± 0.200 | **+163.3%** |
| ROUGE-2 (F1) | 0.099 ± 0.043 | 0.338 ± 0.285 | **+241.8%** |
| ROUGE-L (F1) | 0.153 ± 0.060 | 0.440 ± 0.249 | **+188.3%** |
| Length Ratio | 6.010 ± 2.710 | 1.007 ± 0.417 | **-83.2%** |


## Metric Descriptions

- **ROUGE-1**: Word-level overlap between model output and ground truth
- **ROUGE-2**: Bigram overlap between model output and ground truth
- **ROUGE-L**: Longest common subsequence between model output and ground truth
- **Length Ratio**: Ratio of model response length to ground truth length
