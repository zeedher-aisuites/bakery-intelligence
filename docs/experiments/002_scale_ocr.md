# Experiment 002 — Digital Scale Reading with Muse Spark

## Objective

Evaluate whether Meta Muse Spark 1.3 can extract numerical weights from photographs of digital kitchen scales in bakery environments.

## Dataset

- 10 synthetic photographs of digital kitchen scales.
- Different ingredients, containers, viewing angles, and lighting conditions.
- Reference weights stored in `ground_truth.csv`.
- One low-contrast example included to assess visual difficulty.

## Methodology

Each image was submitted independently to Muse Spark 1.3 using the Meta Model API.

The model was instructed to read the numerical weight and unit displayed on the scale, without estimating the weight from the ingredients or container.

Predictions were compared against the reference values using exact numerical matching after parsing.

## Results

Two evaluation runs were completed using the same 10 images.

| Metric | Run 1 | Run 2 |
|---|---:|---:|
| Images | 10 | 10 |
| Correct predictions | 10 | 10 |
| Accuracy | 100% | 100% |
| Total tokens | 22,969 | 23,102 |
| Average latency | 6.40 s | 5.94 s |
| Estimated API cost | Not recorded | $0.02861 USD |

The second run had an estimated average cost of $0.002861 USD per image.

The Meta usage dashboard showed approximately $0.03 USD of consumption, consistent with the estimate for the second run.

## Limitations

- All 10 images are synthetic.
- Both runs reused the same images.
- The dataset is too small to estimate real-world reliability.
- The evaluation does not yet measure performance on actual bakery photographs.
- The model has not yet been tested on unreadable displays with no recoverable weight.
- API costs are estimates based on token usage and configured pricing assumptions.

## Conclusion

Muse Spark 1.3 successfully extracted all 10 reference weights in both evaluation runs.

These results support the technical feasibility of using multimodal AI for digital scale reading in Bakery Intelligence.

Further evaluation with real photographs, unreadable displays, and a larger dataset is required before making production-level accuracy claims.

## Next Experiment

Experiment 003 will investigate extracting structured bakery recipes from preparation videos, including ingredients, quantities, process steps, and information that requires human verification.