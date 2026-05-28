# Expert1000 Train / Validation / Test Summary

## 1. Expert Demo Split

| split | num | success_rate | avg_return | min_return | max_return |
|---|---:|---:|---:|---:|---:|
| train | 800 | 0.98375 | 4297.89 | 3.88 | 5188.25 |
| val | 100 | 0.99000 | 4362.09 | 55.43 | 5035.77 |
| test | 100 | 0.97000 | 4303.89 | 8.05 | 5240.13 |

## 2. BC Rollout Split

| split | num | success_rate | avg_return | min_return | max_return |
|---|---:|---:|---:|---:|---:|
| train | 800 | 0.9975 | 4390.55 | 4.12 | 5208.17 |
| val | 100 | 1.0000 | 4300.08 | 924.56 | 5055.43 |
| test | 100 | 0.9900 | 4400.07 | 5.52 | 4994.96 |

## 3. Training Loss

| Model | Epochs | First Loss | Last Loss | Min Loss |
|---|---:|---:|---:|---:|
| BC Policy | 120 | 0.06220 | 0.04596 | 0.04596 |
| World Model | 120 | 0.00502 | 0.000197 | 0.000197 |
| Value Model | 80 | 7234.59 | 338.71 | 337.25 |
| Q Model | 80 | 5308.54 | 138.71 | 138.71 |

## 4. Supervised Train / Val / Test MSE

| Model | Train MSE | Val MSE | Test MSE | Target |
|---|---:|---:|---:|---|
| BC Policy | 0.04595 | 0.04770 | 0.04767 | action chunk prediction |
| World Model | 0.0000945 | 0.0001239 | 0.0001400 | future observation prediction |
| Value Model | 3190.69 | 3373.37 | 3156.09 | chunk return / value prediction |
| Q Model | 120.30 | 357.71 | 201.11 | action chunk score prediction |

## 5. Online Evaluation

| Method | Dataset Scale | Success Rate | Average Return | Average Planning Score |
|---|---|---:|---:|---:|
| BC Policy | expert1000 | 1.0 | 4337.64 | - |
| Planning-WV | expert1000 | 1.0 | 4507.41 | 232.77 |

## 6. Main Conclusion

Under the expert1000 setting, the BC Policy achieves 100% success rate and an average return of 4337.64. After training World Model, Value Model, and Q Model on BC rollouts, Planning-WV also achieves 100% success rate and improves the average return to 4507.41. This indicates that with sufficient rollout data and tuned planning parameters, World Model Planning can outperform the direct BC baseline in trajectory quality.

Compared with BC, Planning-WV improves average return by approximately 169.77, corresponding to a relative improvement of about 3.91%.
