### **VAE Results**

| n_steps | FID (↓)   | KID Mean (↓) |
| ------- | --------- | ------------ |
| 0       | 19.38     | 0.00907      |
| 5       | 19.21     | 0.00904      |
| 10      | 19.02     | 0.00905      |
| **20**  | **18.53** | **0.00890**  |
| 50      | 20.43     | 0.01010      |
| 100     | 23.67     | 0.01232      |

---

### **IWAE Results**

| n_steps  | FID (↓)   | KID Mean (↓) |
| -------- | --------- | ------------ |
| 0        | 18.78     | 0.00870      |
| 5        | **18.40** | **0.00869**  |
| 10       | 18.43     | 0.00892      |
| 20       | 19.06     | 0.00936      |
| 50       | 20.97     | 0.01052      |
| 100      | 24.75     | 0.01299      |

---

### **Vamp Prior Results**

| n_steps | FID (↓)   | KID Mean (↓) |
| ------- | --------- | ------------ |
| 0       | 17.98     | 0.00814      |
| 5       | 18.23     | 0.00822      |
| 10      | 18.27     | 0.00828      |
| **20**  | **18.04** | **0.00810**  |
| 50      | 18.55     | 0.00873      |
| 100     | 18.83     | 0.00897      |

---

### **Analysis and Conclusions**

From the experiments with Langevin dynamics (step size 0.001, noise scale 1), we draw a few conclusions:

1. **Effect of Steps (`n_steps`)**:

   * Increasing the number of Langevin steps generally improves FID and KID initially, but excessive steps (≥50) tend to degrade quality for VAE and IWAE.
   * For Vamp Prior, moderate steps (around 20) achieve the best results; higher steps increase KID slightly but keep FID close to optimal.

2. **Model Comparison**:

   * **Vamp Prior consistently outperforms VAE and IWAE** in FID and KID at optimal step counts.
   * IWAE slightly improves over VAE in low-step regimes but worsens quickly with more steps.

3. **Optimal Configurations**:

   * **VAE**: 20 steps
   * **IWAE**: 5 steps
   * **Vamp Prior**: 20 steps
