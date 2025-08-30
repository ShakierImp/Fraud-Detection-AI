# FraudGuardian AI  
## Credits and References

This document acknowledges and attributes all external resources, datasets, libraries, research papers, tutorials, and tools that supported the development of the **FraudGuardian AI** fraud detection system. Proper attribution ensures compliance with open-source licenses, academic integrity, and ethical project practices.

---

## 1. Dataset Attribution

### Public Datasets
| Dataset | Source & Link | License | Notes |
|---------|---------------|---------|-------|
| Credit Card Fraud Detection Dataset | [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) | [Open Data License](https://opendatacommons.org/licenses/odbl/) | Real-world anonymized credit card transactions used for benchmarking fraud detection models. |
| IEEE-CIS Fraud Detection Dataset | [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection) | Kaggle Terms of Service | Large-scale dataset combining transaction and identity information for fraud detection research. |

### Synthetic Data Generation
- **Method:** Synthetic data created using statistical sampling and bootstrapping techniques to augment rare fraud cases.  
- **Tools:** [scikit-learn `make_classification`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html), NumPy random sampling.  
- **Purpose:** To balance class distributions and stress-test models under varied fraud prevalence scenarios.  

---

## 2. Library Dependencies

| Library | Version Range | Purpose | License | URL |
|---------|---------------|---------|---------|-----|
| NumPy | >=1.24 | Numerical computations and array handling | BSD | [numpy.org](https://numpy.org) |
| pandas | >=2.0 | Data manipulation and preprocessing | BSD | [pandas.pydata.org](https://pandas.pydata.org) |
| scikit-learn | >=1.3 | Machine learning algorithms, metrics, and preprocessing | BSD | [scikit-learn.org](https://scikit-learn.org) |
| matplotlib | >=3.7 | Data visualization and plotting | PSF | [matplotlib.org](https://matplotlib.org) |
| seaborn | >=0.12 | Statistical visualization (EDA) | BSD | [seaborn.pydata.org](https://seaborn.pydata.org) |
| PyCaret | >=3.0 | Low-code machine learning experimentation | MIT | [pycaret.org](https://pycaret.org) |
| xgboost | >=1.7 | Gradient boosting for fraud classification | Apache-2.0 | [xgboost.ai](https://xgboost.ai) |
| lightgbm | >=4.0 | Gradient boosting optimized for speed and memory | MIT | [lightgbm.readthedocs.io](https://lightgbm.readthedocs.io) |
| TensorFlow | >=2.12 | Neural network modeling | Apache-2.0 | [tensorflow.org](https://www.tensorflow.org) |
| PyTorch | >=2.0 | Deep learning and anomaly detection research | BSD | [pytorch.org](https://pytorch.org) |

---

## 3. Research Papers

### Fraud Detection Methodologies
- Dal Pozzolo, A., et al. (2015). **Calibrating Probability with Undersampling for Unbalanced Classification.** *IEEE Symposium on Computational Intelligence and Data Mining (CIDM).*  
- Carcillo, F., et al. (2019). **Combining Unsupervised and Supervised Learning in Credit Card Fraud Detection.** *Information Sciences, 557, 317-331.*

### Anomaly Detection Techniques
- Chandola, V., Banerjee, A., & Kumar, V. (2009). **Anomaly Detection: A Survey.** *ACM Computing Surveys (CSUR), 41(3), 1-58.*  
- Schölkopf, B., et al. (2001). **Estimating the Support of a High-Dimensional Distribution.** *Neural Computation, 13(7), 1443-1471.*

### Ensemble Learning
- Dietterich, T. G. (2000). **Ensemble Methods in Machine Learning.** *International Workshop on Multiple Classifier Systems.*  
- Breiman, L. (2001). **Random Forests.** *Machine Learning, 45(1), 5-32.*  

---

## 4. Tutorials & Guides

- [Kaggle Notebooks on Fraud Detection](https://www.kaggle.com/code) – Used as reference for baseline feature engineering and model benchmarking.  
- PyCaret official documentation: [PyCaret Docs](https://pycaret.gitbook.io/docs/) – For experimentation with classification workflows.  
- Towards Data Science Blog (selected articles on fraud detection and anomaly detection).  

---

## 5. Tools & Services

### Development Tools
- **IDE:** [Visual Studio Code](https://code.visualstudio.com/) (MIT License)  
- **Jupyter Notebook** for data exploration and EDA (BSD License).  

### Cloud & Deployment
- **Google Colab** for GPU experimentation (Google Terms of Service).  
- **Netlify** for hosting frontend demos (Free tier license).  

### Testing & CI/CD
- **pytest** (MIT License) for unit testing.  
- **GitHub Actions** for automated CI/CD workflows.  

---

## 6. Acknowledgements

This project was made possible by the open-source community, academic researchers, and data science practitioners who contributed foundational methods, datasets, and tools.

---
