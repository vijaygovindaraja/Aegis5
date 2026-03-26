# Aegis-5: A Hybrid Ensemble Framework for Intrusion Detection in Industry 5.0 Driven Smart Manufacturing Environment

**Published in:** ACM Transactions on Autonomous and Adaptive Systems, 2026

**DOI:** [10.1145/3787224](https://doi.org/10.1145/3787224)

## Authors

- **Vijay Govindarajan** — Colorado State University, Fort Collins, CO, USA
- **Faraz Ahmed** — Crisp Technologies LLC, USA
- **Zaid Bin Faheem** — Wuhan University, Wuhan, China
- **Muhammad Bilal** — Rawalpindi Women University, Pakistan
- **Manel Ayadi** — Princess Nourah Bint Abdulrahman University, Riyadh, Saudi Arabia
- **Jehad Ali** — Ajou University, Suwon, South Korea

## Abstract

Industry 5.0 represents a transformative paradigm that emphasizes synergy between human expertise, intelligent systems, and hyper-connected cyber-physical environments. While this evolution fosters personalized automation and resilient production, it also amplifies the cybersecurity risks inherent in Industrial Internet of Things (IIoT) infrastructures.

We present **Aegis-5**, a novel adaptive hybrid ensemble framework explicitly designed for intrusion detection in Industry 5.0-enabled smart manufacturing ecosystems. The proposed model integrates five diverse classifiers — Random Forest, Gradient Boosting, XGBoost, SVM, and K-Nearest Neighbors — using a dynamic weighting strategy guided by per-class precision, recall, and F1-score performance in real time. A meta-learner further synthesizes these predictions to enhance robustness against sophisticated and zero-day attacks.

We evaluate the model using two benchmark IIoT datasets: **IoT-23** and **CIC-IoT 2023**, both of which capture a broad spectrum of real-world industrial threats. Experimental results demonstrate that our framework achieves superior performance, with accuracy rates of **99.98%** on IoT-23 and **99.95%** on CIC-IoT 2023, coupled with precision (99.97%, 99.93%), recall (99.96%, 99.92%), and F1-score (99.96%, 99.93%) respectively.

## Key Contributions

- Dynamic Hybrid Ensemble Framework integrating five classifiers with real-time adaptive weighting
- Meta-learning via Logistic Regression with hybrid soft/hard voting for robustness against zero-day attacks
- Evaluation on real-world IIoT datasets (IoT-23 and CIC-IoT 2023) with advanced preprocessing
- State-of-the-art detection accuracy (99.98% on IoT-23, 99.94% on CIC-IoT 2023)
- Scalable, latency-sensitive solution aligned with Industry 5.0 cyber-physical requirements

## Keywords

Industry 5.0, Smart Manufacturing, Hybrid Ensemble Framework, Real-Time Threat Detection, IIoT Security, Cyber-Physical Systems, Adversarial Training, Meta-Learning

## Citation

```bibtex
@article{govindarajan2026aegis5,
  title={Aegis-5: A Hybrid Ensemble Framework for Intrusion Detection in Industry 5.0 Driven Smart Manufacturing Environment},
  author={Govindarajan, Vijay and Ahmed, Faraz and Faheem, Zaid Bin and Bilal, Muhammad and Ayadi, Manel and Ali, Jehad},
  journal={ACM Transactions on Autonomous and Adaptive Systems},
  year={2026},
  doi={10.1145/3787224},
  publisher={ACM}
}
```

## Paper

The full paper is available at [ACM Digital Library](https://dl.acm.org/doi/10.1145/3787224).
