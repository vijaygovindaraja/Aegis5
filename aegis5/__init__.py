"""Aegis-5: Adaptive Hybrid Ensemble for Intrusion Detection in Industry 5.0.

Reference implementation of the framework introduced in:

    Govindarajan, V., Ahmed, F., Faheem, Z. B., Bilal, M., Ayadi, M., & Ali, J.
    (2026). Aegis-5: A Hybrid Ensemble Framework for Intrusion Detection in
    Industry 5.0 Driven Smart Manufacturing Environment.
    ACM Transactions on Autonomous and Adaptive Systems.
    DOI: 10.1145/3787224
"""

__version__ = "0.1.0"

from .model import Aegis5, DynamicWeightManager

__all__ = ["Aegis5", "DynamicWeightManager", "__version__"]
