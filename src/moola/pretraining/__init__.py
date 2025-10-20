"""Pre-training infrastructure for self-supervised learning.

This module provides utilities for pre-training models on unlabeled data
using various self-supervised objectives like masked autoencoding and
multi-task learning.

Components:
    - data_augmentation: Augmentation strategies for time series
    - masked_lstm_pretrain: Masked LSTM autoencoder pre-training
    - multitask_pretrain: Multi-task BiLSTM encoder pre-training
"""

from .data_augmentation import TimeSeriesAugmenter
from .masked_lstm_pretrain import MaskedLSTMPretrainer
from .multitask_pretrain import MultiTaskBiLSTM, MultiTaskPretrainer

__all__ = [
    "TimeSeriesAugmenter",
    "MaskedLSTMPretrainer",
    "MultiTaskBiLSTM",
    "MultiTaskPretrainer",
]
