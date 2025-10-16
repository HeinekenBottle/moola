# RunPod Training Results - Backup Manifest

**Date:** 2025-10-16
**RunPod Instance:** root@213.173.102.99:27424
**GPU:** NVIDIA GeForce RTX 4090

## Files Downloaded & Verified ✅

### 1. Pre-trained Encoder
- **File:** `models/ts_tcc/pretrained_encoder.pt`
- **Size:** 3.37 MB
- **MD5:** `8e21830740a4f8a85e9da79548a8203c`
- **Status:** ✅ Verified - No NaN values
- **Training:** 11,873 unlabeled samples, 90 epochs, val loss 5.1674
- **Architecture:** CNN [64, 128, 128] + Transformer

### 2. OOF Predictions - Clean Baseline (98 samples)

| Model | File | MD5 | Status |
|-------|------|-----|--------|
| LogReg | `logreg_clean.npy` | `dd740731a7b5c114cc9b948c851d64f8` | ✅ Verified |
| RF | `rf_clean.npy` | `a316bacb85cd4f76ad1a9621d6acdfc7` | ✅ Verified |
| XGB | `xgb_clean.npy` | `710024b19db15ee7c6bbe9c0e05dd5e5` | ✅ Verified |
| SimpleLSTM | `simple_lstm_clean.npy` | `449159aff705fec60b02b13cbf3288ae` | ✅ Verified |
| CNN-Trans | `cnn_transformer_clean.npy` | `a961601ba5cc9e074ac9643a02f95b76` | ✅ Verified |

### 3. OOF Predictions - Augmented (98 samples)

| Model | File | MD5 | Status |
|-------|------|-----|--------|
| LogReg | `logreg_augmented.npy` | `ea1217123c72683a97730a5bc13c5283` | ✅ Verified |
| RF | `rf_augmented.npy` | `657ac80079961f8a68a7e1c8cbbeef28` | ✅ Verified |
| XGB | `xgb_augmented.npy` | `4f4de7b1707fb8ebf3d392b27c79956b` | ✅ Verified |
| SimpleLSTM | `simple_lstm_augmented.npy` | `5868a2e03286d51f106ec1809bdf94eb` | ✅ Verified |
| CNN-Trans | `cnn_transformer_augmented.npy` | `44c62b33735be96accf61ab306d0dc0b` | ✅ Verified |

### 4. Training Logs
- **File:** `logs/runpod_backup/pretrain_tcc_unlabeled.log`
- **Status:** ✅ Downloaded
- **Contains:** Full TS-TCC training trajectory (90 epochs)

## Integrity Checks Passed ✅

- ✅ All OOF files have shape (98, 2) - correct
- ✅ No NaN values in any OOF predictions
- ✅ No Inf values in any OOF predictions
- ✅ Encoder weights contain no NaN values
- ✅ All MD5 checksums match RunPod source files
- ✅ File sizes are reasonable (1.7KB per OOF, 3.4MB encoder)

## Performance Summary

### Clean Baseline Models
1. **SimpleLSTM:** 57.14% (Best)
2. **LogReg:** 56.12%
3. **XGB:** 55.10%
4. **RF:** 46.94%
5. **CNN-Trans:** 46.94%

### Augmented Models (SMOTE + Mixup + Temporal)
1. **CNN-Trans:** 57.14%
2. **XGB:** 54.08%
3. **SimpleLSTM:** 53.06%
4. **RF:** 50.00%
5. **LogReg:** 48.98%

## What's Safe on Local Machine

✅ **All training artifacts downloaded and verified**
✅ **Checksums match RunPod source**
✅ **No corruption detected**
✅ **Ready for next phase of experimentation**

## Next Steps

1. Use pre-trained encoder for fine-tuning CNN-Transformer
2. Train on CleanLab-cleaned dataset (89 samples)
3. Compare performance vs baseline

## Backup Location

All files are in their standard locations:
- `data/oof/*.npy` - OOF predictions
- `models/ts_tcc/pretrained_encoder.pt` - Pre-trained encoder
- `logs/runpod_backup/` - Training logs

**Repository is safe to commit and push.**
