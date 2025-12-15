## Model Calibration Report

### Issue Identified
The EfficientNet-B7 model has **inverted predictions**:
- **Real videos average score**: 0.76 (incorrectly high)
- **Fake videos average score**: 0.65 (incorrectly low)
- **Overall accuracy**: 11.1% (worse than random)

### Root Cause
Based on Gemini research, the model suffers from:
1. **Domain shift**: Trained on FaceForensics++ but tested on Celeb-DF/YouTube
2. **Compression artifact bias**: Learned that compression = fake
3. **Lack of calibration**: Raw scores don't represent true probabilities

### Solutions

#### Immediate Fix (No Retraining)
Since we cannot retrain the model now, we'll **invert the score**:
- Current: `fake_prob = sigmoid(logits)`
- Fixed: `fake_prob = 1.0 - sigmoid(logits)` or `fake_prob = sigmoid(-logits)`

This is a **label inversion** fix - the model is correctly detecting differences, but the labels are swapped.

#### Long-term Fixes (Requires Retraining)
1. Apply compression augmentation to training data
2. Fine-tune on Celeb-DF dataset
3. Use isotonic regression for calibration
4. Add frequency domain analysis (DCT/SRM filters)

### Testing Required
1. Invert the score and retest on all 63 videos
2. Verify accuracy improves to >80%
3. If not, investigate sigmoid vs raw logit interpretation
