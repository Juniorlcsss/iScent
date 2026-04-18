// Auto-generated ensemble weights
// Generated on: 2026-03-24 19:15:11
// Weights are accuracy^4 from training evaluation
// DT accuracy:   ~71.2%
// KNN accuracy:  ~75.3%
// RF accuracy:   ~82.9%
// Hier accuracy: ~58.9%
//
// Ensemble uses confidence-weighted voting:
//   KNN and RF contribute weight * confidence
//   Hierarchical uses soft probability routing
//   DT is heavily damped (no confidence, weak accuracy)

#ifndef ENSEMBLE_WEIGHTS_H
#define ENSEMBLE_WEIGHTS_H

#define ENSEMBLE_WEIGHT_POWER 4
#define ENSEMBLE_HAS_HIERARCHICAL 1

static const float DT_WEIGHT = 0.257467f;   // acc ~71.2%
static const float KNN_WEIGHT = 0.322225f;  // acc ~75.3%
static const float RF_WEIGHT = 0.471770f;   // acc ~82.9%
static const float HIER_WEIGHT = 0.120388f; // acc ~58.9%

// Relative contribution (normalized):
//   DT:   22.0%
//   KNN:  27.5%
//   RF:   40.3%
//   Hier: 10.3%

#endif // ENSEMBLE_WEIGHTS_H
