// Auto-generated ensemble weights
// Generated on: 2026-09-23 00:48:49
// Weights are accuracy^4 (cross-validated accuracy on the training partition)
// DT accuracy:   ~65.7%
// KNN accuracy:  ~70.7%
// RF accuracy:   ~77.0%
//
// Ensemble uses confidence-weighted voting:
//   DT, KNN and RF contribute weight * confidence to their class
//   and spread weight * (1 - confidence) over the other classes

#ifndef ENSEMBLE_WEIGHTS_H
#define ENSEMBLE_WEIGHTS_H

#define ENSEMBLE_WEIGHT_POWER 4
#define ENSEMBLE_HAS_HIERARCHICAL 0

static const float DT_WEIGHT=0.186157435f;   // acc ~65.7%
static const float KNN_WEIGHT=0.249145672f;  // acc ~70.7%
static const float RF_WEIGHT=0.351516992f;   // acc ~77.0%

// Relative contribution (normalized):
//   DT:   23.7%
//   KNN:  31.7%
//   RF:   44.7%

#endif // ENSEMBLE_WEIGHTS_H
