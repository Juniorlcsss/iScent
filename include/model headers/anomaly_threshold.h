#ifndef ANOMALY_THRESHOLD_H
#define ANOMALY_THRESHOLD_H

// Calibrated from leave-one-out 5-NN mean distances of 583 training samples (55 features)
// anomaly score = min(mean neighbour distance / KNN_DISTANCE_SCALE, 1)
// threshold is in score units (1.5 x the 95th percentile distance)
static const float CALIBRATED_ANOMALY_THRESHOLD = 0.599999964f;
static const float KNN_DISTANCE_SCALE = 8.69076443f;

#endif
