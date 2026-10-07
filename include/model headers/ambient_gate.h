#ifndef AMBIENT_GATE_H
#define AMBIENT_GATE_H

// ambient gate: report ambient without classifying if computeAmbientScore() > AMBIENT_GATE_SCORE
// score = 1 - min(mean |R/R_base - 1| / 0.15, 1); threshold = 1.5 x the largest mean
// deviation of the 30 calibration sweeps (0.05578)
#define AMBIENT_GATE_MAX_DEVIATION 0.0836657286f
#define AMBIENT_GATE_SCORE 0.442228496f

#endif
