// Auto-generated feature selection header
// Generated on: 2026-03-24 19:15:11
// Selected 50 features from 157
// Scaling type: robust (median/IQR)
// Used by ml_inference.cpp for feature projection and normalization

#ifndef FEATURE_SELECT_H
#define FEATURE_SELECT_H

#define SELECTED_FEATURE_COUNT 50
#define ORIGINAL_FEATURE_COUNT 157
#define FEATURE_SCALING_ROBUST 1

static const int SELECTED_INDICES[50] = {
    1,  // [0] gas1_norm_step1
    2,  // [1] gas1_norm_step2
    3,  // [2] gas1_norm_step3
    12,  // [3] gas2_norm_step2
    13,  // [4] gas2_norm_step3
    14,  // [5] gas2_norm_step4
    17,  // [6] gas2_norm_step7
    18,  // [7] gas2_norm_step8
    20,  // [8] cross_ratio_step0
    21,  // [9] cross_ratio_step1
    22,  // [10] cross_ratio_step2
    23,  // [11] cross_ratio_step3
    25,  // [12] cross_ratio_step5
    30,  // [13] diff_step0
    31,  // [14] diff_step1
    32,  // [15] diff_step2
    33,  // [16] diff_step3
    35,  // [17] diff_step5
    40,  // [18] gas1_delta_step0
    42,  // [19] gas1_delta_step2
    63,  // [20] auc2_norm
    64,  // [21] peak_idx1
    70,  // [22] cross_ratio_mean
    71,  // [23] cross_ratio_slope
    75,  // [24] env_gas1_baseline
    89,  // [25] g1g2_ratio_s7
    90,  // [26] g1g2_ratio_s8
    97,  // [27] interact_g2n2×cr1
    98,  // [28] interact_g2n2×crSlope
    99,  // [29] interact_g1n2×diff1
    100,  // [30] interact_slope2×cr0
    101,  // [31] interact_g2d2×cr2
    102,  // [32] interact_g2n3×crMean
    103,  // [33] interact_diff5×cr3
    105,  // [34] interact_g1n2×diff5
    136,  // [35] cr_early_late_diff
    137,  // [36] g1_early_late_ratio
    139,  // [37] g2_early_late_ratio
    142,  // [38] cross_sensor_late_ratio
    145,  // [39] g2n2_div_hum
    146,  // [40] g1n2_div_hum
    147,  // [41] cr0_div_hum
    148,  // [42] diff1_div_hum
    149,  // [43] slope2_div_hum
    150,  // [44] crMean_div_hum
    151,  // [45] g2n2_div_temp
    153,  // [46] fisher_coffee_vs_dtea
    154,  // [47] fisher_dtea_vs_tea
    155,  // [48] fisher_dcoffee_vs_coffee
    156  // [49] fisher_dtea_vs_dcoffee
};

static const float SELECTED_MEDIANS[50] = {
    1.10357650f,  // [0] gas1_norm_step1
    1.03131870f,  // [1] gas1_norm_step2
    1.11360290f,  // [2] gas1_norm_step3
    1.04978570f,  // [3] gas2_norm_step2
    1.05894830f,  // [4] gas2_norm_step3
    1.10414240f,  // [5] gas2_norm_step4
    1.28304770f,  // [6] gas2_norm_step7
    1.28233130f,  // [7] gas2_norm_step8
    0.83836436f,  // [8] cross_ratio_step0
    0.82672536f,  // [9] cross_ratio_step1
    0.81529522f,  // [10] cross_ratio_step2
    0.87515789f,  // [11] cross_ratio_step3
    0.88392550f,  // [12] cross_ratio_step5
    -0.08580738f,  // [13] diff_step0
    -0.10735825f,  // [14] diff_step1
    -0.10034558f,  // [15] diff_step2
    -0.07160041f,  // [16] diff_step3
    -0.07654053f,  // [17] diff_step5
    0.10357654f,  // [18] gas1_delta_step0
    0.08958507f,  // [19] gas1_delta_step2
    10.59864000f,  // [20] auc2_norm
    1.00000000f,  // [21] peak_idx1
    0.87075770f,  // [22] cross_ratio_mean
    0.00711281f,  // [23] cross_ratio_slope
    1.09002690f,  // [24] env_gas1_baseline
    1.05426240f,  // [25] g1g2_ratio_s7
    1.07181120f,  // [26] g1g2_ratio_s8
    0.87392730f,  // [27] interact_g2n2×cr1
    0.00749157f,  // [28] interact_g2n2×crSlope
    -0.11394840f,  // [29] interact_g1n2×diff1
    0.03000608f,  // [30] interact_slope2×cr0
    0.01011654f,  // [31] interact_g2d2×cr2
    0.93948823f,  // [32] interact_g2n3×crMean
    -0.06779818f,  // [33] interact_diff5×cr3
    -0.07929776f,  // [34] interact_g1n2×diff5
    0.05932677f,  // [35] cr_early_late_diff
    0.75887734f,  // [36] g1_early_late_ratio
    0.81064373f,  // [37] g2_early_late_ratio
    1.05446990f,  // [38] cross_sensor_late_ratio
    0.02390625f,  // [39] g2n2_div_hum
    0.02320085f,  // [40] g1n2_div_hum
    0.01889534f,  // [41] cr0_div_hum
    -0.00213956f,  // [42] diff1_div_hum
    0.00079673f,  // [43] slope2_div_hum
    0.01957934f,  // [44] crMean_div_hum
    0.03769124f,  // [45] g2n2_div_temp
    -0.90539521f,  // [46] fisher_coffee_vs_dtea
    -0.06380306f,  // [47] fisher_dtea_vs_tea
    -0.08736777f,  // [48] fisher_dcoffee_vs_coffee
    0.86875999f  // [49] fisher_dtea_vs_dcoffee
};

static const float SELECTED_IQRS[50] = {
    0.05806732f,  // [0] gas1_norm_step1
    0.07416993f,  // [1] gas1_norm_step2
    0.12504804f,  // [2] gas1_norm_step3
    0.05992603f,  // [3] gas2_norm_step2
    0.09189618f,  // [4] gas2_norm_step3
    0.12550378f,  // [5] gas2_norm_step4
    0.30138218f,  // [6] gas2_norm_step7
    0.29803228f,  // [7] gas2_norm_step8
    0.13628846f,  // [8] cross_ratio_step0
    0.12197995f,  // [9] cross_ratio_step1
    0.12539923f,  // [10] cross_ratio_step2
    0.11785066f,  // [11] cross_ratio_step3
    0.10298264f,  // [12] cross_ratio_step5
    0.08368659f,  // [13] diff_step0
    0.09926772f,  // [14] diff_step1
    0.07981455f,  // [15] diff_step2
    0.07668516f,  // [16] diff_step3
    0.08795103f,  // [17] diff_step5
    0.05806732f,  // [18] gas1_delta_step0
    0.04423904f,  // [19] gas1_delta_step2
    1.85411360f,  // [20] auc2_norm
    0.22222221f,  // [21] peak_idx1
    0.10143900f,  // [22] cross_ratio_mean
    0.00662283f,  // [23] cross_ratio_slope
    0.06005859f,  // [24] env_gas1_baseline
    0.07392907f,  // [25] g1g2_ratio_s7
    0.07899094f,  // [26] g1g2_ratio_s8
    0.16948801f,  // [27] interact_g2n2×cr1
    0.00660684f,  // [28] interact_g2n2×crSlope
    0.09706225f,  // [29] interact_g1n2×diff1
    0.03408011f,  // [30] interact_slope2×cr0
    0.03203933f,  // [31] interact_g2d2×cr2
    0.12904596f,  // [32] interact_g2n3×crMean
    0.07055380f,  // [33] interact_diff5×cr3
    0.09002888f,  // [34] interact_g1n2×diff5
    0.04975772f,  // [35] cr_early_late_diff
    0.14040351f,  // [36] g1_early_late_ratio
    0.14943790f,  // [37] g2_early_late_ratio
    0.06353462f,  // [38] cross_sensor_late_ratio
    0.00901230f,  // [39] g2n2_div_hum
    0.00835567f,  // [40] g1n2_div_hum
    0.00934020f,  // [41] cr0_div_hum
    0.00254070f,  // [42] diff1_div_hum
    0.00061093f,  // [43] slope2_div_hum
    0.00909791f,  // [44] crMean_div_hum
    0.00448413f,  // [45] g2n2_div_temp
    3.87823410f,  // [46] fisher_coffee_vs_dtea
    5.66009620f,  // [47] fisher_dtea_vs_tea
    3.56623320f,  // [48] fisher_dcoffee_vs_coffee
    4.00050450f  // [49] fisher_dtea_vs_dcoffee
};

// Normalize a selected feature using robust scaling
static inline float normalize_selected_feature(float value, int selected_idx) {
    if (selected_idx < 0 || selected_idx >= 50) return value;
    return (value - SELECTED_MEDIANS[selected_idx]) / SELECTED_IQRS[selected_idx];
}

#endif // FEATURE_SELECT_H
