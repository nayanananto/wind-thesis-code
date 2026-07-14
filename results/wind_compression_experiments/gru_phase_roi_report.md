# Temporary GRU Phase ROI Report

This file is an isolated ROI check. It does not modify the active HITL phase predictor.

## GRU Final Test Results
 top1_accuracy  top3_accuracy  macro_f1  horizon  history_length                 config_id  n_train  n_eval split
      0.813043       0.979130  0.773675        1               4 gru_h4_e12_u48_d015_lr1e3     6511    1150  test
      0.638261       0.905217  0.535304        3               3 gru_h3_e12_u32_d015_lr1e3     6510    1150  test
      0.482609       0.822609  0.360672        6               3  gru_h3_e8_u24_d010_lr1e3     6507    1150  test
      0.371304       0.724348  0.244864       12               6 gru_h6_e16_u64_d025_lr5e4     6498    1150  test

## GRU vs Transition Comparison
 horizon  top1_accuracy_transition  top1_accuracy_gru  top1_delta_gru_minus_transition  top3_accuracy_transition  top3_accuracy_gru  top3_delta_gru_minus_transition  macro_f1_transition  macro_f1_gru  macro_f1_delta_gru_minus_transition
       1                  0.784091           0.813043                         0.028953                  0.916958           0.979130                         0.062172             0.733105      0.773675                             0.040570
       3                  0.526224           0.638261                         0.112037                  0.803322           0.905217                         0.101896             0.384473      0.535304                             0.150831
       6                  0.342406           0.482609                         0.140203                  0.716418           0.822609                         0.106191             0.193855      0.360672                             0.166817
      12                  0.325684           0.371304                         0.045620                  0.627538           0.724348                         0.096810             0.148144      0.244864                             0.096720

Elapsed seconds: 437.11