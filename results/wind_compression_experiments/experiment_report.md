# Wind Compression Experiment Report

## Saved Files
- `experiment_manifest.json`
- `tuning_results.csv`
- `best_tuning_configs.csv`
- `final_results.csv`
- `final_split_metrics.csv`
- `phase_transition_tuning_results.csv`
- `phase_transition_best_configs.csv`
- `phase_transition_final_results.csv`

## Final Wind-Speed Forecasting Results
| horizon | design | config_id | mae | rmse | smape | skill_vs_persistence | valid_splits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | lstm_compressed_lstm | lstmenc_c8_k12_seq8_u32_huber | 0.3456 | 0.3456 | 27.7366 | -0.3328 | 4 |
| 1 | pca_compressed_lstm | pca_c12_k16_seq12_u48_huber | 0.3954 | 0.3954 | 31.5480 | -2.0577 | 4 |
| 1 | raw_lstm | raw_direct_lb24_u32_huber | 0.4356 | 0.4356 | 34.1893 | -1.3678 | 4 |
| 3 | pca_compressed_lstm | pca_c12_k16_seq12_u48_huber | 0.3335 | 0.3673 | 26.3395 | -0.9063 | 4 |
| 3 | lstm_compressed_lstm | lstmenc_c8_k12_seq8_u32_huber | 0.3978 | 0.4303 | 30.0522 | -0.8921 | 4 |
| 3 | raw_lstm | raw_direct_lb48_u64_huber | 0.4465 | 0.4831 | 34.3456 | -1.3685 | 4 |
| 6 | pca_compressed_lstm | pca_c8_k12_seq8_u32_huber | 0.3906 | 0.4550 | 25.8894 | -0.3662 | 4 |
| 6 | lstm_compressed_lstm | lstmenc_c12_k16_seq12_u48_huber | 0.4410 | 0.5108 | 29.4177 | -0.6017 | 4 |
| 6 | raw_lstm | raw_direct_lb24_u32_huber | 0.4821 | 0.5771 | 32.7846 | -0.6890 | 4 |
| 12 | raw_lstm | raw_direct_lb24_u32_huber | 0.6992 | 0.8866 | 37.6386 | -0.1562 | 4 |
| 12 | lstm_compressed_lstm | lstmenc_c12_k16_seq12_u48_huber | 0.7158 | 0.9299 | 37.7017 | -0.1471 | 4 |
| 12 | pca_compressed_lstm | pca_c12_k16_seq12_u48_huber | 0.8232 | 1.0317 | 46.7958 | -0.4435 | 4 |

## Best Configurations Selected During Tuning
| horizon | design | config_id | mae | rmse | smape | skill_vs_persistence | valid_splits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | lstm_compressed_lstm | lstmenc_c8_k12_seq8_u32_huber | 0.1263 | 0.1263 | 6.9930 | 0.8381 | 2 |
| 3 | lstm_compressed_lstm | lstmenc_c8_k12_seq8_u32_huber | 0.2808 | 0.3515 | 25.4988 | 0.5629 | 2 |
| 6 | lstm_compressed_lstm | lstmenc_c12_k16_seq12_u48_huber | 0.2869 | 0.3361 | 28.4077 | 0.5646 | 2 |
| 12 | lstm_compressed_lstm | lstmenc_c12_k16_seq12_u48_huber | 0.3903 | 0.4802 | 36.3417 | 0.3953 | 2 |
| 1 | pca_compressed_lstm | pca_c12_k16_seq12_u48_huber | 0.2105 | 0.2105 | 13.5158 | 0.6807 | 2 |
| 3 | pca_compressed_lstm | pca_c12_k16_seq12_u48_huber | 0.2514 | 0.2634 | 29.5773 | 0.5469 | 2 |
| 6 | pca_compressed_lstm | pca_c8_k12_seq8_u32_huber | 0.3871 | 0.4549 | 35.1470 | 0.4084 | 2 |
| 12 | pca_compressed_lstm | pca_c12_k16_seq12_u48_huber | 0.4107 | 0.4873 | 38.2719 | 0.3845 | 2 |
| 1 | raw_lstm | raw_direct_lb24_u32_huber | 0.0263 | 0.0263 | 2.3006 | 0.9490 | 2 |
| 3 | raw_lstm | raw_direct_lb48_u64_huber | 0.2076 | 0.2807 | 23.5604 | 0.6049 | 2 |
| 6 | raw_lstm | raw_direct_lb24_u32_huber | 0.3476 | 0.4197 | 37.9365 | 0.3966 | 2 |
| 12 | raw_lstm | raw_direct_lb24_u32_huber | 0.4071 | 0.5273 | 36.0876 | 0.2867 | 2 |

## Final Phase-Transition Results
| horizon | history_length | min_support | top1_accuracy | top3_accuracy | macro_f1 | avg_support | exact_sequence_rate | markov_fallback_rate | n_eval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 6 | 5 | 0.7841 | 0.9170 | 0.7331 | 395.4362 | 0.7220 | 0.2780 | 1144 |
| 3 | 4 | 1 | 0.5262 | 0.8033 | 0.3845 | 330.0420 | 0.9677 | 0.0323 | 1144 |
| 6 | 6 | 5 | 0.3424 | 0.7164 | 0.1939 | 394.5637 | 0.7226 | 0.2774 | 1139 |
| 12 | 6 | 10 | 0.3257 | 0.6275 | 0.1481 | 472.6346 | 0.6364 | 0.3636 | 1133 |

## Phase Configurations Selected During Tuning
| horizon | history_length | min_support | top1_accuracy | top3_accuracy | macro_f1 | avg_support | exact_sequence_rate | markov_fallback_rate | n_eval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 6 | 5 | 0.7684 | 0.9231 | 0.7284 | 314.7212 | 0.6818 | 0.3182 | 1144 |
| 3 | 4 | 1 | 0.5105 | 0.7972 | 0.4207 | 234.5000 | 0.9563 | 0.0437 | 1144 |
| 6 | 6 | 5 | 0.3889 | 0.7006 | 0.2633 | 315.0764 | 0.6813 | 0.3187 | 1139 |
| 12 | 6 | 10 | 0.3389 | 0.6699 | 0.2197 | 386.1209 | 0.5914 | 0.4086 | 1133 |

## Interpretation Notes
- The final wind-speed table uses 4 rolling splits after compact tuning on 2 rolling splits.
- Lower MAE/RMSE/sMAPE is better; `skill_vs_persistence > 0` means better than seasonal persistence.
- In the final runs, compressed LSTM variants outperform raw LSTM at 1h, 3h, and 6h by MAE, while raw LSTM is slightly better at 12h.
- Several final skill scores are negative, so persistence remains a strong baseline. This should be framed as a compression trade-off result, not as a universal accuracy win.
- Phase prediction is strongest at the next-token horizon and degrades naturally at longer horizons, but top-3 accuracy remains useful for review-oriented HITL workflows.