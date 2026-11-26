# Round 1 Baseline Evaluation (`val` and `test` Separate)

Fitted strictly on the `train` split (first 82% = 5,091 days; `val` = 558 days; `test` = 560 days).
Windows are non-overlapping (`min_separation_days = 13`, >= 10 days apart) and selected strictly inside each split.

## Heavy-Rain Cell Fraction (`tp >= 10.0 mm/day`)
- `train` (5091 days): `0.111691` (11.169%)
- `val` (558 days): `0.088646` (8.865%)
- `test` (560 days): `0.131419` (13.142%)
- `held-out (val + test)` (1118 days): `0.110071` (11.007%)

## Split: `val` (50 windows)

### Window Counts by Category (non-overlapping, >= 13 days apart)
| Category | Count | Shortfall |
|---|---:|---:|
| rain_onset | 10 | 0 |
| heavy_rain | 10 | 0 |
| heatwave | 10 | 0 |
| cold_disturbance | 10 | 0 |
| random | 10 | 0 |

### Lat-Weighted RMSE & Best Baseline by Lead
#### Lead Day 1
| Channel | Persistence | Clim (Harmonic) | Clim (Empirical) | Best (All 3) | Best (Pers vs Harm) | Harm - Emp |
|---|---:|---:|---:|---|---|---:|
| `t2m` | 1.1800 | 1.7399 | 1.7591 | persistence | persistence | -0.0192 |
| `tp` | 5.5535 | 6.4983 | 6.2492 | persistence | persistence | +0.2491 |
| `u10` | 1.1170 | 1.6464 | 1.6475 | persistence | persistence | -0.0011 |
| `v10` | 1.1020 | 1.5036 | 1.5080 | persistence | persistence | -0.0043 |
| `msl` | 215.2489 | 280.4111 | 285.2104 | persistence | persistence | -4.7993 |
| `tcwv` | 3.7290 | 6.0704 | 6.0016 | persistence | persistence | +0.0688 |

#### Lead Day 4
| Channel | Persistence | Clim (Harmonic) | Clim (Empirical) | Best (All 3) | Best (Pers vs Harm) | Harm - Emp |
|---|---:|---:|---:|---|---|---:|
| `t2m` | 2.2199 | 1.8306 | 1.8419 | clim_harmonic | clim_harmonic | -0.0113 |
| `tp` | 8.8301 | 6.9553 | 6.5615 | clim_empirical | clim_harmonic | +0.3939 |
| `u10` | 1.9467 | 1.5982 | 1.5946 | clim_empirical | clim_harmonic | +0.0036 |
| `v10` | 1.9273 | 1.5059 | 1.5070 | clim_harmonic | clim_harmonic | -0.0011 |
| `msl` | 384.1142 | 269.2877 | 273.1456 | clim_harmonic | clim_harmonic | -3.8579 |
| `tcwv` | 7.0901 | 6.0674 | 5.9967 | clim_empirical | clim_harmonic | +0.0707 |

#### Lead Day 8
| Channel | Persistence | Clim (Harmonic) | Clim (Empirical) | Best (All 3) | Best (Pers vs Harm) | Harm - Emp |
|---|---:|---:|---:|---|---|---:|
| `t2m` | 2.3574 | 1.7254 | 1.7486 | clim_harmonic | clim_harmonic | -0.0231 |
| `tp` | 11.0275 | 9.4061 | 8.8968 | clim_empirical | clim_harmonic | +0.5093 |
| `u10` | 2.5108 | 1.8480 | 1.8415 | clim_empirical | clim_harmonic | +0.0064 |
| `v10` | 2.4551 | 1.7978 | 1.7903 | clim_empirical | clim_harmonic | +0.0075 |
| `msl` | 417.1239 | 280.3403 | 284.9799 | clim_harmonic | clim_harmonic | -4.6397 |
| `tcwv` | 8.3061 | 6.2909 | 6.2225 | clim_empirical | clim_harmonic | +0.0684 |

### Heavy-Rain CSI (>= 10.0 mm/day) & Event Counts
| Lead | Baseline | CSI | Hits | Misses | False Alarms | True Events (>=10 mm) | Pred Events (>=10 mm) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | persistence | 0.4256 | 3111 | 2257 | 1942 | 5368 | 5053 |
| 1 | clim_harmonic | 0.1972 | 1178 | 4190 | 606 | 5368 | 1784 |
| 1 | clim_empirical | 0.3101 | 2455 | 2913 | 2550 | 5368 | 5005 |
| 4 | persistence | 0.2062 | 1880 | 4064 | 3173 | 5944 | 5053 |
| 4 | clim_harmonic | 0.1831 | 1204 | 4740 | 631 | 5944 | 1835 |
| 4 | clim_empirical | 0.3342 | 2742 | 3202 | 2261 | 5944 | 5003 |
| 8 | persistence | 0.1997 | 1983 | 4876 | 3070 | 6859 | 5053 |
| 8 | clim_harmonic | 0.1781 | 1324 | 5535 | 573 | 6859 | 1897 |
| 8 | clim_empirical | 0.3267 | 2946 | 3913 | 2158 | 6859 | 5104 |

### Day-8 Spatial Variance Ratio (`var(pred) / var(true)`, target band `0.40 .. 2.50`)
| Channel | Persistence | Clim (Harmonic) | Clim (Empirical) |
|---|---:|---:|---:|
| `t2m` | 1.0495 | 0.9614 | 0.9621 |
| `tp` | 1.8416 | 0.1293 | 0.3717 |
| `u10` | 0.9217 | 0.6255 | 0.6263 |
| `v10` | 0.9432 | 0.6025 | 0.5994 |
| `msl` | 1.2386 | 0.8211 | 0.8276 |
| `tcwv` | 0.9815 | 0.8760 | 0.8759 |

## Split: `test` (50 windows)

### Window Counts by Category (non-overlapping, >= 13 days apart)
| Category | Count | Shortfall |
|---|---:|---:|
| rain_onset | 10 | 0 |
| heavy_rain | 10 | 0 |
| heatwave | 10 | 0 |
| cold_disturbance | 10 | 0 |
| random | 10 | 0 |

### Lat-Weighted RMSE & Best Baseline by Lead
#### Lead Day 1
| Channel | Persistence | Clim (Harmonic) | Clim (Empirical) | Best (All 3) | Best (Pers vs Harm) | Harm - Emp |
|---|---:|---:|---:|---|---|---:|
| `t2m` | 1.0669 | 1.8438 | 1.8355 | persistence | persistence | +0.0083 |
| `tp` | 5.5995 | 6.4370 | 6.2485 | persistence | persistence | +0.1885 |
| `u10` | 1.0470 | 1.7756 | 1.7846 | persistence | persistence | -0.0090 |
| `v10` | 1.0688 | 1.4917 | 1.4985 | persistence | persistence | -0.0068 |
| `msl` | 214.5860 | 288.2427 | 285.6199 | persistence | persistence | +2.6228 |
| `tcwv` | 3.7993 | 7.0320 | 7.0769 | persistence | persistence | -0.0449 |

#### Lead Day 4
| Channel | Persistence | Clim (Harmonic) | Clim (Empirical) | Best (All 3) | Best (Pers vs Harm) | Harm - Emp |
|---|---:|---:|---:|---|---|---:|
| `t2m` | 1.9297 | 1.8576 | 1.8447 | clim_empirical | clim_harmonic | +0.0129 |
| `tp` | 8.9663 | 7.5844 | 7.2272 | clim_empirical | clim_harmonic | +0.3573 |
| `u10` | 2.0658 | 1.6105 | 1.6122 | clim_harmonic | clim_harmonic | -0.0017 |
| `v10` | 1.9708 | 1.5001 | 1.4963 | clim_empirical | clim_harmonic | +0.0038 |
| `msl` | 386.5089 | 286.0280 | 286.1042 | clim_harmonic | clim_harmonic | -0.0762 |
| `tcwv` | 7.4710 | 6.9976 | 7.0727 | clim_harmonic | clim_harmonic | -0.0751 |

#### Lead Day 8
| Channel | Persistence | Clim (Harmonic) | Clim (Empirical) | Best (All 3) | Best (Pers vs Harm) | Harm - Emp |
|---|---:|---:|---:|---|---|---:|
| `t2m` | 2.2584 | 1.9279 | 1.9318 | clim_harmonic | clim_harmonic | -0.0039 |
| `tp` | 10.6877 | 9.2365 | 8.8115 | clim_empirical | clim_harmonic | +0.4250 |
| `u10` | 2.5696 | 1.7983 | 1.7954 | clim_empirical | clim_harmonic | +0.0029 |
| `v10` | 2.1059 | 1.5834 | 1.5766 | clim_empirical | clim_harmonic | +0.0068 |
| `msl` | 451.8677 | 328.4180 | 323.4467 | clim_empirical | clim_harmonic | +4.9714 |
| `tcwv` | 9.0565 | 7.2155 | 7.2669 | clim_harmonic | clim_harmonic | -0.0514 |

### Heavy-Rain CSI (>= 10.0 mm/day) & Event Counts
| Lead | Baseline | CSI | Hits | Misses | False Alarms | True Events (>=10 mm) | Pred Events (>=10 mm) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | persistence | 0.3939 | 3740 | 3065 | 2690 | 6805 | 6430 |
| 1 | clim_harmonic | 0.1408 | 1115 | 5690 | 1115 | 6805 | 2230 |
| 1 | clim_empirical | 0.2237 | 2330 | 4475 | 3612 | 6805 | 5942 |
| 4 | persistence | 0.2089 | 2488 | 5478 | 3942 | 7966 | 6430 |
| 4 | clim_harmonic | 0.1728 | 1501 | 6465 | 720 | 7966 | 2221 |
| 4 | clim_empirical | 0.2848 | 3059 | 4907 | 2774 | 7966 | 5833 |
| 8 | persistence | 0.1714 | 2089 | 5755 | 4341 | 7844 | 6430 |
| 8 | clim_harmonic | 0.1667 | 1432 | 6412 | 747 | 7844 | 2179 |
| 8 | clim_empirical | 0.2995 | 3128 | 4716 | 2599 | 7844 | 5727 |

### Day-8 Spatial Variance Ratio (`var(pred) / var(true)`, target band `0.40 .. 2.50`)
| Channel | Persistence | Clim (Harmonic) | Clim (Empirical) |
|---|---:|---:|---:|
| `t2m` | 0.9849 | 1.0446 | 1.0480 |
| `tp` | 1.0046 | 0.1078 | 0.2718 |
| `u10` | 0.9771 | 0.5719 | 0.5743 |
| `v10` | 0.9954 | 0.6267 | 0.6287 |
| `msl` | 0.9803 | 0.8340 | 0.8370 |
| `tcwv` | 0.9964 | 0.8067 | 0.8088 |
