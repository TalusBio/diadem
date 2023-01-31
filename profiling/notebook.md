
# Notebook on the dicisions and changes behind the speed improvements

## Jan 27 2023.

Looking at the line profile, there are two main function bottlenecks in the DIA dsearch mode.
1. The actual scoring.
2. Preparing the search group.

```
   252         9 1721460673.0 191273408.1     54.3          for group in ss.yield_iso_window_groups(progress=True):
   253         9   54622691.0 6069187.9      1.7              group_db = db.prefilter_ms1(group.precursor_range)
   254         9 1262291517.0 140254613.0     39.8              group_results = search_group(group=group, db=group_db, config=config)
```

Withing the search group prep it is pretty obvious that the slowest part is the deisotoping.

Withing the search there is an 80/20 distribution of most time-consuming tasks.
1. 80 - hyperscore calculation (score_arrays internally, in `/Users/sebastianpaez/git/diadem/diadem/index/indexed_db.py`).
    1. Within this, 40% of the time happens yielding cancidates (actual scoring)
    2. the other 30% is done just collecting the scores in a dict.
2. 20 - acquisition of the window from the highest peak.


```
   624   1240133     464241.0      0.4      0.1              candidates = self.yield_candidates(
   625   1240133     255498.0      0.2      0.0                  ms1_range=ms1_range,
   626   1240133     395995.0      0.3      0.0                  ms2_range=(fragment_mz - ms2_tol, fragment_mz + ms2_tol),
   627                                                       )
   628
>> 629  71169567  330876449.0      4.6     40.4              for seq, frag, series in candidates:
   630                                                           # Should tolerances be checked here?
   631  71169567   19390157.0      0.3      2.4                  dm = frag - fragment_mz
   632  65852039   19541028.0      0.3      2.4                  if abs(dm) <= ms2_tol:
   633  41247941   15045881.0      0.4      1.8                      if seq not in scores:
>> 634  41247941  253230054.0      6.1     30.9                          scores[seq] = PeptideScore(
   635  41247941   10562042.0      0.3      1.3                              seq,
   636  41247941   13874819.0      0.3      1.7                              self.config.ion_series,
   637                                                                   )
   638
   639  65852039   76098139.0      1.2      9.3                      scores[seq].add_peak(series, fragment_mz, fragment_intensity)
   640  71169567   22077743.0      0.3      2.7                  comparissons += 1
```

Attempts:
1. Changing the dict collection to a try-except (slightly better, 6.1 > 6.9ms/call)
