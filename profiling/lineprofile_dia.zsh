
# This one does need you to go into the code and decorate with @profile
# what you want profiled
mkdir -p lineprofile_results
sed -ie "s/# @profile/@profile/g" ../diadem/**/*.py
python -m pip install ../.
sed -ie "s/@profile/# @profile/g" ../diadem/**/*.py

set -x
set -e

# PYTHONOPTIMIZE=1
DEBUG_DIADEM=1 python -m kernprof -l run_profile.py
python -m line_profiler run_profile.py.lprof > "line_profile_$(date '+%Y%m%d_%H%M').txt"
python -m line_profiler run_profile.py.lprof > "line_profile_latest.txt"

mokapot lineprofile_results/results.diadem.tsv.pin --test_fdr 0.05 --keep_decoys
R -e 'library(tidyverse) ; foo = readr::read_tsv("mokapot.peptides.txt") ; foo2 = readr::read_tsv("mokapot.decoy.peptides.txt") ; g <- ggplot(bind_rows(foo, foo2), aes(x = `mokapot score`, fill = Label)) + geom_density(alpha=0.4) ; ggsave("td_plot.png", plot = g)'

# 20230207 6pm  Elapsed time: 1571.8632419109344
# 2023-02-07 20:02:42.108 | INFO     | diadem.search.diadem:diadem_main:284 - Elapsed time: 1022.3919858932495

# Python deisotoping
# 2023-02-08 20:46:37.470 | INFO     | diadem.search.diadem:diadem_main:290 - Elapsed time: 313.4971549510956
# [INFO]   - 83 target PSMs and 20 decoy PSMs detected.

# DEBUG_DIADEM=1 PYTHONOPTIMIZE=1  python -m cProfile -s tottime run_profile.py > profile_singlethread.txt
# 2023-02-07 18:48:02.815 | INFO     | diadem.search.diadem:diadem_main:284 - Elapsed time: 572.985399723053

# Changing num_decimal to 2 from 3 ... will make buckets larger ...
# Changing number of isotopes
