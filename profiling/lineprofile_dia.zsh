
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

R -e 'library(tidyverse) ; foo = readr::read_tsv("lineprofile_results/results.diadem.tsv.pin") ; g <- ggplot(foo, aes(x = `Score`, fill = factor(Label))) + geom_density(alpha=0.4) ; ggsave("lineprofile_results/raw_td_plot.png", plot = g)'
R -e 'library(tidyverse) ; foo = readr::read_tsv("lineprofile_results/results.diadem.tsv.pin") ; g <- ggplot(foo, aes(x = seq_along(`Score`), y = Score, colour = factor(Label))) + geom_point(alpha=0.4) ; ggsave("lineprofile_results/iter_score_plot.png", plot = g)'

mokapot lineprofile_results/results.diadem.tsv.pin --test_fdr 0.01 --keep_decoys
R -e 'library(tidyverse) ; foo = readr::read_tsv("mokapot.peptides.txt") ; foo2 = readr::read_tsv("mokapot.decoy.peptides.txt") ; g <- ggplot(bind_rows(foo, foo2), aes(x = `mokapot score`, fill = Label)) + geom_density(alpha=0.4) ; ggsave("lineprofile_results/td_plot.png", plot = g)'

# 20230207 6pm  Elapsed time: 1571.8632419109344
# 2023-02-07 20:02:42.108 | INFO     | diadem.search.diadem:diadem_main:284 - Elapsed time: 1022.3919858932495

# Python deisotoping
# 2023-02-08 20:46:37.470 | INFO     | diadem.search.diadem:diadem_main:290 - Elapsed time: 313.4971549510956
# [INFO]   - 83 target PSMs and 20 decoy PSMs detected.

# Sage Deisotoping
# 2023-02-09 17:07:52.691 | INFO     | diadem.search.diadem:diadem_main:289 - Elapsed time: 193.32771015167236
# [INFO]   - 180 target PSMs and 40 decoy PSMs detected.

# Changing score tracking from objects to list and increased number of fails
# [INFO]  - Found 356 peptides with q<=0.05
# 2023-02-10 13:26:32.888 | INFO     | diadem.search.diadem:diadem_main:290 - Elapsed time: 698.4216511249542

# DEBUG_DIADEM=1 PYTHONOPTIMIZE=1  python -m cProfile -s tottime run_profile.py > profile_singlethread.txt
# 2023-02-07 18:48:02.815 | INFO     | diadem.search.diadem:diadem_main:284 - Elapsed time: 572.985399723053

# Changing num_decimal to 2 from 3, this will make buckets larger
# Changing number of isotopes
