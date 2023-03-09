


# This one does need you to go into the code and decorate with @profile
# what you want profiled
mkdir -p lineprofile_results
sed -ie "s/# @profile/@profile/g" ../diadem/**/*.py
python -m pip install "../.[test,profiling,dev]"
sed -ie "s/@profile/# @profile/g" ../diadem/**/*.py

set -x
set -e

# PYTHONOPTIMIZE=1
mkdir -p lineprofile_results_tims
DEBUG_DIADEM=1 python -m kernprof -l run_profile_tims.py
python -m line_profiler run_profile_tims.py.lprof > "line_profile_tims_$(date '+%Y%m%d_%H%M').txt"
python -m line_profiler run_profile_tims.py.lprof > "line_profile_tims_latest.txt"
python -m pip install -e "../.[test,profiling,dev]"


R -e 'library(tidyverse) ; foo = readr::read_tsv("lineprofile_results_tims/results.diadem.tsv.pin") ; g <- ggplot(foo, aes(x = `Score`, fill = factor(Label))) + geom_density(alpha=0.4) ; ggsave("lineprofile_results_tims/raw_td_plot.png", plot = g)'
R -e 'library(tidyverse) ; foo = readr::read_tsv("lineprofile_results_tims/results.diadem.tsv.pin") ; g <- ggplot(foo, aes(x = seq_along(`Score`), y = Score, colour = factor(Label))) + geom_point(alpha=0.4) ; ggsave("lineprofile_results_tims/iter_score_plot.png", plot = g)'

mokapot lineprofile_results_tims/results.diadem.tsv.pin --test_fdr 0.05 --keep_decoys -d lineprofile_results_tims
R -e 'library(tidyverse) ; foo = readr::read_tsv("lineprofile_results_tims/mokapot.peptides.txt") ; foo2 = readr::read_tsv("mokapot.decoy.peptides.txt") ; g <- ggplot(bind_rows(foo, foo2), aes(x = `mokapot score`, fill = Label)) + geom_density(alpha=0.4) ; ggsave("lineprofile_results_tims/td_plot.png", plot = g)'
