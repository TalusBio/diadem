
PYTHONOPTIMIZE=1 python run_profile_multithread.py

R -e 'library(tidyverse) ; foo = readr::read_tsv("mokapot.peptides.txt") ; foo2 = readr::read_tsv("mokapot.decoy.peptides.txt") ; g <- ggplot(bind_rows(foo, foo2), aes(x = `mokapot score`, fill = Label)) + geom_density(alpha=0.4) ; ggsave("lineprofile_results_multithread/td_plot.png", plot = g)'
R -e 'library(tidyverse) ; foo = readr::read_tsv("lineprofile_results_multithread/results.diadem.tsv.pin") ; g <- ggplot(foo, aes(x = `Score`, fill = factor(Label))) + geom_density(alpha=0.4) ; ggsave("lineprofile_results_multithread/raw_td_plot.png", plot = g)'
R -e 'library(tidyverse) ; foo = readr::read_tsv("lineprofile_results_multithread/results.diadem.tsv.pin") ; g <- ggplot(foo, aes(x = seq_along(`Score`), y = Score, colour = factor(Label))) + geom_point(alpha=0.4) ; ggsave("lineprofile_results_multithread/iter_score_plot.png", plot = g)'

# [INFO]  - Found 26288 peptides with q<=0.01
# 2023-02-13 12:43:55.191 | INFO     | diadem.search.diadem:diadem_main:295 - Elapsed time: 3487.5772829055786