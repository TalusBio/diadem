
# This one does need you to go into the code and decorate with @profile
# what you want profiled
mkdir -p lineprofile_results
sed -ie "s/# @profile/@profile/g" ../diadem/**/*.py
PYTHONOPTIMIZE=1 python -m kernprof -l run_profile.py
python -m line_profiler run_profile.py.lprof > "line_profile_$(date '+%Y%m%d_%H%M').txt"
python -m line_profiler run_profile.py.lprof > "line_profile_latest.txt"
sed -ie "s/@profile/# @profile/g" ../diadem/**/*.py

# python -m cProfile -s tottime run_profile.py > profile_singlethread.txt
