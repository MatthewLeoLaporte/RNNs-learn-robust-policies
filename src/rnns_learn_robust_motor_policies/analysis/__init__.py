

from .part1 import ANALYSIS_SETS as ANALYSIS_SETS_PART1
from .part2 import ANALYSIS_SETS as ANALYSIS_SETS_PART2

EVAL_N_DIRECTIONS = 1
EVAL_REACH_LENGTH = 0.0 

# TODO: This isn't ideal; an alternative would be to pass around *module* keys
# (e.g. `"part1.plant_perts"`) and then load the required attributes from those modules, instead of 
# loading from a central location based on an id (e.g. `"1-1"`)
ANALYSIS_SETS = ANALYSIS_SETS_PART1 | ANALYSIS_SETS_PART2



