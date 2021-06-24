import sys
import pstats
from pstats import SortKey

p = pstats.Stats(sys.argv[1])
p.sort_stats('tottime').print_stats(5)
