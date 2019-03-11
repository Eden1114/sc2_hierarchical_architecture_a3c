from a3c import *

import sys
from absl import flags
flags.DEFINE_string('f', '', 'kernel')
flags.FLAGS(sys.argv)

max_epoch = 1000
map_name = 'Simple64'
parallel = 4  # GeForce GTX1080Ti
# parallel = 3  # GeForce GTX1070

run_a3c(max_epoch, map_name, parallel)
