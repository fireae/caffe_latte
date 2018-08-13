import os
import sys
p = os.path.abspath(__file__)
base, _ = os.path.split(p)
print(base)
sys.path.insert(0, base+'/..')