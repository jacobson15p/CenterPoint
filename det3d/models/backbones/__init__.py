import importlib
spconv_spec = importlib.util.find_spec("spconv")
found = spconv_spec is not None

from .dlav0 import DLA,DLASeg
from .pose_dla_dcn import DLASegv2
if found:
    from .scn import SpMiddleResNetFHD
else:
    print("No spconv, sparse convolution disabled!")

