import importlib
spconv_spec = importlib.util.find_spec("spconv")
found = spconv_spec is not None

from .dlav0 import DLA
if found:
    from .scn import SpMiddleResNetFHD
else:
    print("No spconv, sparse convolution disabled!")

