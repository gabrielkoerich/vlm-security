import os
import tempfile

# Hermetic: no local config file and no run directory outside a temp dir, set before vlmsec loads
os.environ["HOME_NETWORK_CONF"] = os.path.join(tempfile.mkdtemp(), "absent.conf")
os.environ["VLMSEC_RUN"] = tempfile.mkdtemp()
