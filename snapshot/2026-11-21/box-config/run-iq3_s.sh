#!/bin/sh
cd "/home/bonfire/.local/share/strata/terminal/0.1.62"
exec "/home/bonfire/.local/share/strata/terminal/0.1.62/python/bin/python3.12" "/home/bonfire/.local/share/strata/terminal/0.1.62/serve/server.py" "--engine" "strata" "--config" "/home/bonfire/.local/share/strata/install/strata-iq3_s.json" "--port" "8080" "--open"
