#!/bin/sh
cd "~/.local/share/strata/terminal/0.1.62"
exec "~/.local/share/strata/terminal/0.1.62/python/bin/python3.12" "~/.local/share/strata/terminal/0.1.62/serve/server.py" "--engine" "strata" "--config" "~/.local/share/strata/install/strata-iq3_s.json" "--port" "8080" "--open"
