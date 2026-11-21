#!/bin/bash
# rt.sh — run a command on the 7900XTX box (user@box) over key-auth SSH.
# usage: snapshot/2026-11-21/tools/rt.sh 'uname -a'   (key-only SSH; see README)
ssh -o BatchMode=yes -o ConnectTimeout=10 user@box "$@"
