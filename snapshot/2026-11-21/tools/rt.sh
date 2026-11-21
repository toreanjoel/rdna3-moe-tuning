#!/bin/bash
# rt.sh — run a command on the 7900XTX box (bonfire@10.0.0.51) over key-auth SSH.
# usage: snapshot/2026-11-21/tools/rt.sh 'uname -a'   (key-only SSH; see README)
ssh -o BatchMode=yes -o ConnectTimeout=10 bonfire@10.0.0.51 "$@"
