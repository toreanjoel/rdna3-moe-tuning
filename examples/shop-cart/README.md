# shop-cart: a test task for the local model

A tiny Python project with 3 bugs and one missing feature. 6 of 7 tests fail until they are fixed. No dependencies.

```bash
cd examples/shop-cart && python3 -m unittest -v     # before: 6 of 7 fail
```

Point your coding tool at the server (`http://127.0.0.1:8080/v1` for OpenAI-style tools, `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`
for Claude Code), open this folder and give it:

> Run the tests, fix the bugs in cart.py and implement apply_coupon and total so all tests pass. Don't edit the tests.

It passes when `python3 -m unittest` shows 7 OK. To compare with exact answers, run the same task after
`FAST=0 ./strata-rdna3.sh serve`, from a fresh copy (`git checkout examples/shop-cart`). Watch speed meanwhile with
`./strata-rdna3.sh speed`.
