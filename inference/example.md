## Examples of real running commands and output

#### Inference with opencode + custom agent

Command:
```bash
python inference/inference_opencode.py --max-instances 10 --model deepseek/deepseek-v4-flash --agent dt-generation
```

Output:
```bash
Loading dataset: eth-sri/SWT-bench_Lite_bm25_27k_zsb
Total in dataset : 276
Already completed: 0
To process       : 10

============================================================
[1/10] django__django-15202
  Repo: django/django  Commit: 4fd3044c
  Creating workspace ...
  [git] HEAD is now at 4fd3044ca0 Fixed #33368 -- Fixed parse_duration() crash on invalid separators for decimal fractions.
  Running opencode (model=deepseek/deepseek-v4-flash, agent=dt-generation, timeout=Nones) ...
  Log → tmp/workspaces/20260612_161712/django__django-15202_opencode.log

  --- opencode finished (exit=0, elapsed=582s) ---
  Patch: 4002 chars
  ✓ Saved

============================================================
[2/10] django__django-17087
  Repo: django/django  Commit: 4a72da71
  Creating workspace ...
  [git] HEAD is now at 4a72da7100 Refs #27471 -- Made admin's filter choice arrows use cursor pointers.
  Running opencode (model=deepseek/deepseek-v4-flash, agent=dt-generation, timeout=Nones) ...
  Log → tmp/workspaces/20260612_161712/django__django-17087_opencode.log

  --- opencode finished (exit=0, elapsed=478s) ---
  Patch: 938 chars
  ✓ Saved

============================================================
[3/10] sphinx-doc__sphinx-8721
  Repo: sphinx-doc/sphinx  Commit: 82ef497a
  Creating workspace ...
  [git] error: failed to encode 'tests/roots/test-root/wrongenc.inc' from UTF-8 to iso-8859
  [git] error: failed to encode 'tests/roots/test-warnings/wrongenc.inc' from UTF-8 to iso-8859
  [git] error: failed to encode 'tests/roots/test-root/wrongenc.inc' from iso-8859 to UTF-8
  [git] error: failed to encode 'tests/roots/test-warnings/wrongenc.inc' from iso-8859 to UTF-8
  [git] HEAD is now at 82ef497a8 Merge pull request #8702 from tk0miya/4304_linkcheck_same_url
  Running opencode (model=deepseek/deepseek-v4-flash, agent=dt-generation, timeout=Nones) ...
  Log → tmp/workspaces/20260612_161712/sphinx-doc__sphinx-8721_opencode.log

  --- opencode finished (exit=0, elapsed=276s) ---
  Patch: 1838 chars
  ✓ Saved

============================================================
[4/10] scikit-learn__scikit-learn-10508
  Repo: scikit-learn/scikit-learn  Commit: c753b77a
  Creating workspace ...
  [git] HEAD is now at c753b77ac4 DOC mention default base_estimator of AdaBoost (#10501)
  Running opencode (model=deepseek/deepseek-v4-flash, agent=dt-generation, timeout=Nones) ...
  Log → tmp/workspaces/20260612_161712/scikit-learn__scikit-learn-10508_opencode.log

  --- opencode finished (exit=0, elapsed=355s) ---
  Patch: 3221 chars
  ✓ Saved

============================================================
[5/10] django__django-14017
  Repo: django/django  Commit: 466920f6
  Creating workspace ...
  [git] HEAD is now at 466920f6d7 Fixed #32450 -- Fixed crash when ANDing/ORing an empty Q() with not pickleable Q().
  Running opencode (model=deepseek/deepseek-v4-flash, agent=dt-generation, timeout=Nones) ...
  Log → tmp/workspaces/20260612_161712/django__django-14017_opencode.log

  --- opencode finished (exit=0, elapsed=314s) ---
  Patch: 4710 chars
  ✓ Saved

============================================================
[6/10] django__django-11422
  Repo: django/django  Commit: df46b329
  Creating workspace ...
  [git] HEAD is now at df46b329e0 Refs #30485 -- Avoided unnecessary instance checks in urlencode.
  Running opencode (model=deepseek/deepseek-v4-flash, agent=dt-generation, timeout=Nones) ...
  Log → tmp/workspaces/20260612_161712/django__django-11422_opencode.log

  --- opencode finished (exit=0, elapsed=429s) ---
  Patch: 1930 chars
  ✓ Saved

============================================================
[7/10] sympy__sympy-14774
  Repo: sympy/sympy  Commit: 8fc63c2d
  Cloning https://github.com/sympy/sympy.git ...
  [git] error: RPC failed; curl 28 Failed to connect to github.com port 443 after 75001 ms: Couldn't connect to server
  [git] fatal: expected flush after ref listing
  ✗ Command failed: error: RPC failed; curl 28 Failed to connect to github.com port 443 after 75001 ms: Couldn't connect to server
fatal: expected flush after ref listing


============================================================
[8/10] django__django-14915
  Repo: django/django  Commit: 903aaa35
  Creating workspace ...
  [git] HEAD is now at 903aaa35e5 Fixed #33159 -- Reverted "Fixed #32970 -- Changed WhereNode.clone() to create a shallow copy of children."
  Running opencode (model=deepseek/deepseek-v4-flash, agent=dt-generation, timeout=Nones) ...
  Log → tmp/workspaces/20260612_161712/django__django-14915_opencode.log

  --- opencode finished (exit=0, elapsed=354s) ---
  Patch: 2028 chars
  ✓ Saved

============================================================
[9/10] sympy__sympy-22005
  Repo: sympy/sympy  Commit: 2c83657f
  Cloning https://github.com/sympy/sympy.git ...
  [git] error: RPC failed; curl 16 Error in the HTTP2 framing layer
  [git] fatal: expected flush after ref listing
  ✗ Command failed: error: RPC failed; curl 16 Error in the HTTP2 framing layer
fatal: expected flush after ref listing


============================================================
[10/10] pytest-dev__pytest-5221
  Repo: pytest-dev/pytest  Commit: 4a2fdce6
  Cloning https://github.com/pytest-dev/pytest.git ...
  Clone complete: repo-cache/pytest-dev__pytest
  Creating workspace ...
  [git] HEAD is now at 4a2fdce62 Emit a warning for record_property when used with xunit2 (#5204)
  Running opencode (model=deepseek/deepseek-v4-flash, agent=dt-generation, timeout=Nones) ...
  Log → tmp/workspaces/20260612_161712/pytest-dev__pytest-5221_opencode.log

  --- opencode finished (exit=0, elapsed=175s) ---
  Patch: 2383 chars
  ✓ Saved

============================================================
Run timestamp: 20260612_161712
Done. Predictions → predictions/opencode__deepseek_deepseek-v4-flash_20260612_161712.jsonl
Evaluate with:
  python -m src.main --dataset_name princeton-nlp/SWE-bench_Lite \
      --predictions_path predictions/opencode__deepseek_deepseek-v4-flash_20260612_161712.jsonl --filter_swt --run_id opencode_exp1
```

#### Evaluate an instance

Command:
```bash
DOCKER_HOST=unix:///$HOME/.docker/run/docker.sock \
python -m src.main \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path predictions/opencode__deep/iunseek_deepseek-v4-flash_20260612_161712.jsonl \
    --instance_ids scikit-learn__scikit-learn-10508 \
    --filter_swt \
    --max_workers 1 \
    --run_id opencode_pred-20260612_161712_eval-20260612-scikit-learn-10508
```

Output:
```bash
<frozen runpy>:128: RuntimeWarning: 'src.main' found in sys.modules after import of package 'src', but prior to execution of 'src.main'; this may result in unpredictable behaviour
Running 1 unevaluated instances...
Found 1 existing instance images. Will reuse them.
Running 1 instances...
  0%|                                                                                                                                                                             | 0/1 [00:00<?, ?it/s]Base image exec.base.x86_64:latest already exists, skipping build.
Env image exec.env.x86_64.558582a88927a0c8448ac6:latest already exists, skipping build.
Base image exec.base.x86_64:latest already exists, skipping build.
Env image exec.env.x86_64.558582a88927a0c8448ac6:latest already exists, skipping build.
Base image exec.base.x86_64:latest already exists, skipping build.
Env image exec.env.x86_64.558582a88927a0c8448ac6:latest already exists, skipping build.
Base image exec.base.x86_64:latest already exists, skipping build.
Env image exec.env.x86_64.558582a88927a0c8448ac6:latest already exists, skipping build.
Base image exec.base.x86_64:latest already exists, skipping build.
Env image exec.env.x86_64.558582a88927a0c8448ac6:latest already exists, skipping build.
Base image exec.base.x86_64:latest already exists, skipping build.
Env image exec.env.x86_64.558582a88927a0c8448ac6:latest already exists, skipping build.
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [02:16<00:00, 136.97s/it]
All instances run.
Cleaning cached images...
Removed 0 images.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 82.60it/s]
Total instances: 1
Instances completed: 0
Mean coverage: 1.0
Mean coverage delta: 1.0
Instances resolved: 1
Instances unresolved: 0
Instances with errors: 0
Instances still running: 0
Still existing images: 1
Report written to evaluation_results/opencode__deepseek_deepseek-v4-flash.opencode_pred-20260612_161712_eval-20260612-scikit-learn-10508.json
```
