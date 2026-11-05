# GPT-2 kernel-selection sweep (superseded)

`selection-main.log` and `selection-extra.log` are the intermediate sweep that
made each kernel selectable through a temporary environment override. It
established that individual kernels move the wide cases by a few ULP in both
directions, so no single kernel is an accuracy win.

The decision was then made on the full gates rather than this scratch table.
See
[`../20261102-gen9-gpt2-promotion/`](../20261102-gen9-gpt2-promotion/README.md):
all three kernels regress the 166-run text gate (164/166), while Welford plus
the attention reciprocal keeps it at 166/166 and is what ships.
