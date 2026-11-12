#!/usr/bin/env python3
"""CI-portable Phase XIX-X attention negative controls.

The main suite is retained unchanged. CI does not build a release trainer
binary, so the changed-binary control cannot compare a digest there. This
variant keeps all eight controls and exercises the receipt's trainer-binding
format guard when the binary is unavailable; when the binary exists it uses the
same changed-hash assertion as the main suite.
"""

from __future__ import annotations

import unittest

import test_attention_phase19x as base
from test_attention_phase19x import CRATE


@unittest.skipUnless(base.RECEIPT.is_file(), "attention receipt is not present")
class AttentionPhaseXIXXCIControls(base.AttentionPhaseXIXXControls):
    def test_changed_trainer(self):
        trainer = CRATE / "target/release/hierarchos-vulkan-transformer-train.exe"
        if trainer.is_file():
            self.document["binaries"]["trainer"] = "0" * 64
            self.expect("trainer changed")
        else:
            self.document["binaries"]["trainer"] = "not-a-sha256"
            self.expect("missing attention trainer binding")


if __name__ == "__main__":
    unittest.main()
