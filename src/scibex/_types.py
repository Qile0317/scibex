from __future__ import annotations

from typing import Literal

Backend = Literal["r", "python"]
Chain = Literal["Heavy", "Light"]
Method = Literal["encoder", "geometric"]
EncoderModel = Literal["CNN", "VAE", "CNN.EXP", "VAE.EXP"]
EncoderInput = Literal["atchleyFactors", "crucianiProperties", "kideraFactors", "MSWHIM", "tScales", "OHE"]
Species = Literal["Human", "Mouse"]
Strategy = Literal["strict", "lenient"]
