# Target Speaker Extraction Guided by EEG Cues via Structured State Space Models

In this work, we revisit the NeuroHeed framework by replacing its original DPRNN-based separator with a Structured State Space Model (S4M).  This substitution aims to explore whether structured state space models can better capture the alignment between neural and acoustic features, especially over short windows ranging from 1 to 10 seconds — a key constraint in practical neuro-steered systems. Our architecture retains NeuroHeed’s encoder-decoder structure but introduces S4M as a drop-in separator module, potentially enhancing attention decoding robustness and temporal generalization.

