TODO
----
* Investigate the tempogram-based model described on p. 184 (also implemented in librosa). This one's nice because it's linear (based on a wavelet transform), so can use a Kalman Filter for tracking the statistical parameters.
* Try fixing groupdelayfeature by looking for linear portions of the data and just finding the slope of those.
