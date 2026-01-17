

import numpy as np
import pandas as pd


theta = np.linspace(0.0, np.pi, 361)[1:]
rutherford = np.sin(theta/2)**-4

rutherford_cdf = np.cumsum(rutherford) / np.sum(rutherford) 

angles = theta * 180/ np.pi


pd.DataFrame(dict(prob=rutherford_cdf, angles=angles)).to_csv("rutherford.csv", index=False)