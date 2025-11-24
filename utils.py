import numpy as np

def ema(prev, new, alpha=0.2):
    if prev is None:
        return new
    return (1-alpha)*prev + alpha*new