__all__ = ["PWL", "MonoPWL"]
__version__ = "0.1.0"

from .pwl import PointPWL, MonoPointPWL, SlopedPWL, MonoSlopedPWL

PWL = SlopedPWL
MonoPWL = MonoSlopedPWL