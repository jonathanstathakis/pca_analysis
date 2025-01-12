from pca_analysis.cabernet import AbstChrom
from pca_analysis.cabernet.shiraz.shiraz import Shiraz


class Decomposition(AbstChrom):
    def __init__(self, shz: Shiraz):
        self._shz = shz

    def pca(self):
        """
        perform PCA on the input dataarray. Thus particular implementation requires that the input array is 2D thus
        it unfolds along the sample domain.
        """

        if self._shz.dims != 2:
            raise ValueError(
                "expect a 2D dataarray. If dims greater than 2, unfold first with `transform.unfold`"
            )
