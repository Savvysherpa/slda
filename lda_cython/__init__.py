from pkg_resources import get_distribution
from .topic_models import (LDA, SLDA, BLSLDA, GRTM)

__version__ = get_distribution('lda-cython').version
