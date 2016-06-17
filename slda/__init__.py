from pkg_resources import get_distribution, DistributionNotFound
from .topic_models import (LDA, SLDA, BLSLDA, GRTM)

try:
    __version__ = get_distribution('slda').version
except DistributionNotFound:
    pass
