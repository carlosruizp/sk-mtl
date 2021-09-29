import pytest

from sklearn.utils.estimator_checks import check_estimator

from skmtl import MTLSVC


@pytest.mark.parametrize(
    "estimator",
    [MTLSVC()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
