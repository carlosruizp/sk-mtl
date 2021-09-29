import pytest

from sklearn.utils.estimator_checks import check_estimator

from skmtl import TemplateEstimator
from skmtl import TemplateClassifier
from skmtl import TemplateTransformer


@pytest.mark.parametrize(
    "estimator",
    [TemplateEstimator(), TemplateTransformer(), TemplateClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
