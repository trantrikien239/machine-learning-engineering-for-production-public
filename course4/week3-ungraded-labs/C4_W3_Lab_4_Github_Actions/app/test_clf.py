import pickle
from main import clf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def test_accuracy():

    # Load test data
    with open("data/test_data.pkl", "rb") as file:
        test_data = pickle.load(file)

    # Unpack the tuple
    X_test, y_test = test_data

    # Compute accuracy of classifier
    acc = clf.score(X_test, y_test)

    # Accuracy should be over 90%
    assert acc > 0.9

def test_pipeline_and_scaler():
    isPipeline = isinstance(clf, Pipeline)
    assert isPipeline

    if isPipeline:
        first_step = [v for v in clf.named_steps.values()][0]
        assert isinstance(first_step, StandardScaler)