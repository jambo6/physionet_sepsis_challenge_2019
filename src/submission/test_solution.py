from definitions import *
from src.models.evaluators import ComputeNormalizedUtility



if __name__ == '__main__':
    # Get the solution pipeline
    solution_location = ROOT_DIR + '/models/solutions/solution_2.pickle'
    pipeline = load_pickle(solution_location)

    # Hello
    df = load_pickle(DATA_DIR + '/interim/from_raw/df.pickle')
    labels = load_pickle(DATA_DIR + '/processed/labels/original.pickle')

    predictions = pipeline.predict(df)
    score = ComputeNormalizedUtility().score(labels, predictions)