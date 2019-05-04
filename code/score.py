from argparse import ArgumentParser

ALL_TAGS = {"B", "I", "E", "S"}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("prediction_file", help="The path to the prediction file (in BIES format)")
    parser.add_argument("gold_file", help="The path to the gold file (in BIES format)")

    return parser.parse_args()


def is_valid_prediction(prediction_iter, gold_iter):
    assert len(prediction_iter) == len(gold_iter), "Prediction and gold have different lengths"

    prediction_tags = set()
    gold_tags = set()
    nr_line = 1
    for preds, gold in zip(prediction_iter, gold_iter):
        assert len(preds) == len(gold), "Line " + str(nr_line) + ": lengths mismatch"
        prediction_tags.update(preds)
        gold_tags.update(gold)
        nr_line += 1
    
    prediction_tags = {t.upper() for t in prediction_tags}
    gold_tags = {t.upper() for t in gold_tags}
    
    assert len(gold_tags.difference(ALL_TAGS)) == 0, "Unknown tag detected in gold"
    assert len(prediction_tags.difference(ALL_TAGS)) == 0, "Unknown tag detected in predictions"


def score(prediction_iter, gold_iter, verbose=False):
    """
    Returns the precision of the model's predictions w.r.t. the gold standard (i.e. the tags of the
    correct word segmentation).

    :param prediction_iter: List of strings in the BIES format representing the model's predictions.
    :param gold_iter: List of strings in the BIES format representing the gold standard.

    :return: precision [0.0, 1.0]
    
    Ex. predictions_iter = ["BEBESBIIE",
                            "BIIIEBEBESS"]
        gold_iter = ["BEBIEBIES",
                     "BIIESBEBESS"]
        output: 0.7
    
    The same result can be obtain by passing list of lists
    Ex. predictions_iter = [["B", "E", "B", "E", "S", "B", "I", "I", "E"],
                            ["B", "I", "I", "I", "E", "B", "E", "B", "E", "S", "S"]]
        gold_iter = [["B", "E", "B", "I", "E", "B", "I", "E", "S"],
                     ["B", "I", "I", "E", "S", "B", "E", "B", "E", "S", "S"]]
        output: 0.7

    
    """
    
    is_valid_prediction(prediction_iter, gold_iter)

    right_predictions = 0
    wrong_predictions = 0

    for prediction_sentence, gold_sentence in zip(prediction_iter, gold_iter):
        for prediction_tag, gold_tag in zip(prediction_sentence, gold_sentence):
            if prediction_tag == gold_tag:
                right_predictions += 1
            else:
                wrong_predictions += 1

    precision = right_predictions / (right_predictions + wrong_predictions)
    if verbose:
        print("Precision:\t", precision)

    return precision


def label_text_to_iter(file_path):
    iter_ = []
    with open(file_path) as f:
        for line in f:
            line = line.strip().upper()
            iter_.append(line)
    return iter_


if __name__ == '__main__':
    args = parse_args()
    prediction_iter = []
    score(label_text_to_iter(args.prediction_file), label_text_to_iter(args.gold_file), verbose=True)

