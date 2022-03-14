from math import sqrt


def get_tp_tn_fp_fn(output_data, target_data):
    """ A classification metric

    Args:
        output_data(array of dimension (x,2)): considered negative if left value > right value
        target_data(array of dimension (x,1)): considered negative if 0, positive otherewise

    Returns (four floats):
        true positive count (tp)
        true negative count (tn)
        false positive count (fp)
        false negative count (fn)
    """

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    total = output_data.shape[0]
    if total == 0:
        raise ValueError("0 output data entries")

    for index in range(total):
        output0, output1 = output_data[index,:]
        target = target_data[index]

        if output0 > output1:  # negative output

            if target != 0:  # wrong

                fn += 1

            else:  # right

                tn += 1

        else:  # positive output

            if target != 0:  # right

                tp += 1

            else:  # wrong

                fp += 1

    return tp, tn, fp, fn


def get_mcc(tp, tn, fp, fn):
    """ The Mathews Correlation Coefficient

    Args:
        tp (float): true positive count
        tn (float): true negative count
        fp (float): false positive count
        fn (float): false negative count

    Returns (float): Mathews Correlation Coefficient
    """

    numerator = tp * tn - fp * fn
    if numerator == 0:
        return 0.0

    denominator = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        raise ValueError(f"MCC denominator is zero for tp={tp}, tn={tn}, fp={fp}, fn={fn}")

    return numerator / denominator
