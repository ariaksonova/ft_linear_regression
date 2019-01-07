def functional(wt0, wt1, data):
    sum = 0
    errors = 0
    length = len(data)
    for i in range(length):
        sum += wt1 * data[0][i] + wt0 - data[1][i]
    errors = sum / length
    return errors


def train_model(predict_data):
    pass