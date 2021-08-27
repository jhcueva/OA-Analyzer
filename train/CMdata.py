def confusion_matrixx_data(preds, target):
    knee_pred = []
    knee_true = []
    prediction = []
    label = []
    value_preds = preds.cpu().numpy()
    [prediction.append(value_preds[i]) for i in range(len(value_preds))]
    value_target = target.cpu().numpy()
    [label.append(value_target[i]) for i in range(len(value_target))]
    knee_pred.extend(prediction)
    knee_true.extend(label)
    return knee_pred, knee_true
