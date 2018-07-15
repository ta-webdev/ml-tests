def accuracy_score(y_true, y_pred):
  size = len(y_true)
  
  if size != len(y_pred):
    raise ValueError('the lists have different length')
  if 0 == size:
    return 1
  
  true_predictions = 0

  for i in range(size):
    if y_true[i] == y_pred[i]:
      true_predictions += 1

  return true_predictions / size