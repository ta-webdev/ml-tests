import math

def mean(arr):
  res = 0
  for x in arr:
    res += x

  res /= len(arr)

  return res


def ss_arr(arr1, arr2):
  res = 0

  for i in range(0, len(arr1)):
    res += (arr1[i] - arr2[i])**2

  return res


def ss_val(arr, val):
  res = 0

  for x in arr:
    res += (val - x)**2

  return res

def variance(arr, sample = False):
  m = mean(arr)
  
  ss = ss_val(arr, m)
  if sample:
    return ss / (len(arr) - 1)
  else:
    return ss / len(arr)
   
def std_dev(arr, sample = False):
  return math.sqrt(variance(arr, sample))

def ma_dev(arr):
  m = mean(arr)

  s = 0
  for val in arr:
    s += abs(val - m)

  return s / len(arr) 




def r_squared(true_vals, pred_vals):
  true_vals_mean = mean(true_vals)
  pred_vals_mean = mean(pred_vals)
  ss_true_vals = ss_val(true_vals, true_vals_mean)
  ss_pred_vals = ss_val(pred_vals, true_vals_mean)

  # print('true_vals_mean', true_vals_mean)
  # print('ss_pred_vals', ss_pred_vals)
  # print('ss_true_vals', ss_true_vals)
  # return (ss_pred_vals / ss_true_vals)

  return 1 - (ss_arr(true_vals, pred_vals) / ss_val(true_vals, true_vals_mean))

def lin_reg(arr):
  mean_x = mean([x for x, y in arr])
  mean_y = mean([y for x, y in arr])
  
  mean__x_times_y = mean([x*y for x, y in arr])
  mean__x_squared = mean([x ** 2 for x, y in arr])
  
  slope = (mean_x * mean_y - mean__x_times_y) / (mean_x ** 2 - mean__x_squared)
  intercept = mean_y - slope * mean_x

  return (slope, intercept)

def lin_reg_r(arr):
  mean_x = mean([x for x, y in arr])
  mean_y = mean([y for x, y in arr])

  std_dev_x = std_dev([x for x, y in arr], True)  
  std_dev_y = std_dev([y for x, y in arr], True)  
  
  r = sum([(x - mean_x) / std_dev_x * (y - mean_y) / std_dev_y for x, y in arr]) / (len(arr) - 1)

  slope = r * std_dev_y / std_dev_x
  intercept = mean_y - slope * mean_x

  return (slope, intercept, r, r**2)

def lin_reg_gd(arr, alpha = 0.001, epochs = 10000):
  slope = 0
  intercept = 0

  grad_slope = 1
  grad_intercept = 1
  count = 0
  while grad_slope > 0.000000001 or grad_intercept > 0.000000001:
    grad_slope = sum([x*(y - slope * x - intercept)*alpha for x, y in arr])
    grad_intercept = sum([(y - slope * x - intercept)*alpha for x, y in arr])

    slope = slope + grad_slope
    intercept = intercept + grad_intercept
    count += 1
  print('count', count)
  return (slope, intercept)

def lin_reg_gd_multi(X, y, alpha = 0.001, epochs = 10000):
  scale_X = feature_scaling(X)
  print('scale_X', scale_X)
  
  def compute_y_hat(data, params):
    res = 0
    for i in range(len(data)):
      res += (data[i] * params[i])
      
    res += params[-1]
    #print('data', data, 'params', params, 'res', res)

    return res

  num_features = X.shape[1]
  params = [0] * (num_features + 1)
  gradients = [1] * (num_features + 1)
  
  count = 0
  while count < 1000:#max([abs(x) for x in gradients]) > 0.0000001:
    for i in range(num_features):
      gradients[i] = alpha * sum([row[i] * (compute_y_hat(row, params) - a_y) for row, a_y in zip(X, y)])
      
    gradients[-1] = alpha * sum([compute_y_hat(row, params) - a_y for row, a_y in zip(X, y)])

    params = [p - g for p, g in zip(params, gradients)]
    
    # print(gradients, params)
    # print("\n")
    count += 1
  print('count', count, gradients)
  return params

def feature_scaling(X):
  rows = X.shape[0]
  cols = X.shape[1]
  
  mins = [math.inf] * cols
  maxes = [-math.inf] * cols
  means = [0] * cols

  for row in X:
    for i, val in enumerate(row):
      if val < mins[i]:
        mins[i] = val
      if val > maxes[i]:
        maxes[i] = val

      means[i] += val

  means = [val / rows  for val in means]

  for row_idx in range(rows):
    for col_idx in range(cols):
      X[row_idx][col_idx] = (X[row_idx][col_idx] - means[col_idx]) / (maxes[col_idx] - mins[col_idx])

  return (means, [max - min for max, min in zip(maxes, mins)])