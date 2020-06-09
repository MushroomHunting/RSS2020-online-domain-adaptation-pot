import numpy as np
import csv
import sys
import scipy as sp
from sklearn import metrics

def read_txy_csv(fn):
   data = readCsvFile(fn)
   Xtest = data[:, :3]
   Ytest = data[:, 3][:, np.newaxis]
   return Xtest, Ytest


def readCsvFile( fileName ):
    reader = csv.reader(open(fileName,'r') )
    dataList = []
    for row in reader:
        dataList.append( [float(elem) for elem in row ] )
    return np.array( dataList )

def entropy(x):
    x[x < sys.max_info.min] = sys.max_info.min
    return -1*np.sum(x*np.log2(x))

def cross_entropy(act, pred):
    #negative log-loss sklearn
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred))
    return -ll

def neg_ms_log_loss(true_labels, predicted_mean, predicted_var):
    """
    :param true_labels:
    :param predicted_mean:
    :param predicted_var:
    :return: Neg mean squared log loss (neg the better)
    """

    predicted_var += np.finfo(float).eps #to avoid /0 and log(0)
    smse = 0.5*np.log(2*np.pi*predicted_var) + ((predicted_mean - predicted_var)**2)/(2*predicted_var)

def calc_scores(mdl_name, true, predicted, predicted_var=None, time_taken=-11):
   fn = 'output/reports/'+ mdl_name+ '.csv'

   predicted_binarized = np.int_(predicted >= 0.5)
   accuracy = np.round(metrics.accuracy_score(true.ravel(), predicted_binarized.ravel()), 3)

   auc = np.round(metrics.roc_auc_score(true.ravel(), predicted.ravel()), 3)

   nll = np.round(metrics.log_loss(true.ravel(), predicted.ravel()), 3)

   if predicted_var is not None:
      neg_smse = np.round(neg_ms_log_loss(true, predicted[0].ravel(), predicted_var.ravel()), 3)
   else:
      neg_smse = -11

   print(mdl_name+': accuracy={}, auc={}, nll={}, smse={}, time_taken={}'.format(accuracy, auc, nll, neg_smse, time_taken))
   #print(metrics.confusion_matrix(true.ravel(), predicted_binarized.ravel()))

   with open(fn,'ab') as f_handle: #try 'a'
      #np.savetxt(f_handle, np.array([[neg_smse]]), delimiter=',', fmt="%.3f")
      np.savetxt(f_handle, np.array([[accuracy, auc, nll, neg_smse, time_taken]]), delimiter=',', fmt="%.3f")

def calc_scores(mdl_name, true, predicted, predicted_var=None, time_taken=-11, N_points=0, do_return = False, save_report=True):
   fn = 'output/reports/'+ mdl_name+ '.csv'

   predicted_binarized = np.int_(predicted >= 0.5)
   accuracy = np.round(metrics.accuracy_score(true.ravel(), predicted_binarized.ravel()), 3)

   auc = np.round(metrics.roc_auc_score(true.ravel(), predicted.ravel()), 3)

   nll = np.round(metrics.log_loss(true.ravel(), predicted.ravel()), 3)

   if predicted_var is not None:
      neg_smse = np.round(neg_ms_log_loss(true, predicted[0].ravel(), predicted_var.ravel()), 3)
   else:
      neg_smse = -11

   print(mdl_name+': accuracy={}, auc={}, nll={}, smse={}, time_taken={}'.format(accuracy, auc, nll, neg_smse, time_taken))
   #print(metrics.confusion_matrix(true.ravel(), predicted_binarized.ravel()))
   if save_report is True:
       with open(fn,'ab') as f_handle: #try 'a'
          #np.savetxt(f_handle, np.array([[neg_smse]]), delimiter=',', fmt="%.3f")
          np.savetxt(f_handle, np.array([[accuracy, auc, nll, neg_smse, time_taken, N_points]]), delimiter=',', fmt="%.3f")
   if do_return:
       return accuracy, auc, nll
