# mave/dataset.py
"""
This class is a wrapper for all of data required to train or evaluate a model

the dataset_type field is to help standardize notation of different datasets:
       'A':'measured pre-retrofit data',
       'B':'pre-retrofit prediction with pre-retrofit model',
       'C':'pre-retrofit prediction with post-retrofit model',
       'D':'measured post-retrofit data',
       'E':'post-retrofit prediction with pre-retrofit model',
       'F':'post-retrofit prediction with pos-tretrofit model',
       'G':'TMY prediction with pre-retrofit model',
       'H':'TMY prediction with post-retrofit model'

typical comparisons used by mave:
    Pre-retrofit model performance = A vs B
    Single model M&V = D vs E
    Post retrofit model performance  = D vs F
    Dual model M&V, normalized to tmy data = G vs H

@author Paul Raftery <p.raftery@berkeley.edu>
"""
from sklearn import preprocessing
import numpy as np
import os

class Dataset(object):
    DESC ={
       'A':'measured pre-retrofit data',
       'B':'pre-retrofit prediction with pre-retrofit model',
       'C':'pre-retrofit prediction with post-retrofit model',
       'D':'measured post-retrofit data',
       'E':'post-retrofit prediction with pre-retrofit model',
       'F':'post-retrofit prediction with pos-tretrofit model',
       'G':'TMY prediction with pre-retrofit model',
       'H':'TMY prediction with post-retrofit model'}

    def __init__(self,
                 dataset_type=None,
                 base_dataset=None, 
                 X=None, 
                 X_s=None,
                 X_standardizer=None,
                 dts=None,
                 feature_names=None,
                 y=None,
                 y_s=None,
                 y_standardizer=None,
                 save=False):
        assert isinstance(dataset_type,str), \
               "dataset_type is not a char: %s"%dataset_type
        assert dataset_type in set(['A','B','C','D','E','F','G','H']), \
               "dataset_type is no a character from A to H: %s"%dataset_type
        self.dataset_type = dataset_type
        # if a base dataset is passed as an arg, use the relevant fields
        if base_dataset:
            X=base_dataset.X 
            X_s=base_dataset.X_s
            X_standardizer=base_dataset.X_standardizer
            y_standardizer=base_dataset.y_standardizer
            dts=base_dataset.dts 
            feature_names=base_dataset.feature_names
        # ensure standardizers are present
        assert isinstance(X_standardizer, preprocessing.StandardScaler), \
               "X_standardizer is not an instance " + \
               "of preprocessing.StandardScaler:%s"%type(X_standardizer)
        assert isinstance(y_standardizer, preprocessing.StandardScaler), \
               "y_standardizer is not an instance " + \
               "of preprocessing.StandardScaler:%s"%type(y_standardizer)
        self.X_standardizer = X_standardizer
        self.y_standardizer = y_standardizer
        # ensure both representations of X and y are present and same length
        self.X = X
        self.X_s = X_s
        self.y = y
        self.y_s = y_s
        if not isinstance(self.X_s, np.ndarray):
            self.X_s = self.X_standardizer.transform(self.X)
        if not isinstance(self.X, np.ndarray):
            self.X = self.X_standardizer.inverse_transform(self.X_s)
        if not isinstance(self.y_s, np.ndarray):
            self.y_s = self.y_standardizer.transform(self.y.reshape(-1, 1)).ravel()
        if not isinstance(self.y, np.ndarray):
            self.y = self.y_standardizer.inverse_transform(self.y_s.reshape(-1, 1)).ravel()
        assert self.X.shape[0] == len(self.y), \
               "length of X (%s) doesn't match y (%s)"%(self.X.shape[0],len(self.y))
        # ensure datetimes are the correct length
        assert len(dts) == len(self.y), \
               "length of dts (%s) doesn't match y (%s)"%(len(dts),len(self.y))
        self.dts = dts
        # ensure a set of feature names is present and of correct length
        assert isinstance(feature_names,list), \
               "feature_names is not a list object: %s"%type(feature_names)
        assert len(feature_names) == self.X.shape[1], \
               "different num of feature_names than features"
        self.feature_names = feature_names
        
        if save:
            str_date = [arr.strftime('%%Y-%%m-%%d%%T%%H%%M') for arr in self.dts]
            if not os.path.isdir('data'):
                os.mkdir('data')
            os.chdir(os.path.join(os.getcwd(),'data'))
            filename=str(self.DESC[self.dataset_type])+'.csv' 
            header= 'Datetime,' + ','.join(self.feature_names) + ',Data'
            data = np.column_stack((np.array(str_date), self.X, self.y,))
            np.savetxt(filename,
                       data,
                       delimiter=',',
                       header=header,
                       fmt='%s',
                       comments='')
            os.chdir('..')

    def __str__(self):
        return 'Dataset type: %s, %s'\
                %(self.dataset_type,self.DESC[self.dataset_type])

if __name__=='__main__':
   import numpy as np
   from datetime import datetime

   X = np.random.rand(24,3)
   y = np.random.rand(24,)
   X_standardizer = preprocessing.StandardScaler().fit(X)
   y_standardizer = preprocessing.StandardScaler().fit(y.reshape(-1, 1))
   dts = np.arange('2014-01-01T00:00','2014-01-02T00:00',\
                     dtype=('datetime64[h]')).astype(datetime)
   feature_names = ['Minute','Hour','DayOfWeek']
   test = Dataset(dataset_type='A',
                  X_s=X,
                  X_standardizer=X_standardizer,
                  y_s=y,
                  y_standardizer=y_standardizer,
                  dts=dts,
                  feature_names=feature_names,
                  save=False)
   print(test)
   print(test.X)
   print(test.X_s)
   print(test.y)
   print(test.y_s)

# ======================================================================

# mave/estimators.py
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.neighbors import KernelDensity

def normalize(x):
    # normalize numpy array to [0, 1]
    mi = np.min(x) 
    x -= np.sign(mi) * np.abs(mi)
    x /= np.max(x)
    return x

class HourWeekdayBinModel(DummyRegressor):

    def __init__(self, strategy='mean'):
        self.strategy = strategy

    def fit(self, X, y):
        a = np.zeros((24, 7))
        hours = np.copy(X[:, 1])
        weekdays = np.copy(X[:, 2])
        hours = 23 * normalize(hours)
        weekdays = 6 * normalize(weekdays)

        if self.strategy == 'mean':
            counts = a.copy()
            for i, row in enumerate(zip(hours, weekdays)):
                hour = int(row[0])
                day = int(row[1])
                counts[hour, day] += 1
                a[hour, day] += y[i]

            counts[counts == 0] = 1
            self._model = a / counts

        elif self.strategy in ('median', 'kernel'):

            # this is a 3d array 
            groups = [[[] for i in range(7)] for j in range(24)]

            for i, row in enumerate(zip(hours, weekdays)):
                hour = int(row[0])
                day = int(row[1])
                groups[hour][day].append(y[i])

            if self.strategy == 'median':
                for i, j in np.ndindex((24, 7)):
                    if groups[i][j]:
                        a[i,j] = np.median(groups[i][j])
                    else:
                        a[i,j] = np.nan
            elif self.strategy == 'kernel':
                # kernel method computes a kernel density for each of the
                # bins and determines the most probably value ('mode' of sorts)
                grid = np.linspace(np.min(y), np.max(y), 1000)[:, np.newaxis]
                for i, j in np.ndindex((24, 7)):
                    if groups[i][j]:
                        npgroups = np.array(groups[i][j])[:, np.newaxis]
                        kernel = KernelDensity(kernel='gaussian', \
                                                bandwidth=0.2).fit(npgroups)
                        density = kernel.score_samples(grid)
                        dmax = np.max(density)
                        imax = np.where(density==dmax)
                        a[i,j] = grid[imax, 0]
                    else:
                        a[i,j] = np.nan

            self._model = a

        # smooth the model here if there are nans
        return self

    def predict(self, X):
        hours = np.copy(X[:, 1])
        weekdays = np.copy(X[:, 2])
        hours = 23 * normalize(hours)
        weekdays = 6 * normalize(weekdays)
        prediction = [self._model[int(x[0]), int(x[1])] for x in zip(hours, weekdays)]
        return np.array(prediction)
