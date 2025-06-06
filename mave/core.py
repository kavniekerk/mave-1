# mave/core.py
"""
Building Energy Prediction

This software reads an input file (a required argument) containing
building energy data in a format similar to example file.
It then trains a model and estimates the error associated
with predictions using the model.

@author Paul Raftery <p.raftery@berkeley.edu>
@author Tyler Hoyt <thoyt@berkeley.edu>
"""

import csv
import os
import shutil
import pickle
import dateutil.parser
import numpy as np
import pandas as pd
import pprint
from math import sqrt
from datetime import datetime, timedelta
from sklearn import preprocessing, model_selection, metrics
import mave.holidays
import mave.trainers
import mave.comparer
import mave.dataset
import mave.visualize
import logging
log = logging.getLogger("mave.core")

class Preprocessor(object):

    IGNORE_TAG = -1
    PRE_DATA_TAG = 0
    POST_DATA_TAG = 1

    def __init__(self,
                 input_file,
                 use_holidays=True,
                 use_month=False,
                 use_tmy=False,
                 start_frac=0.0,
                 end_frac=1.0,
                 changepoints=None,
                 test_size=0.25,
                 datetime_column_name='LocalDateTime',
                 holiday_keys=['USFederal'],
                 dayfirst=False,
                 locale=None,
                 outside_db_column_name = 'OutsideDryBulbTemperature',
                 outside_dp_column_name = 'OutsideDewPointTemperature',
                 target_column_name = 'EnergyConsumption',
                 remove_outliers = 'SingleValue',
                 remove_zeros = True,
                 X_standardizer = None,
                 previous_data_points = 2,
                 column_names=['LocalDateTime','EnergyConsumption'],
                 n_headers=3,
                 **kwargs):
        log.info("Preprocessing started")
        self.datetime_column_name = datetime_column_name
        self.use_holidays = use_holidays
        self.use_month = use_month
        self.use_tmy = use_tmy
        self.previous_data_points = previous_data_points
        self.X_standardizer = X_standardizer
        self.outside_dp_column_name = outside_dp_column_name
        self.feature_names = ['Minute','Hour','DayOfWeek']
        if use_month:
            self.feature_names.append('Month')
        
        # identify holidays to use (if any)
        self.holidays = set([])
        if use_holidays:
            self.feature_names.append('Holiday')
            for key in holiday_keys:
                if key in mave.holidays.holidays:
                    self.holidays = self.holidays.union(mave.holidays.holidays[key])
                else:
                    log.warn(f"Holiday key '{key}' not recognized. Skipping.")
        
        # read in the input data
        data = pd.read_csv(input_file,
                           skiprows = n_headers,
                           usecols = column_names,
                           na_values = '?',
                           skip_blank_lines = True,
                           parse_dates = [self.datetime_column_name],
                           infer_datetime_format = True,
                           dayfirst=dayfirst,
                           skipinitialspace = True)
        # drop columns with more than 50% nans
        data = data.dropna(axis = 'columns', thresh = int(0.5 * len(data)))
        data = data.to_records(index = False)
        # shrink the input data by start_frac and end_frac
        data_L = len(data)
        start_index = int(start_frac * data_L)
        end_index = int(end_frac * data_L)
        data = data[ start_index : end_index ]
        # convert data types
        dts = data[datetime_column_name]
        dtypes = data.dtype.descr
        columns_names = list(data.dtype.names)
        for i in range(len(dtypes)):
            if dtypes[i][0] != datetime_column_name:
                if dtypes[i][1] == '|O':
                    columns_names.remove(dtypes[i][0])
                dtypes[i] = dtypes[i][0], 'f8' # parse all other data as float
        dtypes = [dtype for dtype in dtypes if dtype[0] in columns_names]
        data = data[columns_names]
        data = data.astype(dtypes)
        log.info("Creating input features from datetimes")
        data, dts, self.interval_seconds, self.vals_per_hr = \
            self.standardize_datetimes(data, dts)
        vectorized_process_datetime = np.vectorize(self.process_datetime)
        d = np.column_stack(vectorized_process_datetime(dts))

        # Skip weather data download for now - would need proper location module
        if locale and not outside_db_column_name in data.dtype.names:
            log.warn("Location-based weather download not implemented. Skipping TMY features.")
            self.use_tmy = False
        
        log.info("Creating other (non datetime related) input features")
        data, target_col_ind = self.append_input_features(
            data,
            d,
            outside_db_column_name,
            target_column_name,
            datetime_column_name)
        log.info("Cleaning up data - removing outliers, missing data, etc.")
        self.X, self.y, self.dts = \
            self.clean_data(data, 
                            dts, 
                            target_col_ind, 
                            remove_zeros,
                            remove_outliers)
        # ensure that the datetimes match the input features
        if len(self.dts) > 0 and len(self.X) > 0:
            if (self.X[:,0] != np.array([dt.minute for dt in self.dts])).any() \
                or (self.X[:,1] != np.array([dt.hour for dt in self.dts])).any():
                raise Exception(("The datetimes in the datetimes array do not"
                                 " match those in the input features array"))
        self.cps = self.changepoint_feature(changepoints=changepoints,
                                            **kwargs)
        log.info("Splitting data into pre- and post-retrofit datasets")
        self.split_dataset(test_size=test_size)

    def clean_data(self,
                   data,
                   datetimes,
                   target_col_ind,
                   remove_zeros=True,
                   remove_outliers='SingleValue'):
        # remove any row with missing data, identified by nan
        keep_inds = ~np.isnan(data).any(axis=1)
        num_to_del = len(keep_inds[~keep_inds])
        if num_to_del > 0:
            datetimes = datetimes[keep_inds]
            data = data[keep_inds]
        # split the data into input and target arrays
        if target_col_ind >= data.shape[1]:
            y = None
        else:
            y = data[:,target_col_ind]
        X = np.hstack((data[:,:target_col_ind], data[:,target_col_ind+1:]))
        # remove outliers
        if y is not None:
            if remove_outliers == 'SingleValue':
                keep_inds = self.is_single_value_outlier(y,med_diff_multiple=10)
            elif remove_outliers == 'MultipleValues':
                keep_inds = self.is_outlier(y, threshold=10)
            else:
                keep_inds = np.ones(len(y),dtype=bool)
            if remove_zeros:
                keep_inds *= y != 0.0
            # log outlier datetimes and values
            outliers = y[~keep_inds]
            outlier_ts =  [str(l) for l in datetimes[~keep_inds]]
            if len(outliers) > 0:
                log.warn(("Removed the following %s outlier value(s): "
                          "\n%s"
                          %(len(outliers),
                            pprint.pformat(list(zip(outlier_ts,outliers))))))
                X = X[keep_inds]
                y = y[keep_inds]
                datetimes = datetimes[keep_inds]
        return X, y, datetimes

    def append_input_features(self, data, d0, outside_db_column_name,
                              target_column_name, datetime_column_name):
        column_names = [name for name in data.dtype.names \
            if name != datetime_column_name]
        d = d0
        for s in column_names:
            if s == outside_db_column_name:
                if np.nanmedian(data[s]) > 32.0:
                    # almost certainly in crazy ancient units [F]
                    # unless the building is in Antarctica
                    log.warn(("The median outside drybulb temperature "
                        " is high (> 32) - assumed that it is in degF and "
                        " converted to degC to match units of weather and "
                        " TMY data."))
                    t = 5*(data[s]-32.0)/9
                else:
                    t = data[s]
                d = np.column_stack( (d, t) )
                self.feature_names.append(str(s))
                if self.previous_data_points > 0:
                    # create input features using historical data
                    # at the intervals defined by n_vals_in_past_day
                    for v in range(1, self.previous_data_points + 1):
                        past_hours = v * 24 / (self.previous_data_points + 1)
                        n_vals = int(past_hours * self.vals_per_hr)
                        past_data = np.roll(t, n_vals)
                        # for the first day in the file
                        # there will be no historical data
                        # use the data from the next day as a rough estimate
                        vals_per_day = int(24 * self.vals_per_hr)
                        past_data[0:n_vals] = past_data[vals_per_day: vals_per_day+n_vals]
                        d = np.column_stack( (d, past_data) )
                        self.feature_names.append(str(s)+'_-'+ str(past_hours))
            elif not s == target_column_name:
                # just add the column as an input feature
                # without historical data
                self.feature_names.append(str(s))
                d = np.column_stack( (d, data[s]) )
        # add the target data
        split = d.shape[1]
        if target_column_name in column_names:
            d = np.column_stack( (d, data[target_column_name]) )
        return d, split

    def is_single_value_outlier(self, y, med_diff_multiple=10):
        # id 2 highest and lowest values (ignoring nans)
        # id a single value as an outlier if the min or max is very far
        # (> 100 times the median difference between values)
        # from the next nearest unique value
        keep_inds = np.ones(len(y), dtype=bool)
        mx = np.amax(y)
        mn = np.amin(y)
        median_diff = np.median(abs(np.diff(y)))
        y_unique = np.unique(y)
        if len(y_unique) >= 2:
            diff_to_max = np.diff(y_unique[np.argpartition(y_unique, -2)][-2:])[0]
            if abs(diff_to_max) > med_diff_multiple*median_diff:
                keep_inds = y < mx
            diff_to_min = np.diff(y_unique[np.argpartition(y_unique, 2)][:2])[0]
            if abs(diff_to_min) > med_diff_multiple*median_diff:
                keep_inds = y > mn
        return keep_inds

    def is_outlier(self, y, threshold=10):
        # outliers detected based on median absolute deviation according to
        # Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        # Handle Outliers", The ASQC Basic References in Quality Control:
        # Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
        if len(y.shape) == 1:
            y = y[:,None]
        median = np.median(y, axis=0)
        diff = np.sum((y - median)**2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)
        if med_abs_deviation > 0:
            modified_z_score = 0.6745 * diff / med_abs_deviation
            keep_inds = modified_z_score <= threshold
        else:
            keep_inds = np.ones(len(y), dtype=bool)
        return keep_inds

    def standardize_datetimes(self, data, dts):
        # calculate the interval between datetimes
        dts = [datetime.utcfromtimestamp(dt.astype('uint64') / 1e9) for dt in dts]
        intervals = [int((dts[i]-dts[i-1]).total_seconds()) for i in range(1, len(dts))]
        median_interval = int(np.median(intervals))
        vals_per_hr = 3600 / median_interval
        assert (3600 % median_interval) == 0,  \
            'Median interval between datetimes must divide evenly into an hour'
        median_interval_minutes = median_interval/60
        assert (median_interval % 60) == 0,  \
            'Median interval between datetimes must be an even num of minutes'
        # round time datetimes according to the median_interval
        vectorized_round_datetime = np.vectorize(self.round_datetime)
        dts = vectorized_round_datetime(dts, median_interval_minutes)
        # remove duplicates and sorts datetimes
        dts, inds = np.unique(dts, return_index = True)
        data = data[inds]
        # updates intervals after datetime rounding and duplicate removal
        intervals = [int((dts[i]-dts[i-1]).total_seconds()) for i in range(1, len(dts))]
        row_length = len(data[0])
        # add datetimes and nans when there are gaps (based on median interval)
        gaps = np.greater(intervals, median_interval)
        gap_inds = np.nonzero(gaps)[0] # contains the left indices of the gaps
        NN = 0 # accumulate offset of gap indices as entries are added
        missing_intervals = []
        for i in gap_inds:
            gap = intervals[i]
            gap_start = dts[i + NN]
            gap_end = dts[i + NN + 1]
            missing_intervals.append(('from ' + str(gap_start),
                                      'to ' + str(gap_end)))
            N = gap / median_interval - 1 # number of entries to add
            for j in range(1, int(N)+1):
                new_dt = gap_start + j*timedelta(seconds=median_interval)
                new_row = np.array([(new_dt,) + (np.nan,) * (row_length - 1)],
                                                         dtype=data.dtype)
                data = np.append(data, new_row)
                dts = np.append(dts, new_dt)
                dts_ind = np.argsort(dts)
                data = data[dts_ind]
            dts = dts[dts_ind] # sorts datetimes
            NN += int(N)
        if len(missing_intervals) > 0:
            log.info(("Missing datetime interval(s) in input file:\n%s"
                      %pprint.pformat(missing_intervals)))
        return data, dts, median_interval, vals_per_hr

    def round_datetime(self, dt, interval):
        # rounds a datetime to a given minute interval
        discard = timedelta(minutes=dt.minute % interval,
                            seconds=dt.second,
                            microseconds = dt.microsecond)
        dt -= discard
        if discard >= timedelta(minutes=interval/2):
            dt += timedelta(minutes=interval)
        return dt

    def process_datetime(self, dt):
        # takes a datetime and returns a tuple of:
        # minute, hour, weekday, (month), and (holiday)
        rv = float(dt.minute), float(dt.hour), float(dt.weekday())
        if self.use_month:
            rv += float(dt.month),
        if self.holidays:
            if dt.date() in self.holidays:
                hol = 3.0 # this day is a holiday
            elif (dt + timedelta(1,0)).date() in self.holidays:
                hol = 2.0 # next day is a holiday
            elif (dt - timedelta(1,0)).date() in self.holidays:
                hol = 1.0 # previous day was a holiday
            else:
                hol = 0.0 # this day is not near a holiday
            rv += hol,
        return rv

    def changepoint_feature(self,
                            changepoints = None,
                            dayfirst = False,
                            timestamp_format = '%Y-%m-%dT%H%M',
                            **kwargs):
        if changepoints is not None:
            # convert timestamps to datetimes
            cps = []
            for timestamp,tag in changepoints:
                try:
                    cp_dt = datetime.strptime(timestamp, timestamp_format)
                except ValueError:
                    cp_dt = dateutil.parser.parse(timestamp,
                                                  dayfirst=dayfirst)
                cps.append((cp_dt, tag))
            # sort by ascending datetime
            cps.sort(key=lambda tup: tup[0])
            feat = np.zeros(len(self.dts))
            for (cp_dt, tag) in cps:
                ind = np.where(self.dts >= cp_dt)[0]
                if len(ind) > 0:
                    feat[ind[0]:] = tag
        else:
            feat = None
        return feat

    def split_dataset(self, test_size):
        if self.X_standardizer is None:
            self.X_standardizer = preprocessing.StandardScaler().fit(self.X)
        self.X_s = self.X_standardizer.transform(self.X)
        if self.y is not None:
            self.y_standardizer = preprocessing.StandardScaler().fit(self.y.reshape(-1, 1))
            self.y_s = self.y_standardizer.transform(self.y.reshape(-1, 1)).ravel()
            if self.cps is not None:
                pre_inds = np.where(self.cps == self.PRE_DATA_TAG)
                post_inds = np.where(self.cps == self.POST_DATA_TAG)
                self.X_pre_s, self.X_post_s = self.X_s[pre_inds],self.X_s[post_inds]
                self.y_pre_s, self.y_post_s = self.y_s[pre_inds],self.y_s[post_inds]
                self.dts_pre, self.dts_post = \
                     self.dts[pre_inds], self.dts[post_inds]
            else:
                # handle case where no changepoint is given
                # by using a predefined fraction of the dataset
                # to split into pre and post datasets.
                # this is useful for testing the accuracy of modeling methods
                # for datasets in which no retrofit is known to have occurred
                pre = int(len(self.X_s)*(1-test_size))
                self.X_pre_s, self.X_post_s = self.X_s[:pre], self.X_s[pre:]
                self.y_pre_s, self.y_post_s = self.y_s[:pre], self.y_s[pre:]
                self.dts_pre, self.dts_post = self.dts[:pre], self.dts[pre:]

    def join_recarrays(self,arrays):
        newtype = sum((a.dtype.descr for a in arrays), [])
        newrecarray = np.empty(len(arrays[0]), dtype=newtype)
        for a in arrays:
            for name in a.dtype.names:
                newrecarray[name] = a[name]
        return newrecarray

class ModelAggregator(object):

    def __init__(self, dataset):
        self.dataset = dataset
        self.models = []
        self.best_model = None
        self.best_score = -np.inf  # Initialize to negative infinity
        self.error_metrics = None

    def train_dummy(self, **kwargs):
        dummy_trainer = mave.trainers.DummyTrainer(**kwargs)
        dummy_trainer.train(self.dataset, randomized_search=False)
        self.models.append(dummy_trainer.model)
        return dummy_trainer.model

    def train_hour_weekday(self, **kwargs):
        hour_weekday_trainer = mave.trainers.HourWeekdayBinModelTrainer(**kwargs)
        hour_weekday_trainer.train(self.dataset, randomized_search=False)
        self.models.append(hour_weekday_trainer.model)
        return hour_weekday_trainer.model

    def train_kneighbors(self, **kwargs):
        kneighbors_trainer = mave.trainers.KNeighborsTrainer(**kwargs)
        kneighbors_trainer.train(self.dataset)
        self.models.append(kneighbors_trainer.model)
        return kneighbors_trainer.model

    def train_svr(self, **kwargs):
        svr_trainer = mave.trainers.SVRTrainer(**kwargs)
        svr_trainer.train(self.dataset)
        self.models.append(svr_trainer.model)
        return svr_trainer.model

    def train_gradient_boosting(self, **kwargs):
        gradient_boosting_trainer = mave.trainers.GradientBoostingTrainer(**kwargs)
        gradient_boosting_trainer.train(self.dataset)
        self.models.append(gradient_boosting_trainer.model)
        return gradient_boosting_trainer.model

    def train_random_forest(self, **kwargs):
        random_forest_trainer = mave.trainers.RandomForestTrainer(**kwargs)
        random_forest_trainer.train(self.dataset)
        self.models.append(random_forest_trainer.model)
        return random_forest_trainer.model

    def train_extra_trees(self, **kwargs):
        extra_trees_trainer = mave.trainers.ExtraTreesTrainer(**kwargs)
        extra_trees_trainer.train(self.dataset)
        self.models.append(extra_trees_trainer.model)
        return extra_trees_trainer.model

    def train_all(self, **kwargs):
        log.info("Training hour and weekday binning models")
        self.train_hour_weekday(**kwargs)
        log.info("Training random forest regressor models")
        self.train_random_forest(**kwargs)
        self.select_model()
        self.score()
        return self.models

    def select_model(self):
        for model in self.models:
            if hasattr(model, 'best_score_') and hasattr(model, 'best_params_'):
                log.info(("Best %s model R2 score: %s, with parameters: %s"
                         %(str(model.estimator).split('(')[0],
                           model.best_score_,
                           model.best_params_)))
                if model.best_score_ > self.best_score:
                    self.best_score = model.best_score_
                    self.best_model = model
        return self.best_model, self.best_score

    def score(self):
        if self.best_model is not None:
            prediction = self.dataset.y_standardizer.inverse_transform(\
                                          self.best_model.predict(self.dataset.X_s).reshape(-1, 1)).ravel()
            self.error_metrics = mave.comparer.Comparer(comparison=prediction,\
                                                   baseline=self.dataset.y)
        return self.error_metrics

    def save(self, model_type):
        os.chdir(os.path.join(os.getcwd(),'models'))
        for m in self.models:
            if hasattr(m, 'best_estimator_'):
                m_name = str(m.best_estimator_).split('(')[0]
                with open(model_type+'_%s_model'%m_name, 'wb') as f:
                    pickle.dump(m.best_estimator_, f, -1)
        if self.error_metrics:
            with open(model_type+'_best_model_error_metrics.pkl', 'wb') as f:
                pickle.dump(self.error_metrics, f, -1)
        os.chdir('..')

    def __str__(self):
        if self.best_model is None:
            return "No models trained yet."
            
        rv = "\n\n=== Selected model ==="
        rv += "\nBest cross validation score on training data: %s"%\
                                                   self.best_model.best_score_
        rv += "\nBest model:\n%s"%self.best_model.best_estimator_
        try:
            imps = self.best_model.best_estimator_.feature_importances_
            feats = self.dataset.feature_names
            rv += ("\nThe relative importances of input features are:\n%s"%
                  pprint.pformat([f+': '+str(i) for f,i in zip(feats,imps)]))
        except:
            rv += ""
        rv += "\n\n=== Fit to the training data ==="
        rv += "\nThese error metrics represent the match between the"+ \
               " pre-retrofit data used to train the model and" + \
               " the model prediction:"
        # check if the results meet the ASHRAE Guideline 14:2002 criteria
        if self.error_metrics:
            if self.error_metrics.cvrmse <= 30 \
                and abs(self.error_metrics.nmbe) <= 10:
                self.meets_criteria = True
            else:
                self.meets_criteria = False
            rv += '\n\nThe model %s the ASHRAE Guideline 14:2002 criteria.'\
                  %(['does not meet', 'meets'][self.meets_criteria])
            rv += str(self.error_metrics)
        return rv

class mave(object):
    def __init__(self,
                 input_file_path,
                 save=False,
                 address=None,
                 plot=False,
                 k=10,
                 datetime_column_name='LocalDateTime',
                 target_column_name = 'EnergyConsumption',
                 ignored_column_names = [],
                 **kwargs):
        if save:
            res_dir = 'results_'+os.path.basename(input_file_path)
            if os.path.exists(res_dir):
                shutil.rmtree(res_dir)
            os.makedirs(os.path.join(res_dir,'models'))
            os.makedirs(os.path.join(res_dir,'data'))
            os.chdir(res_dir)
            shutil.copyfile(input_file_path, 'original_input_file.csv')
        
        # Skip location for now - requires proper location module
        self.locale = None
        if address:
            log.info("Location services not implemented. Skipping location-based features.")
       
        # read up to the first 100 lines of the input_file to check columns 
        with open(input_file_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            headers = []
            cols = None
            for _ in range(100):
                try:
                    row = next(reader)
                    headers.append(row)
                    if len(row)>0:
                        if datetime_column_name in row:
                            cols = [c for c in row \
                                    if not c in ignored_column_names \
                                    and c != '']
                            log.info(("Ignoring the following columns named:%s"
                                      %ignored_column_names))
                            break
                except StopIteration:
                    break
        if cols is None:
            log.error(("Datetime column name %s not found in the input file"
                       ". Please either edit the input file or the config"
                       " file to correctly identify the datetime column" 
                       %datetime_column_name))
            raise Exception("Datetime column name not found in input file",
                            datetime_column_name)
        if not target_column_name in cols:
            log.error(("Target column name %s not found in the input file"
                       ". Please either edit the input file or the config"
                       " file to correctly identify the target column" 
                       %target_column_name))
            raise Exception("Target column name not found in input file",
                            target_column_name)
        n_headers = len(headers)-1
        
        # pre-process the input data file
        log.info("Preprocessing the input file")
        self.p = Preprocessor(input_file_path,
                              locale=self.locale,
                              target_column_name=target_column_name,
                              datetime_column_name=datetime_column_name,
                              n_headers=n_headers,
                              column_names=cols,
                              **kwargs)
        self.use_tmy = self.p.use_tmy
        
        # create datasets
        self.A = mave.dataset.Dataset(dataset_type='A',
                                 X_s=self.p.X_pre_s,
                                 X_standardizer=self.p.X_standardizer,
                                 y_s=self.p.y_pre_s,
                                 y_standardizer=self.p.y_standardizer,
                                 dts=self.p.dts_pre,
                                 feature_names=self.p.feature_names,
                                 save=save)
        self.D = mave.dataset.Dataset(dataset_type='D',
                                 X_s=self.p.X_post_s,
                                 X_standardizer=self.p.X_standardizer,
                                 y_s=self.p.y_post_s,
                                 y_standardizer=self.p.y_standardizer,
                                 dts=self.p.dts_post,
                                 feature_names=self.p.feature_names,
                                 save=save)
        
        self.m_pre = ModelAggregator(dataset=self.A)
        folds = model_selection.KFold(n_splits=k, shuffle=True, random_state=42)
        log.info("Fitting models to the pre-retrofit data")
        self.m_pre.train_all(k = folds, **kwargs)
        
        # single model (no weather lookup, no tmy normalization)
        # evaluate the output of the model against the post-retrofit data
        if self.m_pre.best_model:
            self.E = mave.dataset.Dataset(
                dataset_type='E',
                base_dataset= self.D,
                y_s=self.m_pre.best_model.predict(self.D.X_s),
                save=save)
            self.DvsE = mave.comparer.Comparer(comparison=self.E, baseline=self.D)
            if save:
                self.m_pre.save('pre-retrofit')
                ofp = os.path.join('post-retrofit_error_metrics.pkl')
                with open(ofp, 'wb') as f:
                    pickle.dump(self.DvsE, f, -1)
        
        # TMY analysis skipped - requires location module
        if save:
            with open(os.path.join('text_results.txt'), "w") as f:
                f.write(str(self))

    def __str__(self):
        rv = "\n\n===== Pre-retrofit model training summary ====="
        rv += str(self.m_pre)
        rv += "\n\n===== Results ====="
        rv += "\nThese results quantify the difference between the"+ \
              " measured post-retrofit data and the predicted" + \
              " consumption:"
        if hasattr(self, 'DvsE'):
            rv += str(self.DvsE)
        else:
            rv += "\nNo comparison results available."
        return rv

if __name__=='__main__':
    # Example usage
    cps = [
           ("2012/1/29 13:15", Preprocessor.PRE_DATA_TAG),
           ("2012/12/20 01:15", Preprocessor.IGNORE_TAG),
           ("2013/1/1 01:15", Preprocessor.PRE_DATA_TAG),
           ("2013/9/14 23:15", Preprocessor.POST_DATA_TAG),
          ]
    # Uncomment and modify path as needed
    # mnv = mave(input_file_path='data/ex2.csv',
    #           changepoints=cps,
    #           address=None,
    #           save=True)
    # print(mnv)
