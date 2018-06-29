# use evaluation
# new actual labels are score_diff + handicap > 0

import pandas as pd
import numpy as np
import sklearn.metrics as m
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

# the input should be score_diff+handicap, prediction+handicap


def acc(y_actual, y_pred):
    y_actual = np.array(y_actual)
    y_pred = np.array(y_pred)
    assert len(y_actual) == len(y_pred)
    return np.sum(y_actual==y_pred)/float(len(y_actual))


if __name__ == '__main__':
    full_vars = pd.read_csv('HandiPred/final_var_with_rounded.csv')
    # get the sub_data
    # full_vars = full_vars[abs(full_vars['1/2 handicap']) == 3.5]
    # full_vars.drop(['away', 'home', 'league'], axis=1, inplace=True)
    X_train, X_test, y_train, y_test = \
        train_test_split(full_vars.drop(['score_diff'], axis=1), full_vars['score_diff'], test_size=.3)
    lgb = LGBMRegressor(n_estimators=1200, learning_rate=.01)
    lgb.fit(X_train, y_train)
    lgb_pred = lgb.predict(X_test)

    y_actual = (y_test+X_test['rounded_handicap']).apply(lambda x: 1 if x > 0 else 0)
    y_pred = (lgb_pred+X_test['rounded_handicap']).apply(lambda x: 1 if x > 0 else 0)
    #
    # y_actual = (y_test + X_test['handicap_reg']).apply(lambda x: 1 if x > 0 else 0)
    # y_pred = (lgb_pred + X_test['handicap_reg']).apply(lambda x: 1 if x > 0 else 0)
    accuracy = acc(y_actual, y_pred)
    mad = m.mean_absolute_error(y_test, lgb_pred)
    print 'Accuracy:', accuracy  # 74%
    print 'MAD:', mad  # 1.21



