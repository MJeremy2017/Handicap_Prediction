import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from scipy.stats.stats import pearsonr
from sklearn.model_selection import cross_val_score
import sklearn.metrics as m

all_var = pd.read_csv('HandiPred/all_var.csv')
all_var.shape
all_var.columns
# pearsonr(abs(var_odds['1/2 away_close']), var_odds['score_diff'])


def get_vars():
    var_odds = all_var.drop(['away', 'date', 'day', 'home', 'league', 'month', 'year'], axis=1)
    var_odds['1/2 away_ratio'] = var_odds['1/2 away_close'] / var_odds['1/2 away_open']
    var_odds['1/2 home_ratio'] = var_odds['1/2 home_close'] / var_odds['1/2 home_open']
    var_odds['1x2_away_ratio'] = var_odds['1x2_away_close'] / var_odds['1x2_away_open']
    var_odds['1x2_home_ratio'] = var_odds['1x2_home_close'] / var_odds['1x2_home_open']
    var_odds['1x2_draw_ratio'] = var_odds['1x2_draw_close'] / var_odds['1x2_draw_open']

    var_odds['1/2 away_diff'] = var_odds['1/2 away_close'] - var_odds['1/2 away_open']
    var_odds['1/2 home_diff'] = var_odds['1/2 home_close'] - var_odds['1/2 home_open']
    var_odds['1x2_away_diff'] = var_odds['1x2_away_close'] - var_odds['1x2_away_open']
    var_odds['1x2_home_diff'] = var_odds['1x2_home_close'] - var_odds['1x2_home_open']
    var_odds['1x2_draw_diff'] = var_odds['1x2_draw_close'] - var_odds['1x2_draw_open']

    var_odds['1/2 away_add'] = var_odds['1/2 away_close'] + var_odds['1/2 away_open']
    var_odds['1/2 home_add'] = var_odds['1/2 home_close'] + var_odds['1/2 home_open']
    var_odds['1x2_away_add'] = var_odds['1x2_away_close'] + var_odds['1x2_away_open']
    var_odds['1x2_home_add'] = var_odds['1x2_home_close'] + var_odds['1x2_home_open']
    var_odds['1x2_draw_add'] = var_odds['1x2_draw_close'] + var_odds['1x2_draw_open']

    var_odds['1/2 away_prod'] = var_odds['1/2 away_close'] * var_odds['1/2 away_open']
    var_odds['1/2 home_prod'] = var_odds['1/2 home_close'] * var_odds['1/2 home_open']
    var_odds['1x2_away_prod'] = var_odds['1x2_away_close'] * var_odds['1x2_away_open']
    var_odds['1x2_home_prod'] = var_odds['1x2_home_close'] * var_odds['1x2_home_open']
    var_odds['1x2_draw_prod'] = var_odds['1x2_draw_close'] * var_odds['1x2_draw_open']
    print var_odds['1x2_draw_prod'].iloc[:3]
    # home and away odds
    var_odds['1/2 home_away_open_add'] = var_odds['1/2 home_open'] + var_odds['1/2 away_open']
    var_odds['1/2 home_away_close_add'] = var_odds['1/2 home_close'] + var_odds['1/2 away_close']
    var_odds['1x2_home_away_open_add'] = var_odds['1x2_home_open'] + var_odds['1x2_away_open']
    var_odds['1x2_home_away_close_add'] = var_odds['1x2_home_close'] + var_odds['1x2_away_close']
    print var_odds['1x2_draw_prod'].iloc[:3]
    var_odds['1/2 home_away_open_diff'] = var_odds['1/2 home_open'] - var_odds['1/2 away_open']
    var_odds['1/2 home_away_close_diff'] = var_odds['1/2 home_close'] - var_odds['1/2 away_close']
    var_odds['1x2_home_away_open_diff'] = var_odds['1x2_home_open'] - var_odds['1x2_away_open']
    var_odds['1x2_home_away_close_diff'] = var_odds['1x2_home_close'] - var_odds['1x2_away_close']
    print var_odds['1x2_draw_prod'].iloc[:3]
    var_odds['1x2_home_away_open_ratio'] = var_odds['1x2_home_open'] / var_odds['1x2_away_open']
    var_odds['1x2_home_away_close_ratio'] = var_odds['1x2_home_close'] / var_odds['1x2_away_close']
    print var_odds['1x2_draw_prod'].iloc[:3]
    return var_odds

full_vars = get_vars()
full_vars.shape
full_vars.to_csv('HandiPred/full_vars2.csv', index=False)  # now correct

var_odds = all_var.drop(['away', 'date', 'day', 'home', 'league', 'month', 'year'])
# modeling
mae = m.make_scorer(m.mean_absolute_error)

base_mad = cross_val_score(LGBMRegressor(n_estimators=1200), X=var_odds.drop('score_diff', axis=1),
                y=var_odds['score_diff'], cv=5, scoring=mae, n_jobs=2)

print(np.mean(base_mad))

lgb_mad = cross_val_score(LGBMRegressor(n_estimators=1200), X=full_vars.drop('score_diff', axis=1),
                y=full_vars['score_diff'], cv=5, scoring=mae, n_jobs=2)

print(np.mean(lgb_mad))

# try fewer variables
full_vars.columns
fewer_vars = full_vars[['1/2 handicap', '1/2 home_away_open_add', '1x2_home_away_open_add',
                        '1/2 home_away_open_diff', '1x2_home_away_open_diff', 'score_diff']]
# try another
pearsonr(full_vars['1/2 handicap'], full_vars['score_diff'])  # 0.31
pearsonr(full_vars['1/2 home_away_open_add'], full_vars['score_diff'])  # 0.14
pearsonr(full_vars['1/2 home_away_open_diff'], full_vars['score_diff'])  # 0.11
pearsonr(full_vars['1x2_home_away_open_diff'], full_vars['score_diff'])  # 0.4

fewer_vars = full_vars[[u'1/2 away_close', u'1/2 away_open', u'1/2 handicap', u'1/2 home_close',
                        u'1/2 home_open', u'1x2_away_close', u'1x2_away_open',
                        u'1x2_draw_close', u'1x2_draw_open', u'1x2_home_close',
                        u'1x2_home_open', u'score_diff',
                        u'1/2 home_away_open_diff', u'1/2 home_away_close_diff',
                        u'1x2_home_away_open_diff', u'1x2_home_away_close_diff',
                        u'1x2_home_away_open_ratio', u'1x2_home_away_close_ratio'
                        ]]

# 1/2 use close, 1x2 use open
fewer_vars = full_vars[[u'1/2 away_close', u'1/2 handicap', u'1/2 home_close',
                        u'1x2_away_open', u'1x2_draw_open', u'1x2_home_open',
                        u'score_diff', u'1/2 home_away_close_diff',
                        u'1x2_home_away_open_diff', u'1x2_home_away_open_ratio']]

fewer_lgb_mad = cross_val_score(LGBMRegressor(n_estimators=1200), X=fewer_vars.drop('score_diff', axis=1),
                                y=fewer_vars['score_diff'], cv=5, scoring=mae, n_jobs=2)

print(np.mean(fewer_lgb_mad))

fewer_vars.to_csv('HandiPred/final_vars.csv', index=False)

# try simple linear regression
classifiers = {'Linear Regression': LinearRegression(), 'Logistic Regression': LogisticRegression(),
               'Lasso': Lasso()}

# running on fewer vars
for name, clf in classifiers.items():
    print 'Running', name
    scores = cross_val_score(clf, X=fewer_vars.drop('score_diff', axis=1),
                y=fewer_vars['score_diff'], cv=5, scoring=mae, n_jobs=2)
    print 'Scores', scores, 'Mean', np.mean(scores)

# running on all vars
for name, clf in classifiers.items():
    print 'Running', name
    scores = cross_val_score(clf, X=full_vars.drop('score_diff', axis=1),
                y=full_vars['score_diff'], cv=5, scoring=mae, n_jobs=2)
    print 'Scores', scores, 'Mean', np.mean(scores)


def run_models(data):
    classifiers = {'Linear Regression': LinearRegression(), 'Logistic Regression': LogisticRegression(),
                   'Lasso': Lasso()}
    # running on fewer vars
    for name, clf in classifiers.items():
        print 'Running', name
        scores = cross_val_score(clf, X=data.drop('score_diff', axis=1),
                                 y=data['score_diff'], cv=5, scoring=mae, n_jobs=2)
        print 'Scores', scores, 'Mean', np.mean(scores)


# worse
# ======================================= #
# just to write to file
full_vars['league'] = all_var['league']
full_vars['home'] = all_var['home']
full_vars['away'] = all_var['away']
full_vars.to_csv('HandiPred/full_vars.csv', index=False)

full_vars = pd.read_csv('HandiPred/full_vars.csv')

run_models(fewer_vars[fewer_vars['1/2 handicap'] == 3.5])
# how about use odds difference to predict handicap