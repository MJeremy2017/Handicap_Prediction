# the handicap of sg is not reasonable
# try to predict a more reasonable handicap from other bookmakers and put it into the data
from sklearn.linear_model import LinearRegression, BayesianRidge, Lasso
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
import sklearn.metrics as m
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.stats.stats import pearsonr

# get the original data
# home_odds, draw_odds, away_odds, home_handicap
org_data = np.array([[4.15, 4.00, 1.64, 0.75], [1.48, 4.5, 4.95, -1], [1.24, 5.7, 8.5, -1.75],
                     [1.59, 4.00, 4.5, -1], [4.1, 3.7, 1.9, .5], [2.4, 3.5, 2.95, -0.25],
                     [1.88, 3.55, 4.45, -0.5], [9.5, 5.5, 1.32, 1.5], [2.38, 3.15, 3.3, -0.25],
                     [2.53, 3.25, 2.53, 0], [2.51, 3.1, 2.63, 0], [1.15, 7.0, 12.0, -2.25],
                     [2.42, 3.6, 2.45, 0], [1.14, 6.6, 14.0, -2], [3.30, 3.2, 2.2, 0.25],
                     [1.33, 4.65, 6.4, -1.5], [17.5, 9.6, 1.04, 2.75], [10.5, 5.7, 1.18, 1.75],
                     [1.02, 11.5, 19, -3.25]])

org_data = pd.DataFrame(org_data, columns=['home_odds', 'draw_odds', 'away_odds', 'handicap'])
org_data.shape  # 19 rows

# derive variables
pearsonr(org_data['home_odds'], org_data['handicap'])
pearsonr(org_data['draw_odds'], abs(org_data['handicap']))
pearsonr(org_data['away_odds'], org_data['handicap'])
pearsonr(org_data['home_odds']-org_data['away_odds'], org_data['handicap'])  # .98
pearsonr(org_data['home_odds']/org_data['away_odds'], org_data['handicap'])  # .78

org_data['home_away_diff'] = org_data['home_odds']-org_data['away_odds']
org_data['home_away_dev'] = org_data['home_odds']/org_data['away_odds']
# write the file
org_data.to_csv('HandiPred/pred_handi.csv', index=False)

# run models ...
X_train, X_test, y_train, y_test = train_test_split(org_data.drop(['handicap'], axis=1),
                                                    org_data['handicap'], test_size=.3)

classifiers = {'Linear Regression': LinearRegression(), 'Lasso': Lasso(), 'Ridge': BayesianRidge(),
               'SVM': SVR()}

# running on fewer vars
result = {}
for name, clf in classifiers.items():
    print 'Running', name
    clf.fit(X_train, y_train)
    if name != 'SVM':
        print 'R-square', clf.score(X_train, y_train)
    result[name] = clf.predict(X_test)
result['actual'] = y_test
result = pd.DataFrame(result)
result['mean'] = result[['Lasso', 'Linear Regression', 'Ridge', 'SVM']].mean(axis=1)
result['max'] = result[['Lasso', 'Linear Regression', 'Ridge', 'SVM']].max(axis=1)

# max is better
# predict handicap
final_vars = pd.read_csv('HandiPred/final_vars.csv')
final_vars.columns

# modeling ...
print 'Original correlation of handicap and score_diff', pearsonr(final_vars['1/2 handicap'], final_vars['score_diff'])
res = []
for name, clf in classifiers.items():
    print 'Running', name
    clf.fit(org_data.drop(['handicap'], axis=1), org_data['handicap'])
    y_pred = clf.predict(final_vars[['1x2_home_open', '1x2_draw_open', '1x2_away_open',
                                     '1x2_home_away_open_diff', '1x2_home_away_open_ratio']])
    print 'Correlation of', name, 'is', pearsonr(y_pred, final_vars['score_diff'])
    res.append(y_pred)
res = np.array(res)
res_max = []
for i in range(res.shape[1]):
    row = res[:, i]
    row_max = row[np.argmax(abs(row))]
    res_max.append(row_max)

print 'Final correlation is', pearsonr(res_max, final_vars['score_diff'])  # 0.4

mae = m.make_scorer(m.mean_absolute_error)
final_vars['handicap_reg'] = res_max
fewer_lgb_mad = cross_val_score(LGBMRegressor(n_estimators=1200), X=final_vars.drop(['score_diff', '1/2 handicap'], axis=1),
                y=final_vars['score_diff'], cv=5, scoring=mae, n_jobs=2)
print 'MAD', np.mean(fewer_lgb_mad)  # 1.26 no difference!!!!

# final_vars.to_csv('HandiPred/final_var_test1.csv', index=False)  # no use

