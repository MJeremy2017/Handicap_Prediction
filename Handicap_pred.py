import pypyodbc
import pandas as pd
import re
# import datetime

conn = pypyodbc.connect(
    "Driver={Microsoft Access Driver (*.mdb, *.accdb)};"
    "Dbq=D:\R\CustomerPrediction\Odds - 2010 to 2016 Oct.accdb;")

q = "select * from sprtOdds " \
    "WHERE `Bet Type` IN ('1X2', '1/2 goal')"
oddsExt = pd.read_sql(q, conn)
oddsExt.shape

oddsExt = oddsExt[~ oddsExt['book title'].str.contains('Live')]
oddsExt = oddsExt[~ oddsExt['league name'].str.contains('LIVE')]
oddsExt.to_csv('HandiPred/oddsExt.csv', index=False)

oddsExt = pd.read_csv('HandiPred/oddsExt.csv', parse_dates=True)
oddsExt['book date'] = pd.to_datetime(oddsExt['book date'], format='%Y-%m-%d', unit='D')
# delete rows unsafisfied
book_title = oddsExt['book title']
del_idx = []
start_idx = 0
while start_idx < len(book_title):
    print('current row:', start_idx)
    cur_match = book_title.iloc[start_idx]
    for j in range(1, 5):
        if book_title.iloc[start_idx+j] != cur_match:
            flag = False
            del_idx.extend(range(start_idx, start_idx+j))
            start_idx = start_idx+j
            break
        else:
            flag = True
    if flag:
        start_idx += 5

oddsExt.drop(oddsExt.index[del_idx], inplace=True)
oddsExt.to_csv('HandiPred/oddsExt.csv', index=False)

bet_type = oddsExt['bet type']
del_idx = []
start_idx = 0
while start_idx < len(bet_type):
    print('current row:', start_idx)
    if bet_type.iloc[start_idx] != '1X2' or bet_type.iloc[start_idx+1] != '1X2':
        del_idx.extend(range(start_idx, start_idx+5))
    start_idx += 5

oddsExt = pd.read_csv('HandiPred/oddsExt.csv', parse_dates=True)
oddsExt['book date'] = pd.to_datetime(oddsExt['book date'], format='%Y-%m-%d %H:%M')
oddsExt['book date'] = oddsExt['book date'].apply(lambda x: x.date())

# get the score of matches
conn = pypyodbc.connect(
    "Driver={Microsoft Access Driver (*.mdb, *.accdb)};"
    "Dbq=D:\R\CustomerPrediction\Monthly report (data from 1Apr08 onwards) - till 24 Oct 2016;")

q = "select `Book Date`, `League Name`, `Book Title`, `Bet Type`, `Selections`, `Win Flag` from sprtOdds " \
    "WHERE `Bet Type` = 'Pick The Score' AND `Win Flag` = 'Y'"
match_score = pd.read_sql(q, conn)
match_score['book date'] = match_score['book date'].apply(lambda x: x.date())

full_score = match_score.merge(oddsExt, on=['book date', 'book title'])

oddsExt = full_score
oddsExt = oddsExt[oddsExt['selections_x'] != 'Any Other Score']
oddsExt.to_csv('HandiPred/full_score.csv', index=False)

oddsExt = pd.read_csv('HandiPred/full_score.csv')
oddsExt['book date'] = pd.to_datetime(oddsExt['book date'], format='%m/%d/%Y')
# get features
# date--, league, home, away, 1x2_draw_open, 1x2_home_open, 1x2_away_open,
# 1x2_draw_close, 1x2_home_close, 1x2_away_close, 1/2_handicap, 1/2_home_open
# 1/2_away_open, 1/2_home_close, 1/2_away_close
# top 5 includes data for one match
def melt_table():
    date, year, month, day = [], [], [], []
    league, home, away, draw_open_1x2, home_open_1x2, away_open_1x2 = [], [], [], [], [], []
    draw_close_1x2, home_close_1x2, away_close_1x2 = [], [], []
    handicap_half, home_open_half, away_open_half, home_close_half, away_close_half = [], [], [], [], []
    score_diff = []  # home - away
    num_matches = oddsExt.shape[0]/5
    for i in range(num_matches):
        start_row = 5*i
        print('i', 5*i)
        if oddsExt['bet type_y'].iloc[start_row] != '1X2':
            print('Error!!!!!!!!!!!!!!!')
        m_date = oddsExt['book date'].iloc[start_row]
        date.append(m_date)
        year.append(m_date.year)
        month.append(m_date.month)
        day.append(m_date.day)
        home.append(oddsExt['selections_y'].iloc[start_row+1])
        away.append(oddsExt['selections_y'].iloc[start_row+2])
        league.append(oddsExt['league name_x'].iloc[start_row])
        draw_open_1x2.append(oddsExt['open odds'].iloc[start_row])
        home_open_1x2.append(oddsExt['open odds'].iloc[start_row+1])
        away_open_1x2.append(oddsExt['open odds'].iloc[start_row+2])
        draw_close_1x2.append(oddsExt['close odds'].iloc[start_row])
        home_close_1x2.append(oddsExt['close odds'].iloc[start_row+1])
        away_close_1x2.append(oddsExt['close odds'].iloc[start_row+2])
        handi = re.findall('-?\d\.\d', oddsExt['selections_y'].iloc[start_row+3])
        handicap_half.append(float(handi[0]))
        print('handi', float(handi[0]))
        home_open_half.append(oddsExt['open odds'].iloc[start_row+3])
        away_open_half.append(oddsExt['open odds'].iloc[start_row+4])
        home_close_half.append(oddsExt['close odds'].iloc[start_row+3])
        away_close_half.append(oddsExt['close odds'].iloc[start_row+4])
        score = oddsExt['selections_x'].iloc[start_row]
        score_number = re.findall('\d', score)
        if oddsExt['selections_y'].iloc[start_row+1] in score:
            home_score = score_number[0]
            away_score = score_number[1]
        else:
            home_score = score_number[1]
            away_score = score_number[0]
        score_diff.append(int(home_score)-int(away_score))

    df = pd.DataFrame({'date': date, 'year': year, 'month': month,
                       'day': day, 'score_diff': score_diff,
                       'league': league, 'home': home, 'away': away,
                       '1x2_draw_open': draw_open_1x2, '1x2_home_open': home_open_1x2,
                       '1x2_away_open': away_open_1x2, '1x2_draw_close': draw_close_1x2,
                       '1x2_home_close': home_close_1x2, '1x2_away_close': away_close_1x2,
                       '1/2 handicap': handicap_half, '1/2 home_open': home_open_half,
                       '1/2 away_open': away_open_half, '1/2 home_close': home_close_half,
                       '1/2 away_close': away_close_half})

    return df


all_vars = melt_table()
all_vars.shape  # 31419 matches

all_vars.to_csv('HandiPred/all_var.csv', index=False)



