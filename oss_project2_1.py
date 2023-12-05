import pandas as pd
from IPython.display import display
table = pd.read_csv("2019_kbo_for_kaggle_v2.csv")

#### 1번 ####
print('Requirement 1')
for i in range(2015, 2019):
  print('Top 10 players in', i)
  table1 = table[table['year']==i]
  for j in {'H', 'avg', 'HR', 'OBP'}:
    top10 = table1.sort_values(by=[j], ascending=False)[:10]
    top10 = top10[['batter_name']].T
    print(j, top10.values)
  print('\n')

#### 2번 ####
print('\nRequirement 2')
table2 = table[table['year']==2018]
# positions = {'포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수'}
for i in table2.groupby('cp')['war'].max():
  print(table2[table2['war']==i][['cp', 'batter_name', 'war', 'year']].values)

#### 3번 ####
print('\nRequirement 3')
table3 = table[['H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary']]
print(table3.corr(method='pearson').loc['salary'].sort_values())
print('\n \'RBI\' has the highest correlation with salary')
