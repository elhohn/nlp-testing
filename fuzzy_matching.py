# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 14:48:22 2025

@author: Elliot.Hohn
"""

import pandas as pd
from fuzzywuzzy import process
#from fuzzy_pandas import fuzzy_merge

# load data
omdg = pd.read_excel('USASpending Descriptions FY22-24.xlsx', sheet_name='OMDG')
usas = pd.read_excel('data_pulled_from_usaspending.xlsx')

omdg = omdg[['Legal Name', 'Project Summary']]
#omdg['name_for_match'] = omdg['Legal Name'].str.upper().str.replace(r'[.,]', '', regex=True)

usas['fy'] = usas['period_of_performance_start_date'].dt.year
usas = usas[['award_id_fain', 'recipient_name', 'prime_award_base_transaction_description', 'fy']]
usas = usas[(usas['fy'] >= 2022) & (usas['fy'] <= 2024)]

best = usas.head(100)
best['match_name'] = best['recipient_name'].apply(lambda x: process.extractOne(x, omdg['Legal Name'])[0])
best['match_score_name'] = best['recipient_name'].apply(lambda x: process.extractOne(x, omdg['Legal Name'])[1])
best['match_desc'] = best['prime_award_base_transaction_description'].apply(lambda x: process.extractOne(x, omdg['Project Summary'])[0])
best['match_score_desc'] = best['prime_award_base_transaction_description'].apply(lambda x: process.extractOne(x, omdg['Project Summary'])[1])
best['total_match_score'] = best['match_score'] + best['total_match_score']


best = best.loc[best.groupby('Legal Name')['total_match_score'].idxmax()]

# merge based on fuzzy matches
merged = pd.merge(omdg, usas, left_on = 'Legal Name', right_on='match_name',
                   how = 'left')

#merged2 = fuzzy_merge(omdg, usas, left_on = 'match_name', right_on='recipient_name', method='jaro', keep='all')