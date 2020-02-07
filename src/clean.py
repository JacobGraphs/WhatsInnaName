import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string

# This reads in the original IMDB data
movies = pd.read_csv('/Users/jacobtryba/DSI/assignments/capstone2/data/imdb-extensive-dataset/IMDb movies.csv')
# This is the subset of columns needed for the Naive Bayes model to be applied later
movies_subset = ['reviews_from_critics','genre','description', 'metascore','director', 'title', 'year','duration', 'country','budget', 'usa_gross_income', 'worlwide_gross_income']
movies_subbed = movies[movies_subset]
# This removes all non-USA domestic films
movies_current_usa = movies_subbed.query('country == "USA"')
# This removes all null values for the USA Gross Return column
movies_current_usa_nonnull_ugi = movies_current_usa[(movies_current_usa.usa_gross_income.notnull())]
# This removes all null values for the budget column
movies_current_usa_nonnull_ugi_budget = movies_current_usa_nonnull_ugi[(movies_current_usa_nonnull_ugi.budget.notnull())]
# Sorts the list by year
final_set = movies_current_usa_nonnull_ugi_budget.sort_values('year', ascending = True)
# Removes all of the extraneous characters and information from the budget column. This includes SOME data points with a different currency listed.
# All currencies are pretty close to the dollar (none are orders 100's per USD), and I only need to know if the return cleared budget, not to what extent.
# This leaves a small window for each film with the listed currency (very few) to mark as not exceeding the budget when it actually did.
final_set['budget'] = final_set['budget'].str.replace('$', '')
final_set['budget'] = final_set['budget'].str.replace('$ ', '')
final_set['budget'] = final_set['budget'].str.replace('GBP ', '')
final_set['budget'] = final_set['budget'].str.replace('AUD ', '')
final_set['budget'] = final_set['budget'].str.replace('EUR ', '')
final_set['budget'] = final_set['budget'].str.replace('ESP ', '')
final_set['budget'] = final_set['budget'].str.replace('CAD ', '')
# This removes the $ in budget and worldwide returns and then turns the column into a string
final_set['usa_gross_income'] = final_set['usa_gross_income'].str.replace('$ ', '')
final_set['usa_gross_income'] = final_set['usa_gross_income'].str.replace('$', '').astype('int')
final_set['worlwide_gross_income'] = final_set['worlwide_gross_income'].str.replace('$', '').astype('int')
# This initiates an international gross income for future manipulation and analysis
final_set['international_gross_income'] = (final_set['worlwide_gross_income'] - final_set['usa_gross_income'])
# This initiates a column for a conditional check to initiate the "Profitable" boolean column.
final_set['returns'] = (final_set['worlwide_gross_income'] - final_set['budget'].astype('int'))
final_set['profitable'] = [1 if x > 0 else 0 for x in final_set['returns']]
# This finally removes nulls from the metascore column for future analysis.
final_set = final_set[(final_set.metascore.notnull())]

f = final_set.to_csv('/Users/jacobtryba/DSI/assignments/capstone2/data/cleaned_data.csv')