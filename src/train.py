from pre_train.py import test, train

# Creates test/train data for two groups
train_title_genre_director = train.copy()
test_title_genre_director = test.copy()
train_all = train.copy()
test_all = test.copy()

# Initiates a column for NB utilizing movie title, the genres, and the director
train_title_genre_director['text'] = train_title_genre_director['title'] + ' ' + train_title_genre_director['genre'] + ' ' + train_title_genre_director['director']
test_title_genre_director['text'] = test_title_genre_director['title'] + ' ' + test_title_genre_director['genre'] + ' ' + test_title_genre_director['director']
# Initiates a column for NB utilizing all
train_all['text'] = train_all['title'] + ' ' + train_all['description'] + ' ' + train_all['director'] + ' ' + train_all['genre']
test_all['text'] = test_all['title'] + ' ' + test_all['description'] + ' ' + test_all['director'] + ' ' + test_all['genre']

subset = ['title', 'director', 'text', 'profitable']
train_title_genre_director = train_title_genre_director[subset]
test_title_genre_director = test_title_genre_director[subset]

train_all = train_all[subset]
test_all = test_all[subset]