# WhatsInnaName
### Or In the Description, Director, and Genre?
Using Naive Bayes to predict movie profitability given only a few text columns



![moviessuck](https://github.com/JacobGraphs/WhatsInnaName/blob/master/img/movies_suck.jpg)
</br></br>
Some movies suck. Some movies don't. A decent amount of them make money. End of story.
</br></br>
No, there is more. Just not much more.

### Table of Contents
#### 1. Introduction
#### 2. Cleaning and Preparing for Analysis
#### 3. EDA
#### 4. Naive Bayes and Results
#### 5. Extraneous Tasks That May or May Not Have Been Needed From the Get Go
</br></br>
### Introduction

Just some facts about movies: </br></br>
There are <b>TWELVE</b> Star Trek movies</br></br>
Billy Bob Thornton directed a movie about <b>horses</b>, and they spent 60 million dollars making that thing, losing many millions. But at the same time, he wrote and directed Sling Blade and made 24x the budget back on gross returns.</br></br>
Woody Allen directed too many movies to count, lost money on many of them, and this tracking of Rotten Tomatoes Scores seems to illustrate the presumed issue here</br></br>
![woody](https://github.com/JacobGraphs/WhatsInnaName/blob/master/img/woody_allen.jpg)

</br></br></br>
I thought it may have a positive ROI for studios to invest some time in a machine learning algorithm to help determine if movies should be made or not. In order to do that, I'm going to make the assumption that BigShotStudioExec wants their movie to at least earn back the production budget. I don't think this is a lofty ask of your creatives: Produce something that costs less than what it collects from people. Instead of spending all day trying to decide if <b>*Fast and the Furious 14: Valentines Day*</b> or <b>*Hobbs & Shaw 2: Maybe Gerard Butler's a Villain*</b> will be the last hoorah for the Fast and Furious Series, I figure we let the computer do the work. There's enough movies by now that surely we can figure that out. 
</br></br></br>
To start, I'll go ahead and grab this dataset of movies pulled from IMDB. 
</br>https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset</br>
It's pretty clean and there's a lot of null's.

</br></br>
### Cleaning and Preparing for Analysis
Originally, the data columns are as follows: </br>
- imdb_title_id : long string of characters linked to the title of the movie on their end</br>
- title: the name of the movie</br>
- original title: same</br>
- year: year in which the movie was released</br>
- date published: date in which the movie was released</br>
- genre: string of usually three words indicating the genre mix of the movie</br>
- duration: length of the movie in minutes</br>
- country: origin country of the movie</br>
- language: the language of the movie</br>
- director: a string indicating director of the movie</br>
- writer: a string indicating the screenplay author</br>
- production company: string indicating the production company behind the movie</br>
- actors: a string of actors billed at the top of the movie</br>
- description: a multi-sentence paragraph describing the movie</br>
- avg_vote: an integer</br>
- votes: an integer</br>
- budget: an integer</br>
- usa_gross_income: an integer</br>
- worlwide_gross_income: an integer (and a misspelling)</br>
- meta_score: a float</br>
- reviews_from_users: a float</br>
- reviews_from_critics: a float</br></br>
To start, I subsetted the movies into only a few columns:</br></br> ('title', 'director', 'description', 'year', 'country', 'budget', 'usa_gross_income', and 'worlwide_gross_income')</br></br>
Second, I eliminated any movies where budget, usa_gross_income, and worlwide_gross_income did not have a value.</br></br>
I removed all movies that were not produced in the USA.</br></br>
I had to put in a .str.replace bit to remove '$' and allow the data to be manipulatable.</br></br>
If the total amount in worlwide_gross_income exceeded the budget, a column 'Profitable' would be marked 1, otherwise 0.</br></br> Next up was cleaning up the text columns for analysis. </br></br>
I removed all punctuation from description, title, and genre.</br></br>
Then I lowercased, description, title, genre, and director. </br></br>
Then I removed all stop words from description. 

</br></br>
Combining movie title, director, genre, and description into a singular column, 'text' was my last step before training and testing my data with the Naive Bayes model.
</br></br>
### EDA 
After cleaning everything up, I ran a frequency of directors in the set. I presumed that there would be larger distribution of "directors with x movies" in the profitable subset than the loss subset.
![All](https://github.com/JacobGraphs/WhatsInnaName/blob/master/img/alldirectorhistogram.png)
![Profit](https://github.com/JacobGraphs/WhatsInnaName/blob/master/img/profitdirectorhistogram.png)
![Loss](https://github.com/JacobGraphs/WhatsInnaName/blob/master/img/lossdirectorhistogram.png)
As visible above, there *is* a wider distribution of directors with x movies in the profitable subset than the loss subset. This is because once a director's first movie flops, they tend to not get multiple runs at it again, barring other factors.</br>
Here's the same histogram with all directors with only 1 movie under their belt removes. It becomes rather noticeable.
![Profit](https://github.com/JacobGraphs/WhatsInnaName/blob/master/img/limitprofitdirectorhistogram.png)
![Loss](https://github.com/JacobGraphs/WhatsInnaName/blob/master/img/limitlossdirectorhistogram.png)
</br> I was curious why someone with multiple losses was in the far right ends of the x-axis, so I did some digging. Turns out a lot of Woody Allen's movies straight up lose money. </br>
Here's a breakdown of the movie set filtered and cleaned.
![Hist](https://github.com/JacobGraphs/WhatsInnaName/blob/master/img/profitandnot.png) 
</br></br>
#### Naive Bayes and Results
After producing visuals breaking down director frequency, I moved onto predicting profitability. I took the text column I created earlier and fed it into a vectorizer with n_gram range of 1 to 2 so I could pick up titles and directors. My classifier was the Naive Bayes model and my targets were the binary profitable or not profitable. I randomly split my data and then trained and tested. I did a 70/30 split between training/testing. No matter the seed, I pretty much always got the same accuracy, precision, and recall. I'd have a ton of true positives, few true negatives, few false negatives, and many false positives. Below is the model of "choose the mode, always" and below that is an example of one of my tests.</br> </br>

</br>If you were to take the same data and choose the more likely outcome "Profitable" as your option, you'd have similar accuracy precision and recall, but zero true negatives. Considering the cost of a false positive is someone's millions of dollars funding a creative project apparently not many of us were asking for in the first place, maybe we should focus on getting fewer false positives and more true negatives. Here's the breakdown for choosing the mode and hoping you make money. 1335 movies test through this model. </br>

- True Positives:  801 </br>
- False Positives: 534 </br>
- True Negatives:    0 </br>
- False Negatives:   0 </br></br>

- Accuracy:  60%
- Precision: 60%
- Recall:    100% </br></br>

An Example Run Down of the Confusion Matrix, 1335 movies tested: </br>
- True Positives:  669 </br>
- False Positives: 436 </br>
- True Negatives:  123 </br>
- False Negatives: 107 </br></br>

- Accuracy:  59% </br>
- Precision: 61%</br>
- Recall:    85% </br>


#### Extraneous Tasks That May or May Not Have Been Needed From the Get Go
This model isn't very capable. Secondary steps toward improving the model would be looking at the top actors billed in the credits. I didn't include them because frankly I thought description and director would be enough to accurately predict many. </br> Additionally, experimenting with different weights on the columns could tease more out. I toyed with having director in the text column 5 times, for instance. Overall, I would not pass this model onto the BigShotStudioExec for fear of my job. I would tell them that I could do better than throwing darts at the board and it would cost them less money in underviewed flicks.
