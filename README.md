# WhatsInnaName
### Or In the Description, Director, and Genre?
Using Naive Bayes to predict movie profitability given only a few text columns



![moviessuck](https://github.com/JacobGraphs/WhatsInnaName/blob/master/img/movies_suck.jpg)
</br></br>
Some movies suck. Some movies don't. A decent amount of them make money. End of story.
</br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br>

No, but there is more. Just not much more.

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
Woody Allen directed too many movies to count, lost money on many of them, and this tracking of Rotten Tomatoes Scores seems to illustrate the presumed issue here</br></br>
Billy Bob Thornton directed a movie about <b>horses</b>, and they spent 60 million dollars making that thing, losing many millions. But at the same time, he wrote and directed Sling Blade and made 24x the budget back on gross returns.</br>
</br></br></br>
I thought it may have a positive ROI for studios to invest some time in a machine learning algorithm to help determine if movies should be made or not. In order to do that, I'm going to make the assumption that BigShotStudioExec wants their movie to at least earn back the production budget. I don't think this is a lofty ask of your creatives: Produce something that costs less than what it collects from people. Instead of spending all day trying to decide if <b>*Fast and the Furious 14: Valentines Day*</b> or <b>*Hobbs & Shaw 2: Maybe Gerard Butler's a Villain*</b> will be the last hoorah for the Fast and Furious Series, I figure we let the computer do the work. There's enough movies by now that surely we can figure that out.
</br></br>
To start, I'll go ahead and grab this dataset of movies pulled from IMDB. 
</br>https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset</br>
It's kind of messy and there's a lot of null's.
