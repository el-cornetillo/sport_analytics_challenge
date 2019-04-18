# sport_analytics_challenge
My solution to the X/PSG Sport Analytics Challenge

# Instructions

We are given a database of all the games that took place during the 1st half-season of the Ligue 1 2016/2017. Each game is described by an XML file that contains all the events relative to the two teams playing, along with timestamps, positions (X, Y), team_ids, player_ids, event_ids, outcomes and about ~300 qualifiers that describe the event with more details.

At test time, the participant is given 15 minutes of a game played during the 2nd half of the season. In the corresponding XML file, teams are anonymised and distinguished with a 1 for the team playing at home and 0 for the other one. Also, all player_ids are anonymised except one player, who is indicated with a 1 (all other players are marked as 0). The goal of the challenge is to predict :

* The player identity
* The team that will play the next event (0 or 1)
* The position of the ball at the next event 

The process is repeated thousands of times, and performances are evaluated with the accuracy for the 2 first tasks and the MSE (mean squared error) for the 3rd task.

# My approach

## Modelize the problem as a NLP problem

The first step is to compute embeddings of the numerous distinct kinds of events that appear in the Opta dataset. More specifically, there are roughly ~40 different types of events, for instance Pass, Goal, Foul, Card, Out, ... each of them being also specified by different qualifiers. That is, a Shot was it a Volley ? a Lob ? a Deflection ? Followed a Dribble ? A Pass was it a Long Ball ? a Cross ? a Corner ? 
Each of these events is also specified by the pitch zone in which it took place (Left zone ? Right Zone ? Back zone ? etc...) and which body part was involved (Left foot ? Right foot ? Head ? etc...).
Finally, each event is also associated to some outcome : for instance, a Pass was it successfull ?
Based on this, we denote each event by a tuple (event type, outcome, qualifiers, body part, pitch zone), and use the MD5 hashing library to map it to a unique "word". We then modelize a game as a kind of NLP problem :

There are two agents speaking (the two teams), each of them will speak with some words (the events) to form some sentences (a sequence of actions without losing the ball). Eventually, the agents will consistently try to interrupt themselves (by recovering the ball) to form a new sentence.

We get the embeddings with the Word2vec framework, in its Skip-Gram fashion. Essentially we train a Dense model to predict weither two events are likely to appear together in a same window of W = 5 events.

We generate pairs :

(event1, event2, 1) for two events that co-appears in a 5-window 
(event1, event2, 0) for two events that do not co-appears in a 5-window

And then train a supervised model as :

<p align="center"><img src="/imgs/network_emb.png" height="200" width="700"></p>

We consider a sentence as a unbroken sequence of ball possession, either by an interruption from the opponent team, or a game event such as a goal, ball out of pitch, formation change, injury, ...



<p align="center"><img src="/imgs/action_encoder.png" height="200" width="700"></p>

blablabla


<img src="/imgs/player_net.png" height="550" width="900">


