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


<p align="center"><img src="/imgs/action_encoder.png" height="200" width="700"></p>

blablabla


<img src="/imgs/player_net.png" height="550" width="900">


