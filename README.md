# STAR RRV Hybrid Parliamentary Voting Simulator

Run web-based simulation:
[https://hansioux.github.io/starrrv/starrrv.html](https://hansioux.github.io/starrrv/starrrv.html)

![Simulated STAR + RRV election result](/assets/images/result.png "Simulated STAR + RRV  election result")

## Purpose
Simulate outcomes of STAR voting and Reweighted Range Voting hybrid district plus national party lists voting system and compare that with First Past the Post and MMP.

## Goal
Create a fair, simple and intuitive voting system that encourages multi-party politics and cross party cooperation.  Votes are cast by scoring the candidates and parties based on personal preferences, and would not have to vote only for their second or third preference for tactical voting purposes.  

Options on both tickets will be scored from 0-5 for a unified voting experience.

### STAR Voting for District Seats
STAR (Score Than Automatic Runoff) voting system:
1. Sum up scores for all the candidates.  Two highest scoring candidates are finalists.
2. Elect the finalist with more ballots that scored higher the other finalist.

### Reweighted Range Voting for National Party Lists
This system will allocate seats to maximize total voter satisfaction scores. 

1. Each ballot is given an initial "weight" of 1.
2. The weighted scores on the ballots are summed for each candidate, thus obtaining that candidate's total score.
3. The candidate with the highest total score (who has not already won) is declared a winner. (Note that the first winner in RRV is the same as the winner of a ordinary single-winner Range Voting election using the same ballots.)
4. When a voter "gets her way" in the sense that a candidate she rated highly wins, her ballot weight should be reduced so that she has less influence on later choices of winners. To accomplish that, each ballot is given a new weight = 1/(1+SUM/MAX), where SUM is the sum of the scores that ballot gives to the winners-so-far, while MAX is the maximum allowed score (e.g. MAX=99 if allowed scores are in the range 0 to 99).
5. Repeat steps b-d until the desired number of winners has been chosen.
