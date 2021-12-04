# CS 136 Final Project
# Charumathi Badrinath, Catherine Cui, Lawrence Zhang
# Note: Python 3.6+ required

import random
import galeshapley as gs
from collections import defaultdict
from numpy import random as rand
from math import exp

# number of agents on each side
n = 1000
# number of attributes per agent
m = 10
# number of initial random Elo rounds
x = 30
# number of following systematic Elo rounds
y = 70

# options for each attribute for each agent
attribute_values = ['A', 'B', 'C', 'D', 'F']
# number of options for each attribute
attribute_options = len(attribute_values)

# roughly the global proportion of agents who have A, ..., F 
# respectively as their most preferred setting of an attribute
global_preferences = [5/15, 4/15, 3/15, 2/15, 1/15]


class Agent:
    def __init__(self, id_num):
        # unique id of agent
        self.id = id_num

        # each agent has random attributes that can take each take on 5 values
        self.attributes = [random.choice(attribute_values) for _ in range(m)]

        # list of 10 dicts mapping A, ..., F to their respective point values
        # for every attribute, A most likely to be the most preferred globally, and F is the least likely
        self.attribute_preferences = []
        self.generate_attribute_preferences()

        # list of 10 randomly generated weights that sum to 1
        self.weights = [random.random() for _ in range(m)]
        self.weights = [x / sum(self.weights) for x in self.weights]

        # dict mapping agent id to this agent's score for that agent - populated in generate_scores
        self.scores = {}
        # agent's own score for him/herself
        self.score = 0

        # starting Elo of agent
        self.elo = 400

    def generate_attribute_preferences(self):
        for _ in range(m):
            # generate attribute preference ordering correlated with global preferences
            attribute_preferences_list = list(rand.choice(attribute_values, size=attribute_options, replace=False, p=global_preferences))
            attribute_preferences_dict = {attribute_preferences_list[i]: attribute_options - i for i in range(attribute_options)}
            self.attribute_preferences.append(attribute_preferences_dict)


def generate_scores(agents):
    for i in range(2 * n):
        other_side = range(n, 2 * n) if i < n else range(n)
        for j in other_side:
            # Generate weighted sum for agent i's score of agent j
            score = sum([agents[i].weights[k] * agents[i].attribute_preferences[k][agents[j].attributes[k]] for k in range(m)])
            agents[i].scores[agents[j].id] = score

        # Generate agent i's score of him/herself
        agents[i].score = sum([agents[i].weights[k] * agents[i].attribute_preferences[k][agents[i].attributes[k]] for k in range(m)])


# Input: list of agents
# Output: dict that maps agent ids to true full preference profiles, which are lists of agent ids
def generate_true_preference_profile(agents):
    preference_profile = {}
    for agent in agents:
        agent_ids_sorted_by_score = [agent_id for agent_id, _ in sorted(agent.scores.items(), key=lambda item: item[1], reverse=True)]
        preference_profile[agent.id] = agent_ids_sorted_by_score
    return preference_profile


# Input: list of agents
# Output: dict that maps agent ids to estimated full preference profiles, which are lists of agent ids
def elo(agents):
    # ensure each agent paired with different agent every round
    seen = defaultdict(set)

    # x initial random Elo rounds and y systematic Elo rounds
    for curr_round in range(x + y):
        # stores results of round (maps agent id to (win/loss, other agent's Elo))
        results = {}

        # every boy has his profile shown to a randomly chosen girl, and vice versa
        for curr_side, other_side in [(range(n), range(n, 2 * n)), (range(n, 2 * n), range(n))]:
            for i in curr_side:
                if curr_round < x:
                    # randomly choose agent j that sees agent i's profile if in initial x random Elo rounds
                    j = random.choice(other_side)
                    while (j in seen[i]):
                        j = random.choice(other_side)
                else:
                    # choose the agent j that is closest in Elo to agent i if in y systematic Elo rounds
                    j = min([j for j in other_side if j not in seen[i]], key=lambda j: abs(agents[i].elo - agents[j].elo))

                seen[i].add(j)

                if random.random() <= 1 / (1 + exp(-2 * (agents[j].scores[i] - agents[j].score))):
                    # agent j swipes right, so agent i "wins" against agent j
                    results[i] = (True, agents[j].elo)
                else:
                    # agent j swipes left, so agent i "loses" against agent j
                    results[i] = (False, agents[j].elo)
        
        # update Elos based on results of round
        for i in range(2 * n):
            expected_score_i = 1 / (1 + pow(10, ((results[i][1] - agents[i].elo) / 400)))
            agents[i].elo += 32 * (int(results[i][0]) - expected_score_i)

    # generate estimated full preference profiles based on Elos
    estimated_preference_profiles = {}
    for i in range(2 * n):
        possible_matches = list(range(n, 2 * n)) if i < n else list(range(n))
        # agent i prefers the agents who have the closest Elo to him/her
        estimated_preference_profiles[i] = sorted(possible_matches, key=lambda j: abs(agents[i].elo - agents[j].elo))

    return estimated_preference_profiles


# Input: a dict that maps agent id (man) to agent id (woman) that is a match,
# and a dict mapping agent id to the true full preference profile of that agent
# Output: number of blocking pairs in the given matching
def blocking_pairs(matching, true_preferences):
    blocking_pairs = 0 
    # check all pairs of men and women
    for i in range(n):
        for j in range(n, 2 * n):
            # if this man and woman aren't already matched together
            if matching[i] != j:
                man_curr_match = matching[i]
                woman_curr_match = next((k for k, v in matching.items() if v == j), None)
                # if they both prefer each other more than their current matches
                if true_preferences[i].index(man_curr_match) > true_preferences[i].index(j):
                    if true_preferences[j].index(woman_curr_match) > true_preferences[j].index(i):
                        blocking_pairs += 1

    return blocking_pairs


def main():
    # Seed the number generators
    random.seed(783387355)
    rand.seed(783387355)

    # Generate 2n agents: [0, n-1] boys, [n, 2n-1] girls
    agents = [Agent(i) for i in range(2 * n)]
    # Generate each agent's score for every other agent
    generate_scores(agents)
    # Generate agents' true full preference profile 
    true_preference_profiles = generate_true_preference_profile(agents)

    # Run Elo-based rounds to generate estimated full preference profiles
    estimated_preference_profiles = elo(agents)

    # Compare results of boy-proposing GaleShapley on true preference profiles vs. estimated preference profiles
    true_gs = gs.GaleShapley(true_preference_profiles)
    true_gs.match()

    estimated_gs = gs.GaleShapley(estimated_preference_profiles)
    estimated_gs.match()



if __name__ == '__main__':
    main()
