# CS 136 Final Project
# Charumathi Badrinath, Catherine Cui, Lawrence Zhang
# Note: Python 3.6+ required

import random
import statistics
import galeshapley as gs
from collections import defaultdict
import numpy
from math import exp

# number of trials to run
trials = 1

# number of agents on each side
n = 1000
# number of attributes per agent
m = 10
# number of initial random Elo rounds
x = 30
# number of following systematic Elo rounds
y = 70

# starting elo for agents
starting_elo = 400

# options for each attribute for each agent
attribute_values = ['A', 'B', 'C', 'D', 'F']
# number of options for each attribute
attribute_options = len(attribute_values)

# roughly the global proportion of agents who have A, ..., F 
# respectively as their most preferred setting of an attribute
global_preferences = [0.9, 0.09, 0.009, 0.0009, 0.0001]


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
        self.elo = starting_elo

    def generate_attribute_preferences(self):
        for _ in range(m):
            # generate attribute preference ordering correlated with global preferences
            attribute_preferences_list = list(numpy.random.choice(attribute_values, size=attribute_options, replace=False, p=global_preferences))
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
    # maps agent id (woman) to agent id (man) that is a match
    reverse_matching = dict(zip(matching.values(),matching.keys()))
    # check all pairs of men and women
    for i in range(n):
        for j in range(n, 2 * n):
            # if this man and woman aren't already matched together
            if matching[i] != j:
                # if they both prefer each other more than their current matches
                if true_preferences[i].index(matching[i]) > true_preferences[i].index(j):
                    if true_preferences[j].index(reverse_matching[j]) > true_preferences[j].index(i):
                        blocking_pairs += 1

    return blocking_pairs


# Input: list of agents, a dict that maps agent id (man) to agent id (woman) that is a match,
# a dict mapping agent id to the true full preference profile of that agent,
# as well as a cutoff elo that agents must satisfy in order to be included
# Output: average preference choice (1-indexed) that each man/woman got
def happiness(agents, matching, true_preferences, cutoff, above):
    reverse_matching = dict(zip(matching.values(),matching.keys()))

    men, women = [], []
    for i in range(n):
        if (above and agents[i].elo >= cutoff) or (not above and agents[i].elo < cutoff):
            men.append(true_preferences[i].index(matching[i]) + 1)
    for j in range(n, 2 * n):
        if (above and agents[j].elo >= cutoff) or (not above and agents[j].elo < cutoff):
            women.append(true_preferences[j].index(reverse_matching[j]) + 1)

    return statistics.mean(men), statistics.mean(women)


# Output: random matching (dict mapping agent id (man) to agent id (woman))
def generate_random_matching():
    men, women = list(range(n)), list(range(n, 2 * n))
    random.shuffle(men)
    random.shuffle(women)
    return {man: woman for man, woman in zip(men, women)}


def main():
    # Seed the number generators
    random.seed(783387355)
    numpy.random.seed(783387355)

    # Store results of each trial (men/women happiness and number of blocking pairs)
    num_blocking_pairs = [[], [], []] 
    men_happiness = [[], [], []] 
    women_happiness = [[], [], []] 
    # happiness of agents at or above a certain elo cutoff
    attractive_men_happiness = [[], [], []]
    attractive_women_happiness = [[], [], []]
     # happiness of agents below a certain elo cutoff
    unattractive_men_happiness = [[], [], []]
    unattractive_women_happiness = [[], [], []]

    # Helper function to store results of each trial
    def store_results(i, agents, matching, true_preferences, cutoff, above):
        if cutoff == float('-inf'):
            men, women = happiness(agents, matching, true_preferences, float('-inf'), True)
            num_blocking_pairs[i].append(blocking_pairs(matching, true_preferences))
            men_happiness[i].append(men)
            women_happiness[i].append(women)
        elif above:
            men, women = happiness(agents, matching, true_preferences, cutoff, True)
            attractive_men_happiness[i].append(men)
            attractive_women_happiness[i].append(women)
        else:
            men, women = happiness(agents, matching, true_preferences, cutoff, False)
            unattractive_men_happiness[i].append(men)
            unattractive_women_happiness[i].append(women)


    for _ in range(trials):
        # Generate 2n agents: [0, n-1] boys, [n, 2n-1] girls
        agents = [Agent(i) for i in range(2 * n)]
        # Generate each agent's score for every other agent
        generate_scores(agents)

        # 0: Generate agents' true full preference profiles and true matching
        true_preference_profiles = generate_true_preference_profile(agents)
        true_gs = gs.GaleShapley(true_preference_profiles)
        true_gs.match()
        store_results(0, agents, true_gs.matches, true_preference_profiles, float('-inf'), True)

        # 1: Generate random matching
        random_matching = generate_random_matching()
        store_results(1, agents, random_matching, true_preference_profiles, float('-inf'), True)

        # 2: Generate standard Elo estimated preference profiles and matching
        elo_preference_profiles = elo(agents)
        elo_gs = gs.GaleShapley(elo_preference_profiles)
        elo_gs.match()
        store_results(2, agents, elo_gs.matches, true_preference_profiles, float('-inf'), True)

        elos = numpy.array([agent.elo for agent in agents])
        above_cutoff = numpy.percentile(elos, 90)
        below_cutoff = numpy.percentile(elos, 10)

        store_results(0, agents, true_gs.matches, true_preference_profiles, above_cutoff, True)
        store_results(1, agents, random_matching, true_preference_profiles, above_cutoff, True)
        store_results(2, agents, elo_gs.matches, true_preference_profiles, above_cutoff, True)

        store_results(0, agents, true_gs.matches, true_preference_profiles, below_cutoff, False)
        store_results(1, agents, random_matching, true_preference_profiles, below_cutoff, False)
        store_results(2, agents, elo_gs.matches, true_preference_profiles, below_cutoff, False)

    # Compute statistics
    print(num_blocking_pairs)
    print(men_happiness)
    print(women_happiness)
    print(attractive_men_happiness)
    print(attractive_women_happiness)
    print(unattractive_men_happiness)
    print(unattractive_women_happiness)


if __name__ == '__main__':
    main()
