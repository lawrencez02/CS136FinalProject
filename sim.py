import random
import galeshapley
from collections import defaultdict
from math import exp, pow, abs

# number of agents on each side
n = 1000
# number of attributes per agent
m = 10

# number of initial random Elo rounds
x = 30
# number of following systematic Elo rounds
y = 70


class Agent:
    def __init__(self, id_num):
        # unique id of agent
        self.id = id_num

        # each agent has random attributes that can take each take on 5 values
        self.attributes = [random.choice(['A', 'B', 'C', 'D', 'F']) for _ in range(m)]

        # list of 10 dicts mapping 'A', 'B', 'C', 'D', 'F' to their respective point values
        # TODO: Charu
        self.attribute_preferences = [[]]

        # list of 10 randomly generated weights that sum to 1
        self.weights = [random.random() for _ in range(m)]
        self.weights = [x / sum(self.weights) for x in self.weights]

        # dict mapping agent id to this agent's score for that agent
        # TODO: Charu
        self.scores = {}
        # agent's own score for him/herself
        self.score = sum([self.weights[i] * self.attribute_preferences[i][self.attributes[i]] for i in range(m)])

        # starting Elo of agent
        self.elo = 400


# Input: list of agents
# Output: dict that maps agent ids to true full preference profiles, which are lists of agent ids
def generate_true_preference_profile(agents):
    # TODO: Charu
    pass


# Input: list of agents
# Output: dict that maps agent ids to estimated full preference profiles, which are lists of agent ids
def elo(agents):
    # ensure each agent paired with different agent every round
    seen = defaultdict(set)

    # x initial random Elo rounds and y systematic Elo rounds
    for round in range(x + y):
        # stores results of round (maps agent id to (win/loss, other agent's Elo))
        results = {}

        # every boy has his profile shown to a randomly chosen girl, and vice versa
        for curr_side, other_side in [(range(n), range(n, 2 * n - 1)), (range(n, 2 * n - 1), range(n))]:
            for i in curr_side:
                if round < x:
                    # randomly choose agent j that sees agent i's profile if in initial x random Elo rounds
                    j = random.choice(other_side)
                    while (j in seen[i]):
                        j = random.choice(other_side)
                else:
                    # choose the agent j that is closest in elo to agent i
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


def main():
    # Generate 2n agents: [0, n-1] boys, [n, 2n-1] girls
    agents = [Agent(i) for i in range(2 * n)]

    # Generate agents' true full preference profile 
    true_preference_profiles = generate_true_preference_profile(agents)
    # Run boy-proposing Gale-Shapley
    true_matches = galeshapley.GaleShapley(true_preference_profiles)

    # Run Elo-based rounds to generate estimated full preference profiles
    estimated_preference_profiles = elo(agents)
    # Run boy-proposing Gale-Shapley
    estimated_matches = galeshapley.GaleShapley(estimated_preference_profiles)

    # TODO: compare true_matches and estimated_matches


if __name__ == '__main__':
    main()
