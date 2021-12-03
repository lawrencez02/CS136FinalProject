import random
import galeshapley

# number of agents
n = 1000
# number of attributes per agent
m = 10

# number of initial random elo rounds
x = 20
# number of following systematic elo rounds
y = 100

class Agent:
    def __init__(self, id_num):
        # each agent has random attributes that can take each take on 5 values
        self.attributes = [random.choice(['A', 'B', 'C', 'D', 'F']) for _ in range(m)]

        # list of 10 lists with A, B, C, D, F in some order
        self.attribute_preferences = [[]]

        # list of 10 randomly generated weights that sum to 1
        self.weights = [random.uniform(0, 1) for _ in range(m)]
        self.weights = [x / sum(self.weights) for x in self.weights]

        # unique id of agent
        self.id = id_num
        # starting elo of agent
        self.elo = 400

# Input: list of agents
# Output: dict that maps agent ids to true full preference profiles, which are lists of agent ids
def generate_true_preference_profile(agents):
    # TODO: Charu
    pass

# Input: list of agents
# Output: dict that maps agent ids to estimated full preference profiles, which are lists of agent ids
def elo(agents):
    # TODO: Lawrence
    pass


def main():
    # Generate 2n agents: [0, n-1] boys, [n, 2n-1] girls
    agents = [Agent(i) for i in range(2 * n)]

    # Generate agents' true full preference profile 
    # and call Gale-Shapley algorithm
    true_preference_profiles = generate_true_preference_profile(agents)
    true_matches = galeshapley.GaleShapley(true_preference_profiles)

    # Run Elo-based rounds to generate estimated full preference profiles
    estimated_preference_profiles = elo(agents)
    estimated_matches = galeshapley.GaleShapley(estimated_preference_profiles)

    # TODO: compare true_matches and estimated_matches


if __name__ == '__main__':
    main()
