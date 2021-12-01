import random
import galeshapley

# Number of agents
n = 1000
# Number of attributes per agent
m = 10
# Agent id counter
id_counter = 0

class Agent:
    def __init__(self, id_num):
        # each agent has random attributes that can take each take on 5 values
        self.attributes = [random.choice(['A', 'B', 'C', 'D', 'F']) for _ in range(m)]

        # list of 10 lists with A, B, C, D, F in some order
        self.attribute_preferences = [[]]

        # list of 10 randomly generated weights that sum to 1
        self.weights = [random.randint() for _ in range(m)]
        total = sum(self.weights)
        self.weights = [x / total for x in self.weights]

        self.id = id_num
        self.elo = 400

    

def main():
    # Simulation runs here
    # Generate 2n agents
    agents = [Agent(i) for i in range(2 * n)]
    

main()


