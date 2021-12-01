import sim

# returns true if an agent is a boy
def isBoy(agent):
    return agent.id < sim.n

# Input: dict that maps agent ids to full preference profiles, which are lists of agent ids
# Output: dict that maps agent ids to agent ids of their current match
def GaleShapley(preferences):
    #