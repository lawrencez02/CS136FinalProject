class GaleShapley:
    # Input: dict that maps agent ids to full preference profiles, which are lists of agent ids
    def __init__(self, preferences):
        # number of each gender
        self.num = len(preferences) // 2

        # create separate preference dictionaries for men and women 
        self.men = dict(list(preferences.items())[:self.num])
        self.women = dict(list(preferences.items())[self.num:])

        # create a list of pointers for men (which 0-indexed preference woman they are about
        # to propose to) and women (which 0-indexed preference man they are holding onto)
        self.menp = [0] * self.num
        self.womenp = [self.num] * self.num

        # create a list for unmatched men
        self.unmatchedmen = list(self.men.keys())

        # dict that maps agent id (man) to agent id (woman) that is a match
        self.matches = {}

    # Runs boy-proposing Gale-Shapley and generates matches
    def match(self):
        while self.unmatchedmen:
            for man in self.unmatchedmen:
                # woman that man is about to propose to 
                woman = self.men[man][self.menp[man]]
                self.accept(man, woman)
                
        for man in self.men.keys():
            self.matches[man] = self.men[man][self.menp[man]-1]

    # Determines what happens when man proposes to woman       
    def accept(self, man, woman):
        new = self.women[woman].index(man)
        previous = self.womenp[woman-self.num]

        if new < previous:
            if previous < self.num:
                self.unmatchedmen.append(self.women[woman][previous])
            self.womenp[woman-self.num] = new
            self.unmatchedmen.remove(man)
        
        self.menp[man] += 1

