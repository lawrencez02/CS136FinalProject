class GaleShapley:
    # Input: dict that maps agent ids to full preference profiles, which are lists of agent ids
    def __init__(self, preferences):
        # number of each gender
        self.num = len(preferences) // 2

        # create separate list for men and women 
        self.men = dict(list(preferences.items())[:self.num])
        self.women = dict(list(preferences.items())[self.num:])

        # create a dictionary of pointers for men and women
        self.menp = [0 for man in self.men.keys()]
        self.womenp = [self.num for woman in self.women.keys()]

        # create a list for unmatched men and dict for matches
        self.unmatchedmen = list(self.men.keys())
        
        self.matches = {}

    def match(self):
        while self.unmatchedmen:
            for man in self.unmatchedmen:
                # person they're about to propose to 
                wife = self.men[man][self.menp[man]]
                self.accept(man, wife)
                
        for man in self.men.keys():
            self.matches[man] = self.men[man][self.menp[man]-1]
            
    def accept(self, man, woman):
        new = self.women[woman].index(man)
        previous = self.womenp[woman-self.num]

        if new < previous:
            if previous < self.num:
                self.unmatchedmen.append(self.women[woman][previous])
            self.womenp[woman-self.num] = new
            self.unmatchedmen.remove(man)
        
        self.menp[man] += 1

