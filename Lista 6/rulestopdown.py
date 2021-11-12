from rules import Rules

class RulesTopDown(Rules):
    def __init__(self, csv_file, debugging):
        Rules.__init__(self, csv_file, debugging)

    def _learn_rule(self, data, class_value):
        '''
        Aprende uma regra
        '''

        r1 = [class_value]

        while true:
            r2 = self._expand_rule(data, class_value, r1)
            if (r1 == r2):
                break

            r1 = r2

        return r1




    def _expand_rule(self, data, class_value, rule):
        X = set(self.att_list)
        cond = []
        best = Rules._aval(class_value,data,rule)
        for A in X:
            V =