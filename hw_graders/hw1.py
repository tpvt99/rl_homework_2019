from autograder import Autograder


q_i = {'name': None, 'prefix': None, 'key': None, 'thresh': None, 'weight': None}

q1 = {'name': 'Q1', 'prefix': 'bc','key': 'Eval_AverageReturn', 'thresh': 2500, 'weight': 0.5}
q2 = {'name': 'Q2', 'prefix': 'dagger', 'key': 'Eval_AverageReturn', 'thresh': 4300, 'weight': 0.5}

HW1_scheme = [q1, q2]

if __name__ == '__main__':
    autograder = Autograder(HW1_scheme, make_visible=False)
    autograder.process_submission()

