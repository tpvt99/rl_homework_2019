from autograder import Autograder

q_i = {'name': None, 'prefix': None, 'key': None, 'thresh': None, 'weight': None}

#q1 = Not included as model error isn't included in event files is it?

q2 = {'name': 'Q2', 'prefix': ['hw4' , 'q2'], 'key': 'Eval_AverageReturn', 'thresh': -75, 'weight': 1/7}

q3a = {'name': 'Q3a', 'prefix': ['hw4' , 'q3', 'obstacles'], 'key': 'Eval_AverageReturn', 'thresh': -60, 'weight': 1/7}

q3b = {'name': 'Q3b', 'prefix': ['hw4' , 'q3', 'reacher'], 'key': 'Eval_AverageReturn', 'thresh': -350, 'weight': 1/7}

q3c = {'name': 'Q3c', 'prefix': ['hw4' , 'q3', 'cheetah'], 'key': 'Eval_AverageReturn', 'thresh': 250, 'weight': 1/7}

q4a = {'name': 'Q5a', 'prefix': ['hw4' , 'q4', 'horizon'], 'key': 'Eval_AverageReturn', 'thresh': -325, 'weight': 1/7}

q4b = {'name': 'Q5b', 'prefix': ['hw4' , 'q4', 'numseq'], 'key': 'Eval_AverageReturn', 'thresh': -325, 'weight': 1/7}

q4c = {'name': 'Q5c', 'prefix': ['hw4' , 'q4', 'ensemble'], 'key': 'Eval_AverageReturn', 'thresh': -325, 'weight': 1/7}

HW4_scheme = [q2, q3a, q3b, q3c, q4a, q4b, q4c]

if __name__ == '__main__':
    autograder = Autograder(HW4_scheme, make_visible=False)
    autograder.process_submission()
