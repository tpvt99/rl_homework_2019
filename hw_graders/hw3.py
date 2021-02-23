from autograder import Autograder

q_i = {'name': None, 'prefix': None, 'key': None, 'thresh': None, 'weight': None}

q1 = {'name': 'Q1', 'prefix': ['hw3', 'q1'],'key': 'Train_AverageReturn', 'thresh': 1500, 'weight': 0.2}
q2a = {'name': 'Q2a', 'prefix': ['hw3', 'q2_dqn'], 'key': 'Train_AverageReturn', 'thresh': 115, 'weight': 0.15}
q2b = {'name': 'Q2b', 'prefix': ['hw3', 'q2_doubledqn'], 'key': 'Train_AverageReturn', 'thresh': 135, 'weight': 0.15}
q3 = {'name': 'Q3', 'prefix': ['hw3', 'q3'], 'key': 'Train_AverageReturn', 'thresh': 0, 'weight': 0.1}
q4 = {'name': 'Q4', 'prefix': ['hw3', 'q4'], 'key': 'Train_AverageReturn', 'thresh': 200, 'weight': 0.1}
q5a = {'name': 'Q5a', 'prefix': ['hw3', 'q5', 'InvertedPendulum'], 'key': 'Train_AverageReturn', 'thresh': 1000, 'weight': 0.15}
q5b = {'name': 'Q5b', 'prefix': ['hw3', 'q5', 'HalfCheetah'], 'key': 'Train_AverageReturn', 'thresh': 140, 'weight': 0.15}

HW3_scheme = [q1, q2a, q2b, q3, q4, q5a, q5b]

if __name__ == '__main__':
    autograder = Autograder(HW3_scheme, make_visible=False)
    autograder.process_submission()
