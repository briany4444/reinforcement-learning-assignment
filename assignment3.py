# COMP3106 A3   Brian Jean and Rayhaan

import numpy
import sys

class td_qlearning:

  # definitions from pdf
  alpha = 0.2
  gamma = 0.9

  # the 2D dictionary that stores the q values for each action-state pair
  # the states are the keys, and the values are dictionaries for action-qvalue pairs
  qFunc = {}

  # trial_filepath is the path to a file containing a trial through state space
  def __init__(self, trial_filepath):

    STATE = 0
    ACTION = 1
    possibleSquares = ["X", "Y", "W", "Z"]

    # creates a dictionary for every state, where i is the mouse and j is the cat's state
    for i in range(len(possibleSquares)):
        for j in range(len(possibleSquares)):

            # the stringified state (which will be a new key in the qFunc dictionary)
            state = possibleSquares[i] + possibleSquares[j]

            # ensures faster runtime
            if state in self.qFunc:
                continue

            # initializes a dictionary for each state. Keys are the actions, values are the q-values
            stateDictionary = {}

            # given the mouse current state, adds all its possible actions to a list
            possibleActions = ["N"]     # every state gives the mouse the option to not move
            if possibleSquares[i] == "W":
                possibleActions.append("R")
                possibleActions.append("D")
            elif possibleSquares[i] == "X":
                possibleActions.append("L")
                possibleActions.append("D")
            elif possibleSquares[i] == "Y":
                possibleActions.append("R")
                possibleActions.append("U")
            else:
                possibleActions.append("L")
                possibleActions.append("U")

            # the dictionary for this state is initialized to all 0s for every
            # action possible
            for k in range(len(possibleActions)):
                stateDictionary[possibleActions[k]] = 0

            # adds the new key value pair to the main dictionary
            self.qFunc[state] = stateDictionary

    # reads csv
    file = open(trial_filepath, 'r')
    lines = file.readlines()

    # iterates through the trials and updates q funcs
    for i in range(len(lines)):
        # gets the current state and action from row
        strippedLine = lines[i].strip('\n')
        currState = strippedLine.split(",")[STATE]
        currAction = strippedLine.split(",")[ACTION]

        # gets current Q(S, A)
        currValue = self.qvalue(currState, currAction)

        # obtains the action for the subsequent state that maximizes that state's value
        # doesn't update the last row since no states follows
        if i + 1 < len(lines):
            # finds the next state (aka the following row)
            strippedNextLine = lines[i+1].strip()
            nextState = strippedNextLine.split(",")[STATE]

            # obtains the maximum value in the inner state dictionary given
            # in other words, the action that maximizes the q value (policy)
            nextValue = self.qvalue(nextState, self.policy(nextState))

            # reward is -1 if both the cat and mouse in same block, otherwise 1
            reward = 1
            if currState[0] == currState[1]:
                reward = -1

            # updates the q value using temporal difference learning formula
            self.qFunc[currState][currAction]=currValue+self.alpha*(reward+(self.gamma*nextValue)-currValue)

  # state is a string representation of a state
  # action is a string representation of an action
  def qvalue(self, state, action):
    q = "error"
    if (state in self.qFunc.keys()) and (action in self.qFunc[state].keys()):
        q = self.qFunc[state][action]
    return q

  # state is a string representation of a state
  def policy(self, state):
    a = "error"
    if (state in self.qFunc.keys()):
        # Return the optimal action under the learned policy
        a = max(self.qFunc[state], key=self.qFunc[state].get)
    return a

hm = td_qlearning("Examples/Example1/trial.csv")
print(hm.qvalue("ZX", "N"))
