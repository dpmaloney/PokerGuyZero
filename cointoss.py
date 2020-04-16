import random


class coinToss():

    def __init__(self):
        self.coin = random.randint(0,1) #1 means heads, 0 means tails
        self.state = 0 #0 = sell/play phase 1 = p2 guess phase

    def action(self, action):
        #actions
        #state == 0 (buy/sell state)
        #action = 0 means sell
        #action =1 means play
        if self.state == 0:
            if action == 0:
                if self.coin == 0:
                    return .5 #lucky
                else:
                    return -.5 # unlucky
            else:
                state = 1
                return 0
        #state 1 (player 2 guesses)
        #action -1 means forefeit
        #action 0 means heads, 1 means tails

        elif self.state == 1:
            if action == -1:
                return 1
            elif action == self.coin:
                return -1
            else:
                return 1



