import torch
import ChessDove
import random
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import time

all_moves = [((i, j), (k, l)) for i in range(8) for j in range(8) for k in range(8) for l in range(8)]
whi = 0
bla = 0
draw = 0
castled = 0

class BlackChessDoveAI(nn.Module):
    def __init__(self,n_outputs):
        super(BlackChessDoveAI, self).__init__()
        self.memory = []
        self.gamma = 0.99

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.MaxPool2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_outputs),
            nn.Softmax(dim=-1)
        )


        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)


    def forward(self, x):
        return self.network(x)
    
    def remember(self, state, action, reward):
        self.memory.append((state, action, reward))

    def train_policy_network(self):
        states, actions, rewards = zip(*self.memory)
        states = torch.stack(states).float().unsqueeze(1)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)

        discounted_rewards = self.compute_returns(rewards)

        self.optimizer.zero_grad()
        action_probs = []
        for state in states:
            prob = self.network(state.unsqueeze(0))
            action_probs.append(prob)
        loss = self.policy_loss(action_probs, actions, discounted_rewards)
        loss.backward()
        self.optimizer.step()

    def policy_loss(self, probs, actions, rewards):
        selected_log_probs = []
        for i in range(len(actions)):
            selected_log_probs.append(torch.log(probs[i][0][all_moves.index(action)]))
        return -torch.mean(torch.stack(selected_log_probs) * rewards)

    def compute_returns(self, rewards):
        R = 0
        returns = []
        for r in rewards:
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns)

    def clear_memory(self):
        self.memory = []


class WhiteChessDoveAI(nn.Module):
    def __init__(self, n_outputs):
        super(WhiteChessDoveAI, self).__init__()
        self.memory = []
        self.gamma = 0.99

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.MaxPool2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_outputs),
            nn.Softmax(dim=-1)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)


    def forward(self, x):
        return self.network(x)
    
    def remember(self, state, action, reward):
        self.memory.append((state, action, reward))

    def train_policy_network(self):
        states, actions, rewards = zip(*self.memory)
        states = torch.stack(states).float().unsqueeze(1)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)

        discounted_rewards = self.compute_returns(rewards)

        self.optimizer.zero_grad()
        action_probs = []
        for state in states:
            prob = self.network(state.unsqueeze(0))
            action_probs.append(prob)
        loss = self.policy_loss(action_probs, actions, discounted_rewards)
        loss.backward()
        self.optimizer.step()

    def policy_loss(self, probs, actions, rewards):
        selected_log_probs = []
        for i in range(len(actions)):
            selected_log_probs.append(torch.log(probs[i][0][all_moves.index(action)]))
        return -torch.mean(torch.stack(selected_log_probs) * rewards)

    def compute_returns(self, rewards):
        R = 0
        returns = []
        for r in rewards:
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns)

    def clear_memory(self):
        self.memory = []

n_inputs = 8 * 8
gamma = 0.99
n_outputs = 4096


def promotion(position, turn, promotionto):
    if turn == 1:
        pawns = torch.nonzero((position.abs() == 1) & (position > 0), as_tuple=False)
        for pawn in pawns:
            if pawn[0].item() == 0:
                position[pawn[0]][pawn[1]] = promotionto
                return position
            else:
                continue
        return position
    else:
        pawns = torch.nonzero((position.abs() == 1) & (position < 0), as_tuple=False)
        for pawn in pawns:
            if pawn[0].item() == 7:
                position[pawn[0]][pawn[1]] = promotionto * turn
                return position
            else:
                continue
        return position

def get_move_index(move):
    return all_moves.index(move)


def select_legal_action(network_output, position, legal_moves, gamehis, turn, last_move, epsilon=0.15):
    if turn == 1:
        for move in legal_moves:
            if move == ((7, 4), (7, 6)) and position[7][4] == 6:
                oppolegal = ChessDove.get_legalmoves(newpos, ChessDove.turn * -1, last_move, gamehis)
                for oppomovetype in oppolegal:
                    for oppomove in oppomovetype:
                        if oppomove[1] == (7, 5):
                            if move in legal_moves:
                                legal_moves.remove(move)
                            else:
                                break
            elif move == ((7, 4), (7, 2)) and position[7][4] == 6:
                oppolegal = ChessDove.get_legalmoves(newpos, ChessDove.turn * -1, last_move, gamehis)
                for oppomovetype in oppolegal:
                    for oppomove in oppomovetype:
                        if oppomove[1] == (7, 3):
                            if move in legal_moves:
                                legal_moves.remove(move)
                            else:
                                break
    else:
        for move in legal_moves:
            if move == ((0, 4), (0, 6)) and position[0][4] == -6:
                oppolegal = ChessDove.get_legalmoves(newpos, ChessDove.turn * -1, last_move, gamehis)
                for oppomovetype in oppolegal:
                    for oppomove in oppomovetype:
                        if oppomove[1] == (0, 5):
                            if move in legal_moves:
                                legal_moves.remove(move)
                            else:
                                break
            elif move == ((0, 4), (0, 2)) and position[0][4] == -6:
                oppolegal = ChessDove.get_legalmoves(newpos, ChessDove.turn * -1, last_move, gamehis)
                for oppomovetype in oppolegal:
                    for oppomove in oppomovetype:
                        if oppomove[1] == (0, 3):
                            if move in legal_moves:
                                legal_moves.remove(move)
                            else:
                                break


    if random.random() <= epsilon:
        return random.choice(legal_moves)
    legal_move_indices = [get_move_index(move) for move in legal_moves]
    mask = torch.zeros_like(network_output)
    mask[0, legal_move_indices] = 1
    masked_output = network_output * mask
    sum_probs = masked_output.sum()
    if torch.all(torch.isnan(masked_output)):
        return random.choice(legal_moves)
    elif sum_probs <= 0:
        return random.choice(legal_moves)
    else:
        action = torch.multinomial(masked_output, 1)
        return all_moves[action.item()]



times = 100
epochs = 100
for times in range(times):
    whitenet = WhiteChessDoveAI(n_outputs)
    whitenet.load_state_dict(torch.load('WhiteChessDove.pth'))
    blacknet = BlackChessDoveAI(n_outputs)
    blacknet.load_state_dict(torch.load('BlackChessDove.pth'))
    for epoch in range(epochs):
        proto = 5
        pos_his = []
        gamehis = []
        last_move = ((0,0),(0,0))
        newpos = ChessDove.position
        pos_his.append(newpos)
        legalmove = ChessDove.get_final_legal(newpos, ChessDove.turn, last_move, gamehis)
        while ChessDove.check_con(newpos, ChessDove.turn, legalmove, last_move, pos_his, gamehis):
            kings = torch.nonzero(newpos.abs() == 6, as_tuple=False)
            legalmove = ChessDove.get_final_legal(newpos, ChessDove.turn, last_move, gamehis)
            if len(legalmove) == 0:
                ChessDove.check_con(newpos, ChessDove.turn, legalmove, last_move, pos_his, gamehis)
                break
            action_probs = whitenet(torch.tensor(newpos.unsqueeze(0)).float().unsqueeze(0)) ##
            action = select_legal_action(action_probs, newpos, legalmove, gamehis, ChessDove.turn, last_move)
            if action == ((7, 4), (7, 6)) and ChessDove.checkcastright(ChessDove.turn, gamehis) or action == ((7, 4), (7, 2)) and ChessDove.checkcastleft(ChessDove.turn, gamehis):
                newpos = ChessDove.Castle_move(newpos, action)
                castled += 1
            else:
                newpos = ChessDove.make_move(newpos, action)
            
            newpos = promotion(newpos, ChessDove.turn, proto)
            material_equ = ChessDove.material_equ(newpos)
            print(newpos)
            whitelaspos = newpos
            pos_his.append(newpos)
            last_move = action
            gamehis.append(last_move)
            matdif = ChessDove.material_diff(pos_his)
            if matdif == 0:
                blacknet.remember(newpos, action, -3)
            else:
                blacknet.remember(newpos, action, matdif)
            
            print("\n\n")
            
            print(f"epoch: {epoch + 1}")
            print(f"time: {times + 1}")
            ChessDove.changeturn()
            legalmove = ChessDove.get_final_legal(newpos, ChessDove.turn, last_move, gamehis)
            if len(legalmove) == 0:
                ChessDove.check_con(newpos, ChessDove.turn, legalmove, last_move, pos_his, gamehis)
                break

            blackaction_probs = blacknet(torch.tensor(newpos.unsqueeze(0)).float().unsqueeze(0))
            blackaction = select_legal_action(blackaction_probs, newpos, legalmove, gamehis, ChessDove.turn, last_move)
            if blackaction == ((0, 4), (0, 6)) and ChessDove.checkcastright(ChessDove.turn, gamehis) or blackaction == ((0, 4), (0, 2)) and ChessDove.checkcastleft(ChessDove.turn, gamehis):
                newpos = ChessDove.Castle_move(newpos, blackaction)
                castled += 1
            else:
                newpos = ChessDove.make_move(newpos, blackaction)
            newpos = promotion(newpos, ChessDove.turn, proto)
            blalaspos = newpos
            pos_his.append(newpos)
            print(newpos)
            material_equ = ChessDove.material_equ(newpos)
            blacklast_move = blackaction
            matdif = ChessDove.material_diff(pos_his)
            if matdif == 0:
                blacknet.remember(newpos, blackaction, -3)
            else:
                blacknet.remember(newpos, blackaction, matdif * -1)
            print("\n\n")
            print(f"epoch: {epoch + 1}")
            print(f"time: {times + 1}")
            ChessDove.changeturn()




        if ChessDove.winner == 1:
            whitenet.remember(whitelaspos, last_move, 200000)
            blacknet.remember(blalaspos, blacklast_move, -200000)
            whi = whi + 1
        elif ChessDove.winner == -1:
            whitenet.remember(whitelaspos, last_move, -200000)
            blacknet.remember(blalaspos, blacklast_move, 200000)
            bla = bla + 1
        else:
            whitenet.remember(whitelaspos, last_move, -500)
            blacknet.remember(blalaspos, blacklast_move, -500)
            draw = draw + 1

        whitenet.train_policy_network()
        blacknet.train_policy_network()
        print("Train done")
        whitenet.clear_memory()
        blacknet.clear_memory()
    torch.save(whitenet.state_dict(), 'WhiteChessDove.pth')
    torch.save(blacknet.state_dict(), 'BlackChessDove.pth')

print(f"Summary: White won {whi} times, black won {bla} times, drew {draw} times, castled {castled} time(s).")


