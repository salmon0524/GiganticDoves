import torch
turn = 1
winner = 0
kingsaf = 0
kingmoved = 0
blakingmoved = 0
castle_mov_w1 = ((7, 4),(7, 2))
castle_mov_w2 = ((7, 4),(7, 6))
castle_mov_b1 = ((0, 4),(0, 2))
castle_mov_b2 = ((0, 4),(0, 6))

# position = torch.tensor([
#         [-2, -3, -4, -5, -6, -4, -3, -2],
#         [-1, -1, -1, -1, -1, -1, -1, -1],
#         [0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 1, 1],
#         [2, 3, 4, 5, 6, 0, 0, 2]
#         ])

position = torch.tensor([
        [-2, -3, -4, -5, -6, -4, -3, -2],
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [2, 3, 4, 5, 6, 4, 3, 2]
        ])



def pawn_moves(position, turn):
    moves = []
    if turn == 1:
        pawns = torch.nonzero((position.abs() == 1) & (position > 0), as_tuple=False)
        for pawn in pawns:
            x, y = pawn[0].item(), pawn[1].item()
            above = position[x - 1, y]
            if above == 0:
                moves.append(((x, y),(x - 1, y)))
                if x == 6 and position[x - 2, y] == 0:
                    moves.append(((x, y), (x - 2, y)))
        
        return moves
    
    else:
        pawns = torch.nonzero((position.abs() == 1) & (position < 0), as_tuple=False)
        for pawn in pawns:
            x, y = pawn[0].item(), pawn[1].item()
            below = position[x + 1, y]
            if below == 0:
                moves.append(((x, y), (x + 1, y)))
                if x == 1 and position[x + 2, y] == 0:
                    moves.append(((x, y), (x + 2, y)))
        
        return moves
    
def pawn_attack(position,turn):
    moves = []
    if turn == 1:
        pawns = torch.nonzero((position.abs() == 1) & (position > 0), as_tuple=False)
        for pawn in pawns:
            x, y = pawn[0].item(), pawn[1].item()
            if y + 1 < 8:
                right_up = position[x - 1, y + 1]
                if right_up < 0:
                    moves.append(((x, y), (x - 1, y + 1)))
            if y - 1 > -1 :
                left_up = position[x - 1, y - 1]
                if left_up < 0:
                    moves.append(((x, y),(x - 1, y - 1)))
        return moves
    else:
        pawns = torch.nonzero((position.abs() == 1) & (position < 0), as_tuple=False)
        for pawn in pawns:
            x, y = pawn[0].item(), pawn[1].item()
            if y + 1 < 8:
                right_up = position[x + 1, y + 1]
                if right_up > 0:
                    moves.append(((x, y),(x + 1, y + 1)))
            if y - 1 > -1 :
                left_up = position[x + 1, y - 1]
                if left_up > 0:
                    moves.append((((x, y),(x + 1, y - 1))))
        return moves  
def pawn_en_passant(position, turn, last_move):
    moves = []
    if turn == 1:
        pawns = torch.nonzero((position.abs() == 1) & (position > 0), as_tuple=False)
        for pawn in pawns:
            x, y = pawn[0].item(), pawn[1].item()
            if y + 1 < 8:
                if position[last_move[1]] == -1 and last_move[0][0] == 6 and last_move[1][0] == 4 and last_move[1][1] == y + 1:
                    moves.append(((x, y), (x - 1, y + 1)))
            if y - 1 > -1 :
                if position[last_move[1]] == -1 and last_move[0][0] == 6 and last_move[1][0] == 4 and last_move[1][1] == y - 1:
                    moves.append(((x, y),(x - 1, y - 1)))
        return moves
    else:
        pawns = torch.nonzero((position.abs() == 1) & (position < 0), as_tuple=False)
        for pawn in pawns:
            x, y = pawn[0].item(), pawn[1].item()
            if y + 1 < 8:
                if position[last_move[1]] == 1 and last_move[0][0] == 1 and last_move[1][0] == 3 and last_move[1][1] == y + 1:
                    moves.append(((x, y),(x + 1, y + 1)))
            if y - 1 > -1 :
                if position[last_move[1]] == 1 and last_move[0][0] == 1 and last_move[1][0] == 3 and last_move[1][1] == y - 1:
                    moves.append((((x, y),(x + 1, y - 1))))
        return moves
def bishop_moves(position, turn):
    moves = []
    if turn == 1:
        bishops = torch.nonzero((position.abs() == 4) & (position > 0), as_tuple=False)
        for bishop in bishops:
            x, y = bishop[0].item(), bishop[1].item()
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                while 0 <= nx < 8 and 0 <= ny < 8:
                    if position[nx, ny] == 0:
                        moves.append(((x, y), (nx, ny)))
                    elif position[nx, ny] < 0:
                        moves.append(((x, y), (nx, ny)))
                        break
                    else:
                        break
                    nx += dx
                    ny += dy
    else:
        bishops = torch.nonzero((position.abs() == 4) & (position < 0), as_tuple=False)
        for bishop in bishops:
            x, y = bishop[0].item(), bishop[1].item()
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                while 0 <= nx < 8 and 0 <= ny < 8:
                    if position[nx, ny] == 0:
                        moves.append(((x, y), (nx, ny)))
                    elif position[nx, ny] > 0:
                        moves.append(((x, y), (nx, ny)))
                        break
                    else:
                        break
                    nx += dx
                    ny += dy
    return moves
def knight_moves(position, turn):
    moves = []
    if turn == 1:
        knights = torch.nonzero((position.abs() == 3) & (position > 0), as_tuple=False)
        for knight in knights:
            x, y = knight[0].item(), knight[1].item()
            directions = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 8 and 0 <= ny < 8:
                    if position[nx, ny] <= 0:
                        moves.append(((x, y), (nx, ny)))
    else:
        knights = torch.nonzero((position.abs() == 3) & (position < 0), as_tuple=False)
        for knight in knights:
            x, y = knight[0].item(), knight[1].item()
            directions = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 8 and 0 <= ny < 8:
                    if position[nx, ny] >= 0:
                        moves.append(((x, y), (nx, ny)))
    return moves
def rook_moves(position, turn):
    moves = []
    if turn == 1:
        rooks = torch.nonzero((position.abs() == 2) & (position > 0), as_tuple=False)
        for rook in rooks:
            x, y = rook[0].item(), rook[1].item()
            for dx in [-1, 1]:
                nx = x + dx
                while 0 <= nx < 8:
                    if position[nx, y] == 0:
                        moves.append(((x, y), (nx, y)))
                    elif position[nx, y] < 0:
                        moves.append(((x, y), (nx, y)))
                        break
                    else:
                        break
                    nx += dx
            for dy in [-1, 1]:
                ny = y + dy
                while 0 <= ny < 8:
                    if position[x, ny] == 0:
                        moves.append(((x, y), (x, ny)))
                    elif position[x, ny] < 0:
                        moves.append(((x, y), (x, ny)))
                        break
                    else:
                        break
                    ny += dy
    else:
        rooks = torch.nonzero((position.abs() == 2) & (position < 0), as_tuple=False)
        for rook in rooks:
            x, y = rook[0].item(), rook[1].item()
            for dx in [-1, 1]:
                nx = x + dx
                while 0 <= nx < 8:
                    if position[nx, y] == 0:
                        moves.append(((x, y), (nx, y)))
                    elif position[nx, y] > 0:
                        moves.append(((x, y), (nx, y)))
                        break
                    else:
                        break
                    nx += dx
            for dy in [-1, 1]:
                ny = y + dy
                while 0 <= ny < 8:
                    if position[x, ny] == 0:
                        moves.append(((x, y), (x, ny)))
                    elif position[x, ny] > 0:
                        moves.append(((x, y), (x, ny)))
                        break
                    else:
                        break
                    ny += dy
    return moves
def queen_moves(position, turn):
    moves = []
    if turn == 1:
        queens = torch.nonzero((position.abs() == 5) & (position > 0), as_tuple=False)
        for queen in queens:
            x, y = queen[0].item(), queen[1].item()
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                while 0 <= nx < 8 and 0 <= ny < 8:
                    if position[nx, ny] == 0:
                        moves.append(((x, y), (nx, ny)))
                    elif position[nx, ny] < 0:
                        moves.append(((x, y), (nx, ny)))
                        break
                    else:
                        break
                    nx += dx
                    ny += dy
            for dx in [-1, 1]:
                nx = x + dx
                while 0 <= nx < 8:
                    if position[nx, y] == 0:
                        moves.append(((x, y), (nx, y)))
                    elif position[nx, y] < 0:
                        moves.append(((x, y), (nx, y)))
                        break
                    else:
                        break
                    nx += dx
            for dy in [-1, 1]:
                ny = y + dy
                while 0 <= ny < 8:
                    if position[x, ny] == 0:
                        moves.append(((x, y), (x, ny)))
                    elif position[x, ny] < 0:
                        moves.append(((x, y), (x, ny)))
                        break
                    else:
                        break
                    ny += dy
    else:
        queens = torch.nonzero((position.abs() == 5) & (position < 0), as_tuple=False)
        for queen in queens:
            x, y = queen[0].item(), queen[1].item()
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                while 0 <= nx < 8 and 0 <= ny < 8:
                    if position[nx, ny] == 0:
                        moves.append(((x, y), (nx, ny)))
                    elif position[nx, ny] > 0:
                        moves.append(((x, y), (nx, ny)))
                        break
                    else:
                        break
                    nx += dx
                    ny += dy
            for dx in [-1, 1]:
                nx = x + dx
                while 0 <= nx < 8:
                    if position[nx, y] == 0:
                        moves.append(((x, y), (nx, y)))
                    elif position[nx, y] > 0:
                        moves.append(((x, y), (nx, y)))
                        break
                    else:
                        break
                    nx += dx
            for dy in [-1, 1]:
                ny = y + dy
                while 0 <= ny < 8:
                    if position[x, ny] == 0:
                        moves.append(((x, y), (x, ny)))
                    elif position[x, ny] > 0:
                        moves.append(((x, y), (x, ny)))
                        break
                    else:
                        break
                    ny += dy
    return moves
def king_moves(position, turn, gamehis, last_move):
    moves = []
    if turn == 1:
        kings = torch.nonzero((position.abs() == 6) & (position > 0), as_tuple=False)
        king = kings[0]
        x, y = king[0].item(), king[1].item()
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                if position[nx, ny] <= 0:
                    moves.append(((x, y), (nx, ny)))
    else:
        kings = torch.nonzero((position.abs() == 6) & (position < 0), as_tuple=False)
        king = kings[0]
        x, y = king[0].item(), king[1].item()
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                if position[nx, ny] >= 0:
                    moves.append(((x, y), (nx, ny)))

    
    castl = is_castlable(gamehis, turn, position, last_move)
    moves.extend(castl)
    return moves

def is_castlable(gamehis, turn, pos, last_move):
    castle_mov = []
    if turn == 1:
        if all(move[0] != (7, 4) for move in gamehis):
            if all(move[0] != (7, 7) for move in gamehis):
                if position[7][5].item() == 0 and position[7][6] == 0 and position[7][4] == 6 and position[7][7] == 2:
                    castle_mov.append(castle_mov_w2)
            if all(move[0] != (7, 0) for move in gamehis):
                if position[7][1].item() == 0 and position[7][2] == 0 and position[7][3] == 0 and position[7][4] == 6 and position[7][0] == 2:
                    castle_mov.append(castle_mov_w1)
    else:
        if all(move[0] != (0, 4) for move in gamehis):
            if all(move[0] != (0, 7) for move in gamehis):
                if position[0][5].item() == 0 and position[0][6] == 0 and position[0][4] == 6 and position[0][7] == 2:
                    castle_mov.append(castle_mov_b2)
            if all(move[0] != (0, 0) for move in gamehis):
                if position[0][1].item() == 0 and position[0][2] == 0 and position[0][3] == 0 and position[0][4] == 6 and position[0][0] == 2:
                    castle_mov.append(castle_mov_w1)
    return castle_mov

def checkcastright(turn, gamehis):
    if turn == 1:
        if all(move[0] != (7, 4) for move in gamehis):
            if all(move[0] != (7, 7) for move in gamehis):
                if position[7][5].item() == 0 and position[7][6] == 0 and position[7][4] == 6 and position[7][7] == 2:
                    return True
    else:
        if all(move[0] != (0, 4) for move in gamehis):
            if all(move[0] != (0, 7) for move in gamehis):
                if position[0][5].item() == 0 and position[0][6] == 0 and position[0][4] == 6 and position[0][7] == 2:
                    return True
    return False
                
def checkcastleft(turn, gamehis):
    if turn == 1:
        if all(move[0] != (7, 4) for move in gamehis):
            if all(move[0] != (7, 0) for move in gamehis):
                if position[7][1].item() == 0 and position[7][2] == 0 and position[7][3] == 0 and position[7][4] == 6 and position[7][0] == 2:
                    return True
    else:
        if all(move[0] != (0, 4) for move in gamehis):
            if all(move[0] != (0, 0) for move in gamehis):
                if position[0][1].item() == 0 and position[0][2] == 0 and position[0][3] == 0 and position[0][4] == 6 and position[0][0] == 2:
                    return True
    return False

                



def material_equ(position):
    material_score = 0
    pawns = torch.nonzero(position == 1, as_tuple=False)
    for pawn in pawns:
        material_score += 124
    bishops = torch.nonzero(position == 4, as_tuple=False)
    for bishop in bishops:
        material_score += 825
    knights = torch.nonzero(position == 3, as_tuple=False)
    for knight in knights:
        material_score += 781
    rooks = torch.nonzero(position == 2, as_tuple=False)
    for rook in rooks:
        material_score += 1276
    queens = torch.nonzero(position == 5, as_tuple=False)
    for queen in queens:
        material_score += 2538
    pawns = torch.nonzero(position == -1, as_tuple=False)
    for pawn in pawns:
        material_score -= 124
    bishops = torch.nonzero(position == -4, as_tuple=False)
    for bishop in bishops:
        material_score -= 825
    knights = torch.nonzero(position == -3, as_tuple=False)
    for knight in knights:
        material_score -= 781
    rooks = torch.nonzero(position == -2, as_tuple=False)
    for rook in rooks:
        material_score -= 1276
    queens = torch.nonzero(position == -5, as_tuple=False)
    for queen in queens:
        material_score -= 2538
    
    return material_score

def material_diff(poshis):
    oldpos = poshis[len(poshis) - 2]
    newpos = poshis[len(poshis) - 1]
    oldmat = material_equ(oldpos)
    newmat = material_equ(newpos)
    return (newmat - oldmat) * -1

def get_legalmoves(position, turn, last_move, gamehis):
    legalmoves = []
    legalmoves.append(pawn_moves(position, turn))
    legalmoves.append(pawn_attack(position,turn))
    legalmoves.append(pawn_en_passant(position, turn, last_move))
    legalmoves.append(bishop_moves(position, turn))
    legalmoves.append(knight_moves(position, turn))
    legalmoves.append(rook_moves(position, turn))
    legalmoves.append(queen_moves(position, turn))
    legalmoves.append(king_moves(position, turn, gamehis, last_move))
    legalmoves = [move for move in legalmoves if move]
    return legalmoves

def make_moves(position, legalmoves):
    positions = []
    for type in legalmoves:
        for move in type:
            positions.append(make_move(position, move).tolist())
        return torch.tensor(positions)
    
def make_move(position, move):
    start, finish = move[0], move[1]
    oldposition = position.clone()
    piece = position[start[0], start[1]] 
    oldposition[start[0], start[1]] = 0
    oldposition[finish[0], finish[1]] = piece
    position = oldposition
    return position

def Castle_move(curposition, move):
    if move == ((7, 4), (7, 6)):
        position2 = curposition.clone()
        position2[7][4] = 0
        position2[7][6] = 6
        position2[7][7] = 0
        position2[7][5] = 2
        return position2
    elif move == ((7, 4), (7, 2)):
        position2 = curposition.clone()
        position2[7][4] = 0
        position2[7][2] = 6
        position2[7][0] = 0
        position2[7][3] = 2
        return position2
    elif move == ((0, 4), (0, 6)):
        position2 = curposition.clone()
        position2[0][4] = 0
        position2[0][6] = -6
        position2[0][7] = 0
        position2[0][5] = -2
        return position2
    elif move == ((0, 4), (0, 2)):
        position2 = curposition.clone()
        position2[0][4] = 0
        position2[0][2] = -6
        position2[0][0] = 0
        position2[0][3] = -2
        return position2


def remove_pinned(position, turn, legalmoves, last_move, gamehis): 
    global kingsaf
    kingsaf = 0
    pinned = []
    presentlegal = legalmoves
    turn = turn * -1
    for thing in presentlegal:
        for one in thing:
            
            pos = make_move(position, one)
            legalmoves = get_legalmoves(pos, turn, last_move, gamehis)
            for type in legalmoves:
                if type == [[]]:
                    continue
                for move in type:
                    finish = [move[1]]
                    king_cor = [(torch.nonzero(pos * turn == -6,as_tuple=False).tolist()[0][0], torch.nonzero(pos * turn == -6,as_tuple=False).tolist()[0][1])]
                    if finish == king_cor:
                        kingsaf -= 1
                        pinned.append(one)
    
    if pinned == []:
        presentlegal = [move for thing in presentlegal for move in thing]
        return presentlegal
    else:
        presentlegal = [move for thing in presentlegal for move in thing]
        presentlegal = [move for move in presentlegal if move not in pinned]
        return presentlegal

def check_con(position, turn, legalmove, last_move, history, gamehis):
    global winner
    positioncount = 0
    pieces = []
    piece = torch.nonzero(position, as_tuple=False)
    for cor in piece:
        pieces.append(position[cor[0], cor[1]].item())
    pieces.sort()

    for hisposition in history:
        if torch.equal(hisposition, position):
            positioncount = positioncount + 1

    if len(legalmove) == 0:
        turn = turn * -1
        legalmoves = get_legalmoves(position, turn, last_move, gamehis)
        for move in legalmoves:
            mov = move[0]
            finish = [mov[1]]
            king_cor = [(torch.nonzero(position * turn == -6,as_tuple=False).tolist()[0][0], torch.nonzero(position * turn == -6,as_tuple=False).tolist()[0][1])]
            if finish == king_cor:
                winner = turn
                print("Checkmate by " + str(winner))
                return False
            else:
                continue
        print("Stalemate!")
        winner = 0.5
        return False
    elif pieces == [-6,6] or pieces == [-6,-3,6] or pieces == [-6,-4,6] or pieces  == [-6,3,6] or pieces == [-6,4,6]:
        winner = 0.5
        print("Draw by insufficient material!")
        return False
    elif positioncount >= 3:
        winner = 0.5
        print("Draw by repetition!")
        return False
    else:
        return True


    

def get_final_legal(position, turn, last_move, gamehis):
    legalmoves = get_legalmoves(position, turn, last_move, gamehis)
    legalmoves = remove_pinned(position, turn, legalmoves, last_move, gamehis)
    return legalmoves

def get_material_equ():
    return material_equ(position)

def get_num_moves(legalmove):
    return len(legalmove)

def changeturn():
    global turn
    turn = -turn

def get_king_safe():
    return kingsaf
