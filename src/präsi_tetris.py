import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Tuple 
import torch
import numpy as np

height = 20  # Höhe des Tetris-Spielfeldes
width = 10   # Breite des Tetris-Spielfeldes

# Beispiel-Spielbrett
game_board = [
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 6, 6, 6, 0],
    [0, 0, 0, 0, 0, 0, 6, 4, 1, 1],
    [0, 0, 0, 0, 0, 0, 4, 4, 1, 1],
    [0, 0, 0, 0, 0, 3, 4, 7, 7, 4],
    [5, 6, 6, 5, 5, 5, 5, 3, 3, 0],
    [5, 0, 3, 3, 0, 1, 1, 6, 2, 2],
    [7, 7, 7, 6, 1, 1, 5, 4, 4, 0],
    [7, 7, 0, 6, 4, 4, 5, 5, 4, 4],
    [5, 7, 6, 0, 5, 2, 2, 3, 5, 5]
]

piece_colors = [
    (0, 0, 0),        # Empty space
    (225, 225, 0),    # O piece, yellow
    (147, 88, 254),   # T piece, purple
    (54, 175, 144),   # S piece, green
    (255, 0, 0),      # Z piece, red
    (102, 217, 238),  # I piece, cyan
    (254, 151, 32),   # L piece, orange
    (0, 0, 255),      # J piece, blue
]

def plot_gamegrid_heatmap(name, highlight_data=[], title=None):
    normalized_colors = [(r/255, g/255, b/255) for r, g, b in piece_colors]
    cmap = ListedColormap(normalized_colors)
    plt.figure(figsize=(width, height))
    ax = sns.heatmap(game_board, cmap=cmap, linewidths=0.5, cbar=False, annot=True, fmt="d", annot_kws={"size": 12, "weight": "bold", "color": "white"})
    
    # Zellen hervorheben
    for (i, j) in highlight_data:
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=3))
    if title:
        plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.xticks(ticks=np.arange(0.5, len(game_board[0]), 1), labels=[str(i) for i in range(len(game_board[0]))])
    plt.yticks(ticks=np.arange(0.5, len(game_board), 1), labels=[str(i) for i in range(len(game_board))])
    plt.savefig(f'./src/Images_of_tetris/{name}.svg', bbox_inches='tight', dpi=300)
    plt.close()

def get_state_properties(board:list[list[int]]) -> torch.FloatTensor:
    lines_cleared, board = check_cleared_rows(board)
    holes = get_holes(board)
    bumpiness, height = get_bumpiness_and_height(board)

    return torch.FloatTensor([lines_cleared, holes, bumpiness, height])

def check_cleared_rows(board:list[list[int]]) -> Tuple[int, list]:
    to_delete = []
    for i, row in enumerate(board[::-1]):
        if 0 not in row:
            to_delete.append(len(board) - 1 - i)
    return len(to_delete), board

def get_holes(board:list[list[int]]) -> int:
    num_holes = 0
    holes_coordinates = []
    for col_id, col in enumerate(zip(*board)):
        row = 0
        while row < height and col[row] == 0: row += 1
        num_holes += len([x for x in col[row + 1 :] if x == 0])
        holes_coordinates.extend([(row + i + 1, col_id) for i, x in enumerate(col[row + 1 :]) if x == 0])
    plot_gamegrid_heatmap('Holes_Tetris_Board_next_position', highlight_data=holes_coordinates, title=f'Holes in Tetris Board: {num_holes}')
    return num_holes

def get_bumpiness_and_height(board:list[list[int]]) -> Tuple[int, int]:
    board = np.array(board)
    mask = (board != 0)
    invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), height)
    heights = height - invert_heights
    total_height = np.sum(heights)
    total_bumpiness = np.sum(np.abs(heights[:-1] - heights[1:]))
    plot_gamegrid_heatmap('Heights_Tetris_Board_next_position', highlight_data=[(i, j) for j, i in enumerate(invert_heights)], title=f'Heights in Tetris Board: {total_height}')

    bumpiness_cells = []
    for index in range(len(heights) - 1):
        if heights[index] != heights[index + 1]:
            lower_height = min(heights[index], heights[index + 1])
            higher_height = max(heights[index], heights[index + 1])
            for i in range(lower_height + 1, higher_height + 1):
                bumpiness_cells.append((height - i, index + 1)) 
    
    plot_gamegrid_heatmap('Bumpiness_Tetris_Board_next_position', highlight_data=bumpiness_cells, title=f'Bumpiness in Tetris Board: {total_bumpiness}')
    
    return total_bumpiness, total_height

def get_next_states() -> dict[Tuple[int, int], torch.FloatTensor]:
    states = {}
    piece_id = 0
    piece = [[1, 1], [1, 1]]
    curr_piece = [row[:] for row in piece]
    
    # Determine the number of rotations based on the piece type
    if piece_id == 0: num_rotations = 1
    elif piece_id == 2 or piece_id == 3 or piece_id == 4: num_rotations = 2
    else: num_rotations = 4

    for i in range(num_rotations):
        valid_xs = width - len(curr_piece[0])
        for x in range(valid_xs + 1):
            piece, pos = [row[:] for row in curr_piece], {"x": x, "y": 0}
            while not check_collision(piece, pos): pos["y"] += 1
            truncate(piece, pos)
            board = store(piece, pos)
            states[(x, i)] = get_state_properties(board)
        curr_piece = rotate(curr_piece)
    return states

def check_collision(piece:list[list[int]], pos:dict[str,int]) -> bool:
    future_y = pos["y"] + 1
    for y in range(len(piece)):
        for x in range(len(piece[y])):
            if (future_y + y > height - 1
                or game_board[future_y + y][pos["x"] + x]
                and piece[y][x]):
                return True
    return False

def truncate(piece:list[list[int]], pos:dict[str,int]) -> bool:
    gameover = False
    last_collision_row = -1
    for y in range(len(piece)):
        for x in range(len(piece[y])):
            if game_board[pos["y"] + y][pos["x"] + x] and piece[y][x]:
                if y > last_collision_row:
                    last_collision_row = y

    if pos["y"] - (len(piece) - last_collision_row) < 0 and last_collision_row > -1:
        while last_collision_row >= 0 and len(piece) > 1:
            gameover = True
            last_collision_row = -1
            del piece[0]
            for y in range(len(piece)):
                for x in range(len(piece[y])):
                    if (game_board[pos["y"] + y][pos["x"] + x]
                        and piece[y][x]
                        and y > last_collision_row):
                        last_collision_row = y
    return gameover

def store(piece:list[list[int]], pos:dict[str,int]) -> list[list[int]]:
    board = [x[:] for x in game_board]
    for y in range(len(piece)):
        for x in range(len(piece[y])):
            if piece[y][x] and not board[y + pos["y"]][x + pos["x"]]:
                board[y + pos["y"]][x + pos["x"]] = piece[y][x]
    return board

def remove_row(board:list[list[int]], indices:list) -> list[list[int]]:
    for i in indices[::-1]:
        del board[i]
        board = [[0 for _ in range(width)]] + board
    return board

def rotate(piece:list[list[int]]) -> list[list[int]]:
    num_rows_orig = num_cols_new = len(piece)
    num_rows_new = len(piece[0])
    rotated_array = []

    for i in range(num_rows_new):
        new_row = [0] * num_cols_new
        for j in range(num_cols_new):
            new_row[j] = piece[(num_rows_orig - 1) - j][i]
        rotated_array.append(new_row)
    return rotated_array

# states = get_next_states()
# for x in states:
#     print(f"Position: {x}, State Properties: {states[x]}")

# print(get_state_properties(game_board))
# plot_gamegrid_heatmap('Tetris_Board_next_position')