import torch
import re
map = {
        'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10,
        'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20,
        'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26,
        '0': 27, '1': 28, '2': 29, '3': 30, '4': 31, '5': 32, '6': 33, '7': 34, '8': 35, '9': 36,
        '+': 37, '-': 38, '*': 39, '/': 40, '=': 41, ' ': 42, '?': 43, '\'': 44,
        '(': 45, ')': 46, '[': 47, ']': 48, '{': 49, '}': 50, '.': 51, ',': 52, '!': 53
}

nummap = {
    27: 0, 28: 1, 29: 2, 30: 3, 31: 4, 32: 5, 33: 6, 34: 7, 35: 8, 36: 9, 51: "."
}


def embed(text):
    list = []

    for char in text.lower():
        list.append(map[char])

    return torch.tensor(list)

def detect_num(tensor):
    numbers = []
    num = ""
    input = tensor.numpy().tolist()
    for emb in input:
        if emb in nummap:
            num += str(nummap[emb])
        elif emb not in nummap and num != "":
            numbers.append(float(num))
            num = ""
    if num != "":
        numbers.append(float(num))
    return numbers




input_text = input("Input your question: ")
embedded_input = embed(input_text)
print("Embedded sequence in tensor:")
print(embedded_input)

numbers = detect_num(embedded_input)
print("Detected numbers:")
print(numbers)
