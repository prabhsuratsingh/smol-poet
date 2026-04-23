from collections import Counter


def line_structure_score(text):
    lines = text.split("\n")
    lines = [l.strip() for l in lines if l.strip()]

    if len(lines) < 3:
        return -1.0

    avg_len = sum(len(l.split()) for l in lines) / len(lines)

    # sweet spot: 4–10 words per line
    if 4 <= avg_len <= 10:
        return 1.0
    elif 2 <= avg_len <= 14:
        return 0.5
    else:
        return -0.5


# def repetition_penalty(text):
#     words = text.split()
#     if len(words) < 10:
#         return -1.0

#     counts = Counter(words)
#     max_freq = max(counts.values())

#     ratio = max_freq / len(words)

#     # penalize heavy repetition
#     # return -ratio * 2.0
#     return -ratio * 3.0

def repetition_penalty(text):
    words = text.split()
    if len(words) < 10:
        return -1.0

    counts = Counter(words)
    max_freq = max(counts.values())
    ratio = max_freq / len(words)

    # harsher penalty curve
    return - (ratio ** 1.5) * 4.0

def length_score(text):
    n = len(text.split())

    if 40 <= n <= 120:
        return 1.0
    elif 20 <= n <= 160:
        return 0.5
    else:
        return -0.5

# def rhyme_score(text):
#     lines = [l.strip() for l in text.split("\n") if l.strip()]
#     if len(lines) < 2:
#         return 0.0

#     endings = [l.split()[-1][-3:] for l in lines if len(l.split()) > 0]

#     matches = 0
#     for i in range(len(endings) - 1):
#         if endings[i] == endings[i+1]:
#             matches += 1

#     return matches / len(lines)

import re

def rhyme_score(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 2:
        return 0.0

    endings = []
    for l in lines:
        words = l.split()
        if not words:
            continue
        w = re.sub(r'[^a-zA-Z]', '', words[-1].lower())
        if len(w) >= 3:
            endings.append(w[-3:])

    if len(endings) < 2:
        return 0.0

    matches = sum(endings[i] == endings[i+1] for i in range(len(endings)-1))
    return matches / len(endings)

def entropy_bonus(text):
    words = text.split()
    unique_ratio = len(set(words)) / len(words)

    # return unique_ratio
    return min(unique_ratio, 0.6)

def word_validity_score(text):
    words = text.split()
    if not words:
        return 0.0

    valid = sum(
        w.isalpha() and 3 <= len(w) <= 12
        for w in words
    )

    return valid / len(words)

def compute_reward(text):
    return (
        1.5 * line_structure_score(text) +
        1.5 * repetition_penalty(text) +
        0.5 * length_score(text) +
        0.8 * rhyme_score(text) +
        0.3 * entropy_bonus(text) +
        1.0 * word_validity_score(text)
    )