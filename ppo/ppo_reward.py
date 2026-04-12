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


def repetition_penalty(text):
    words = text.split()
    if len(words) < 10:
        return -1.0

    counts = Counter(words)
    max_freq = max(counts.values())

    ratio = max_freq / len(words)

    # penalize heavy repetition
    return -ratio * 2.0

def length_score(text):
    n = len(text.split())

    if 40 <= n <= 120:
        return 1.0
    elif 20 <= n <= 160:
        return 0.5
    else:
        return -0.5

def rhyme_score(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 2:
        return 0.0

    endings = [l.split()[-1][-3:] for l in lines if len(l.split()) > 0]

    matches = 0
    for i in range(len(endings) - 1):
        if endings[i] == endings[i+1]:
            matches += 1

    return matches / len(lines)

def entropy_bonus(text):
    words = text.split()
    unique_ratio = len(set(words)) / len(words)

    return unique_ratio

def compute_reward(text):
    return (
        1.5 * line_structure_score(text) +
        1.5 * repetition_penalty(text) +
        0.5 * length_score(text) +
        0.8 * rhyme_score(text) +
        0.3 * entropy_bonus(text)
    )