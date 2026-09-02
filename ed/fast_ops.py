def occupied(state, orb):
    return (state >> orb) & 1 # checks if orb is 1

def hop_matrix_elements(states, i, j):
    """
    Stores all finite <bra|c_i^dagger c_j|ket> matrix elements, 
    which are either 1 or -1, depending if the sites in between 
    i and j are even or odd.
    """
    index = {s: k for k, s in enumerate(states)}
    lo, hi = (i, j) if i < j else (j, i)
    between_mask = ((1 << hi) - 1) & ~((1 << (lo + 1)) - 1) # bits between lo and hi are 1, rest are zero

    entries = []
    for ket, init_state in enumerate(states):
        if not occupied(init_state, j):
            continue
        tmp = init_state & ~(1 << j) # sets bit j to 0
        if occupied(tmp, i):
            continue
        final_state = tmp | (1 << i) # sets bit i to 1
        bra = index[final_state]
        sign = -1 if bin(init_state & between_mask).count("1") % 2 else 1
        entries.append((ket, bra, sign))
    return entries
