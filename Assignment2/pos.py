core_tags = {'N', 'V', 'D', 'C', 'NUM', 'R', 'J', 'P', 'I', 'POS'}

generic_to_core_pos = {
    'NN' : 'N',
    'NNP' : 'N',
    'NNS' : 'N',
    'NNPS' : 'N',
    'VB' : 'V',
    'VBP' : 'V',
    'VBG' : 'V',
    'VBN' : 'V',
    'VBD' : 'V',
    'DT' : 'D',
    'WDT' : 'D',
    'CC' : 'C',
    'CD' : 'NUM',
    'RB' : 'R',
    'WRB' : 'R',
    'JJ' : 'J',
    'PRP' : 'P',
    'IN' : 'I',
    'POS' : 'POS' # Is it correct to map?
    }

core_to_generic_pos = {}

for key, value in generic_to_core_pos.items():
    if value not in core_to_generic_pos:
        core_to_generic_pos[value] = {key}
    else:
        core_to_generic_pos[value].add(key)


