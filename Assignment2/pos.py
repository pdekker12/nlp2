
generic_to_core_pos = {}

f = open("../data/en-ptb.map","r")
lines = f.readlines()
for line in lines:
    splitline = line.split()
    generic_to_core_pos[splitline[0]] = splitline[1]

core_to_generic_pos = {}

for key, value in generic_to_core_pos.items():
    if value not in core_to_generic_pos:
        core_to_generic_pos[value] = {key}
    else:
        core_to_generic_pos[value].add(key)

core_tags = list(core_to_generic_pos.keys())
