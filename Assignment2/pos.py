
from collections import defaultdict

generic_to_core_dict = defaultdict(dict)
core_to_generic_dict = defaultdict(dict)

f = {}
f["en"] = open("../data/en-ptb.map","r")
f["de"] = open("../data/de-negra.map","r")
f["fr"] = open("../data/fr-paris.map","r")
f["es"] = open("../data/es-cast3lb.map","r")

for language in f:
    lines = f[language].readlines()
    for line in lines:
        splitline = line.split()
        generic_to_core_dict[language][splitline[0]] = splitline[1]

    for key, value in generic_to_core_dict[language].items():
        if value not in core_to_generic_dict[language]:
            core_to_generic_dict[language][value] = {key}
        else:
            core_to_generic_dict[language][value].add(key)

    core_tags = list(core_to_generic_dict[language].keys())
#print(core_tags)

def generic_to_core_pos(language,tag):
    if (language=="es"):
        # If a tag is not available, try a shorter version of the same tag
        while tag not in generic_to_core_dict[language] and tag is not "":
            #print(tag)
            tag = tag[:-1]
    
    if tag == "":
        return ""
    
    return generic_to_core_dict[language][tag]

def core_to_generic_pos(language,tag):
    return core_to_generic_dict[language][tag]
