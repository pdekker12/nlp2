

import subprocess

sources = ["de","en","fr","nl"]
targets = ["cs"]

path_root = "../data/europarl/"

for target in targets:
    tfile = path_root + target + ".txt"
    for source in sources:
        print(source + "-" + target)
        output = ""
        sfile = path_root + source + ".txt"
        f = open(path_root + source + "-" + target + ".txt","w")
        output = subprocess.call(["python3", "create_fastalign_input.py", sfile, tfile],stdout=f)
        
