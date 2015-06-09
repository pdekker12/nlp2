import string

f_input = open("../data/cs-ud-test.conllu","r")
lines = f_input.readlines()

output_string = ""

i=0
while(i < len(lines)):
    line = lines[i]
    if len(line) > 1:
        spacesplit = line.split()
        if spacesplit[1] == "sent_id":
            # New sentence found
            i+=1 # skip other comment line
            output_string += "\n"
        else:
            tabsplit = line.split("\t")
            word = tabsplit[1]
            tag = tabsplit[3]
            if word not in string.punctuation:
                output_string += word + "\\" + tag + " "
    i+=1

print(output_string)
