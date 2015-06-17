import string

f_input = open("../data/cs-ud-dev.conllu","r")
lines = f_input.readlines()

output_string = ""

i=0
prev_line_number = 0
while(i < len(lines)):
    line = lines[i]
    if len(line) > 1:
        spacesplit = line.split()
        if spacesplit[1] == "sent_id":
            # New sentence found
            i+=1 # skip other comment line
        else:
            tabsplit = line.split("\t")
            line_number = tabsplit[0]
            if line_number.isdigit():
                if (int(line_number) <= int(prev_line_number)):
                    # If the line number is suddenly low, new sentence
                    output_string += "\n"
                prev_line_number = line_number
                word = tabsplit[1]
                tag = tabsplit[3]
                if word not in string.punctuation:
                    output_string += word + "\\" + tag + " "
    i+=1

print(output_string)
