# Creating folders to store the results:
bash_line2 = "for i in $(seq 0 " + str(n_alt-1)+ "); do mkdir core"+ str(core) + "_alt${i}_" + structure + "; done;"

# Current core where the analysis is happening:
core = parser.core

# Number of alternative runs per bin:
n_alt = parser.alt
