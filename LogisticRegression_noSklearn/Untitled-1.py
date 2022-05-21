import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

p = 0.2 # allele frequency
q = 1-p
f = 0.2 # inbreeding coefficient 
generation = 50
p_AA,p_AG,p_GG = [],[],[]
col = ['AA','GG','AG']
for i in range(0, generation):
    prob_nextGen_AA = p**2 + p*q*f
    prob_nextGen_AG = 2*p*q*(1-f)
    prob_nextGen_GG = q**2 + p*q*f
    p_AA.append(prob_nextGen_AA)
    p_GG.append(prob_nextGen_GG)
    p_AG.append(prob_nextGen_AG)
    p = (prob_nextGen_AA+prob_nextGen_AG)/2 # probability of A allele given genotype
    q = 1-p
# Formatting my dataframe
genotype = np.array((p_AA,p_GG,p_AG))
genotype = np.transpose(genotype)
genotype_freq = pd.DataFrame(genotype, columns=col)

#-------Customizing my plots--------------------------
number_generation = range(0,generation)
plt.figure()
plt.bar(number_generation, genotype_freq.iloc[:, 2], color='g', align='edge')
plt.bar(number_generation, genotype_freq.iloc[:, 1] , bottom =genotype_freq.iloc[:,2], color='purple', align = 'edge')
plt.bar(number_generation, genotype_freq.iloc[:, 0], bottom=genotype_freq.iloc[:, 2], color='b', align = 'edge')
plt.legend(col,loc="upper left")
plt.xlabel("Generations")
plt.ylabel("Allele Frequency")
plt.show()
