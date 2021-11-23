import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5
zero_shot_classifier = pipeline("zero-shot-classification")   
#candidate_labels=["Flight Travel", "Cabs Travel", "Reminders", "Food", "Movies","enjoyment","PVR cinemas"]
sequences="I love to watch cricket in a mobile phone while having a popcorn"
file = open("input.txt", "r")

items= [] 
for line in file:
  stripped_line = line.strip()
  new_line= stripped_line.split()
  if(new_line):
      items.append(" ".join(new_line))
file.close()
#print(items)

result = zero_shot_classifier(sequences , items ,multi_class= True)   
#print(result,result["labels"],result["scores"],end="\n")
x=result["labels"][:5]
y=result["scores"][:5]
print(x)
print("--------------------")
y=[i*100 for i in result["scores"][:5]]
print(y)
plt.bar(x,y)
plt.yticks(list(np.arange(0, 1, 0.1)))
plt.show()