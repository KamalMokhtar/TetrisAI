import matplotlib.pyplot as plt
import numpy as np

# here you have to choose which file you would like to plot
filetoread = 'lines_logging/linesfile-20200501-003031.txt'
totallinescleared = []
data = np.loadtxt(filetoread, dtype=int, delimiter=' ')
x = np.linspace(1, len(data), num=len(data))
msize = 25
figure = plt.figure()
lines = plt.scatter(x,data[:,1],s=msize)
doubles = plt.scatter(x,data[:,2], s=msize)
triples = plt.scatter(x,data[:,3], s=msize)
tetrises = plt.scatter(x,data[:,4], s=msize)
plt.legend((lines, doubles, triples, tetrises), ('1-lines', '2-lines', '3-lines', 'Tetrises'))

plt.show()