import matplotlib.pyplot as plt
import numpy as np

filetoread = '/home/jordi/PycharmProjects/tetris-ai/logs/100k-3x64-2x32-otherrep/linesfile.txt'
totallinescleared = []
data = np.loadtxt(filetoread, dtype=int, delimiter=' ')
x = np.linspace(1, len(data), num=len(data))
figure = plt.figure()
plt.plot(x,data[:,1])
plt.plot(x,data[:,3])
plt.plot(x,data[:,4])

plt.show()




