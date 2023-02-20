import pyvisgraph as vg
import time
a= time.time()
polys = [[vg.Point(0.0,1.0), vg.Point(3.0,0.0), vg.Point(5,2), vg.Point(4.5,5), vg.Point(1.5,4.0)],
          [vg.Point(4.0,4.0), vg.Point(7.0,4.0), vg.Point(5.5,8.0), vg.Point(3.0, 6.0)]]
g = vg.VisGraph()
g.build(polys)
shortest = g.shortest_path(vg.Point(1.0,0.0), vg.Point(7.0, 7.0))
f = g.shortest_path(vg.Point(5.0,6.0), vg.Point(2.0, 2.0))
print(shortest)
print(time.time()-a)

from matplotlib import pyplot as plt

obsticles = [[[],[]] for _ in range(len(polys))]
for i in range(len(polys)):
    for j in range(len(polys[i])):
        obsticles[i][0].append(polys[i][j].x)
        obsticles[i][1].append(polys[i][j].y)
    obsticles[i][0].append(polys[i][0].x)
    obsticles[i][1].append(polys[i][0].y)
plt.plot(obsticles[0][0],obsticles[0][1],'b-')
# plt.plot(obsticles[0][0],obsticles[0][1],'bo')
plt.plot(obsticles[1][0],obsticles[1][1],'b-')
# plt.plot(obsticles[1][0],obsticles[1][1],'bo')
plt.plot([a.x for a in shortest], [a.y for a in shortest],'r-')
plt.plot([a.x for a in shortest], [a.y for a in shortest],'ro')
# plt.plot([a.x for a in f], [a.y for a in f],'g-')
# plt.plot([a.x for a in f], [a.y for a in f],'go')
plt.show()