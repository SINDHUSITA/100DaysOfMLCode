import math  
class Point(object):
    def __init__(self,x,y):
        self.x=x
        self.y=y
    def distance(self,object):
        i=(self.x-object.x)**2
        j=(self.y-object.y)**2
        return(math.sqrt(i+j))
class Cluster(object):
    def __init__(self, x, y):
        self.center = Point(x, y)
        self.points = []




    
    def update(self):
        x_new = 0
        y_new = 0
        for cluster_point in self.points:
            x_new = x_new + cluster_point.x
            y_new = y_new + cluster_point.y
        x_new = x_new/(len(self.points))
        y_new = y_new/(len(self.points))
        self.center = Point(x_new, y_new)
    
    def add_point(self, point):
        self.points.append(point)
def compute_result(points):
    points = [Point(*point) for point in points]
    a = Cluster(1,0)
    b = Cluster(-1,0)
    #a_old = []
    for _ in range(10000): # max iterations
        for point in points:
           
            if point.distance(a.center) < point.distance(b.center):
                a.add_point(point)
                a.update()
                # add the right point
            else:
                b.add_point(point)
                b.update()
                # add the right point
        # if a_old == a.points:
        #     break
        # a_old = a.points
        # a.update()
        # b.update()
        
    return [(a.center.x,a.center.y),(b.center.x,b.center.y)]




print(compute_result([(2,0),(0,2),(3,1),(1,2)]))

#grader.score.vc__k_means(compute_result)