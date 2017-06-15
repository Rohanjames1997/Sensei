import math
from numpy import poly1d
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from random import randint
from random import seed

def main():
    #test()
    seed(0)
    generate_data()
    generate_data('test.csv', 10000)

def generate_data(name='train.csv', records=100000):
    f = open(name, 'w')
    f.write('cx,cy,angle,tx,ty,hit,miss\n')
    for i in range(records):
        acceptable = False
        while (not acceptable):
            cx = randint(0,20)
            cy = randint(0,9)
            ca = randint(0,180)
            tx = randint(0,20)
            ty = randint(10,19)
            tr = 2#randint(1,10)
            game = Console(Canon(x=cx, y=cy, angle=ca), Target(x=tx, y=ty, radius=tr))
            result = game.shoot()
            if not result:
                acceptable = (randint(0,10) < 2)
            else:
                acceptable = True
        if (result):
            result = '1,0'
        else:
            result = '0,1'
        line = '{},{},{},{},{},{}\n'.format(cx,cy,ca,tx,ty,result)
        f.write(str(line))
    f.close()

def test():
    print('Cannon: ')
    c = Canon(x=float(input('enter X: ')), y=float(input('enter Y: ')), angle=float(input('enter Angle: ')))
    traj = c.get_trajectory()
    print('equation of trajectory: ')
    print(traj)
    print('---------------------------------\nTarget: ')
    t = Target(x=float(input('enter X: ')), y=float(input('enter Y: ')), radius=float(input('enter Radius: ')))
    print('shortest distance: ')
    print(Canvas.shortest_distance((t.X, t.Y), traj))
    game = Console(c, t)
    game.shoot(silent=False)
    game.display()



class Thing:
    
    def __init__(self, x=0, y=0, name=''):
        self.name = name
        self.X = x
        self.Y = y
    
    def __repr__(self):
        return '"{}" is at location ({}, {})'.format(self.name, self.X, self.Y)


class Canon(Thing):
    
    def __init__(self, x=0, y=0, angle=None, name=''):
        super().__init__(x, y, name)
        self.angle = angle
    
    def __repr__(self):
        return 'Canon "{}" is at location ({}, {}), inclined at an angle of {} degrees'.format(self.name, self.X, self.Y, self.angle)
    
    def __str__(self):
        return 'The canon is at ({}, {}), facing {} degrees.'.format(self.X, self.Y, self.angle)
    
    def get_trajectory(self):
        if self.angle is None:
            raise ValueError('Canon Angle is undefined')
        #if abs(self.angle -90) < 0.5:
        m = math.tan(math.radians(self.angle))
        c = float(self.Y) - m*self.X
        return poly1d([m, c])


class Target(Thing):
    
    def __init__(self, x=0, y=0, radius=1, name=''):
        super().__init__(x, y, name)
        self.radius = radius
    
    def __repr__(self):
        return 'Target "{}" is at location ({}, {}), with a radius of {} units'.format(self.name, self.X, self.Y, self.radius)
    
    def __str__(self):
        return 'The target of size {} is at ({}, {}).'.format(self.radius, self.X, self.Y)


class Canvas:

    def __init__(self, width=100, height=100):
        if width < 0:
            raise ValueError("Canvas width can't be negative.")
        if height < 0:
            raise ValueError("Canvas height can't be negative.")
        self.width = width
        self.height = height
    
    def __repr__(self):
        return 'Canvas of size {} x {}'.format(self.width, self.height)
    
    def shortest_distance(point, line):
        x, y = point
        if len(line.c) == 2: # assume line is of the form y = m*x + c
            m = line.c[0]
            c = line.c[1]
        else: # assume line is of the form y = c
            m = 0
            c = line.c[0]
        num = float(m)*float(x) - float(y) + float(c)
        den = math.sqrt(m*m+1)
        return (abs(num)/den)
    
    def setup_display(self):
        #fig_size = 
        pwidth = int(self.width*1.2)+2
        pheight = int(self.height*1.2)+2
        fig = plt.figure(1)
        #plt.rcParams["figure.figsize"] = [pwidth, pheight]
        ground = fig.add_subplot(111, aspect='equal')
        ground.add_patch(patches.Rectangle((0, 0), self.width, self.height, facecolor='brown', edgecolor='brown', linestyle='dotted', alpha=0.1))
        ground.set_xlim([-(pwidth-self.width)/2, self.width+((pwidth-self.width)/2)])
        ground.set_ylim([-(pheight-self.height)/2, self.height+((pheight-self.height)/2)])
    
    def plot_circle(self, centre, rad):
        fig = plt.figure(1)
        ground = plt.subplot(111, aspect='equal')
        ground.add_patch(patches.Circle(centre, radius=rad, color='red'))
    
    def plot_line_segment(self, x, y, dx, dy):
        fig = plt.figure(1)
        ground = plt.subplot(111, aspect='equal')
        ground.add_patch(patches.Arrow(x, y, dx, dy))


class ScoreBoard:

    def __init__(self, hits=0, misses=0):
        self.hits = hits
        self.misses = misses
        if self.hits < 0 or self.misses < 0:
            raise ValueError('Cannot initialise invalid score of {} hits and {} misses'.format(self.hits, self.misses))

    def __repr__(self):
        return 'Score is currently {} hits, and {} misses.'.format(self.hits, self.misses)
    
    def hit(self):
        self.hits += 1
    
    def miss(self):
        self.misses += 1


class Console:
    
    def __init__(self, canon=None, target=None, width=100, height=100, hits=0, misses=0, canonX=0, canonY=0, targetX=0, targetY=0):
        self.canvas = Canvas(width, height)
        self.score_board = ScoreBoard(hits, misses)
        if canon is not None:
            self.c = canon
        else:
            self.c = Canon(x=canonX, y=canonY)
        if self.c.X > self.canvas.width or self.c.X < 0:
            raise ValueError('Cannot place cannon at x={}'.format(c.X))
        if self.c.Y > self.canvas.height or self.c.Y < 0:
            raise ValueError('Cannot place cannon at y={}'.format(c.Y))
        if target is not None:
            self.t = target
        else:
            self.t = target(x=canonX, y=canonY)
        if self.t.X > self.canvas.width or self.t.X < 0:
            raise ValueError('Cannot place target at x={}'.format(t.X))
        if self.t.Y > self.canvas.height or self.t.Y < 0:
            raise ValueError('Cannot place target at y={}'.format(t.Y))
        
    def get_canon_position(self):
        return (self.c.X, self.c.Y)
    
    def get_canon_angle(self):
        return self.c.angle
    
    def get_target_position(self):
        return (self.t.X, self.t.Y)
    
    def get_target_radius(self):
        return self.t.radius
    
    def _hit(self):
        line = self.c.get_trajectory()
        point = self.get_target_position()
        sd = Canvas.shortest_distance(point, line)
        return sd <= self.get_target_radius() # True for hit
    
    def shoot(self, silent=True):
        result = self._hit()
        if (result):
            self.score_board.hit()
        else:
            self.score_board.miss()
        if not silent:
            if result:
                print('Hit!')
            else:
                print('Miss.')
        return result
    
    def display(self):
        self.canvas.setup_display()
        self.canvas.plot_circle(self.get_target_position(), self.get_target_radius())
        x, y = self.get_canon_position()
        dx = self.get_target_radius() * 3 * math.cos(math.radians(self.get_canon_angle()))
        dy = self.get_target_radius() * 3 * math.sin(math.radians(self.get_canon_angle()))
        self.canvas.plot_line_segment(x, y, dx, dy)
        plt.show()
        

if __name__ == '__main__':
    main()
