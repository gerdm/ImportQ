import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from numpy import sin, cos

fxy = lambda x, y: sin(x*y) / ((x*y)**2 + 1)
pfx = lambda x, y: -2*x*y**2*sin(x*y)/(x**2*y**2 + 1)**2 + y*cos(x*y)/(x**2*y**2 + 1)
pfy = lambda x, y: -2*x**2*y*sin(x*y)/(x**2*y**2 + 1)**2 + x*cos(x*y)/(x**2*y**2 + 1)

Fxy = np.vectorize(fxy)
pFx = np.vectorize(pfx)
pFy = np.vectorize(pfy)

class Roll(object):
    def __init__(self, x, y, func, grad_x, grad_y, alpha, axes):
        # Costant coordinates
        self.initial_x = [x]
        self.initial_y = [y]
        self.initial_z = [func(x,y)]
        # Variable coordinates
        self.x = [x]
        self.y = [y]
        self.z = [func(x,y)]

        self.alpha = alpha
        self.func =  func
        self.grad_x = grad_x
        self.grad_y = grad_y
        self.axes = axes
    
    def update_pos(self, nx, ny, nz):
        self.x.append(nx)
        self.y.append(ny)
        self.z.append(nz)

    def pos_out(self):
        """Return the last computed coordinates"""
        return (self.x[-1], self.y[-1], self.z[-1])

    def remove_points(self):
        self.x = []
        self.y = []
        self.z = []

    def reset_coordinates(self):
        self.x = self.initial_x
        self.y = self.initial_y
        self.z = self.initial_z

    def compute_new_pos(self, in_x, in_y, in_z):
        """Compute the new x and y positions for given (x,y)"""
        out_dx = in_x - self.alpha * self.grad_x(in_x, in_y)
        out_dy = in_y - self.alpha * self.grad_y(in_x, in_y)
        out_dz = self.func(in_x, in_y)
        return out_dx, out_dy, out_dz

    def compute_update(self):
        dx, dy, dz = self.pos_out()
        step = self.axes.plot3D([dx], [dy], [dz], "o", alpha=0.5)
        new_x, new_y, new_z = self.compute_new_pos(dx, dy, dz)
        self.update_pos(new_x, new_y, new_z)
        return step


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

roll1 = Roll(x=0.64, y=1.05, func=fxy, grad_x=pfx, grad_y=pfy, alpha=0.04, axes=ax)
roll2 = Roll(x=-1, y=-1, func=fxy, grad_x=pfx, grad_y=pfy, alpha=0.04, axes=ax)
roll3 = Roll(x=2, y=-2, func=fxy, grad_x=pfx, grad_y=pfy, alpha=0.04, axes=ax)
roll4 = Roll(x=2, y=0.3, func=fxy, grad_x=pfx, grad_y=pfy, alpha=0.04, axes=ax)

rolls = [roll1, roll2, roll3, roll4]

x, y = np.linspace(-3, 3, 250), np.linspace(-3, 3, 250)
x, y = np.meshgrid(x, y)
z = Fxy(x, y)

max_frames = 130


def gradient_desc(i):
    print("At frame {} of {}".format(i+1, max_frames), end="\r")
    ax.clear()
    ax.set_title(r"$\frac{\sin{\left (x y \right )}}{x^{2} y^{2} + 1}$")
    ax.view_init(62, 180 - 0.15* i)

    for roll in rolls:
        roll.compute_update()

    surface = ax.plot_surface(x, y, z, cmap=plt.cm.BrBG, alpha=0.7)

    if i == (max_frames - 1):
        for roll in rolls:
            roll.reset_coordinates()
    return 

animate = animation.FuncAnimation(fig, gradient_desc, interval=30, frames = max_frames, blit=False)
#animate.save('gd.mp4', fps=15, extra_args=['-vcodec', 'libx264'])
print("Processing...")
animate.save('gd.gif', fps=60, writer='imagemagick')
print("\nDONE!")
#plt.show()
