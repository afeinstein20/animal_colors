import colorsys
import numpy as np
from pylab import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

__all__ = ['AnimalColormaps']

class AnimalColormaps(object):

    def __init__(self, sensitivity, N=2056):
        """
        Creates animal friendly colormaps based 
        on known cone sensitivity.

        Parameters
        ----------
        sensitivity : Sensitivity object
        N : int
           Number of unique colors to create.

        Attributes
        ----------
        animal : str
        N : int
        rgb : np.ndarray
           RGB color array.
        """
        self.animal = sensitivity.animal
        self.N = N

        if self.animal == 'human':
            self.rgb = sensitivity.mapped
        else:
            self.rgb = sensitivity.total_map

        self.create_colors()
        self.theta = np.linspace(0, 2*np.pi, N)


    def create_colors(self):
        """ 
        Creates N discrete hex colors from the rgb list. 

        Attributes
        ----------
        colors : np.ndarray
           Array of N colors in the color wheel.
        """
        cmap_seq = LinearSegmentedColormap.from_list('sequential', self.rgb, N=self.N)

        cmap = cm.get_cmap(cmap_seq, self.N)
        colors = []
        for i in range(cmap.N):
            rgb = cmap(i)[:3]
            colors.append(mpl.colors.rgb2hex(rgb))
        colors = np.array(colors)

        self.colors = np.array(colors)

        if self.animal != 'human':
            _, args = np.unique(colors, return_index=True)
            unicolors = self.colors[args]
            
            cmap_seq = LinearSegmentedColormap.from_list('sequential', unicolors, N=self.N)

            cmap = cm.get_cmap(cmap_seq, self.N)
            colors = []
            for i in range(cmap.N):
                rgb = cmap(i)[:3]
                colors.append(mpl.colors.rgb2hex(rgb))
            colors = np.array(colors)

            self.colors = np.array(colors)


    def check(self, x):
        """ Checks to see if value is within 360 degrees of a circle. """
        if x > 360:
            return x-360
        else:
            return x


    def get_ind(self, x):
        """
        Gets corresponding color to input angle.
        """
        inds = np.zeros(len(x), dtype=int)
        print(x)

        for i in range(len(x)):
            inds[i] = np.where(np.rad2deg(self.theta) >= x[i])[0][0]

        return self.colors[inds]
        
    def complementary(self, x):
        """ 
        Pulls complementary color from a given color. 

        Parameters
        ----------
        x : float
           Angle, in radians, to map onto colorwheel.
        """
        ang = self.check(np.abs(x+180))
        hex_vals = self.get_ind([x, ang])
        degrees = [x, ang]
        return hex_vals, degrees


    def split_complementary(self, x):
        """ 
        Pulls split complementary colors from a given color. 

        Parameters      
        ----------    
        x : float
           Angle, in degrees, to map onto colorwheel.  
        """
        x1 = self.check(np.abs(x+150))
        x2 = self.check(np.abs(x+210))
        hex_vals = self.get_ind([x, x1, x2])
        degrees = [x, x1, x2]
        return hex_vals, degrees


    def triadic(self, x):
        """ 
        Pulls triadic colors from a given color. 

        Parameters
        ----------
        x : float
           Angle, in radians, to map onto colorwheel.
        """
        x1 = self.check(np.abs(x+120))
        x2 = self.check(np.abs(x+240))
        hex_vals = self.get_ind([x, x1, x2])
        degrees = [x, x1, x2]
        return hex_vals, degrees

    
    def tetradic(self, x):
        """ 
        Pulls tetradic colors from a given color. 

        Parameters
        ----------
        x : float
           Angle, in radians, to map onto colorwheel.
        """
        x1 = self.check(np.abs(x+90))
        x2 = self.check(np.abs(x+180))
        x3 = self.check(np.abs(x+270))
        degrees = [x, x1, x2, x3]
        hex_vals = self.get_ind(degrees)
        return hex_vals, degrees


    def analagous(self, x):
        """
        Pulls analagous colors from a given color. 

        Parameters
        ----------
        x : float
           Angle, in radians, to map onto colorwheel.
        """
        x1 = self.check(np.abs(x+30))
        x2 = self.check(np.abs(x+60))
        x3 = self.check(np.abs(x+90))
        degrees = [x, x1, x2, x3]
        hex_vals = self.get_ind(degrees)
        return hex_vals, degrees


    def rgb_to_hsv(self, r, g, b):
        """ Converts RGB colors to HSV. """
        return colorsys.rgb_to_hsv(r, g, b)


    def hsv_to_rgb(self, h, s, v):
        """ Converts HSV colors to RGB. """
        return colorsys.hsv_to_rgb(h, s, v)


    def plot_colorwheel(self):
        """
        Plots a color wheel for a given RGB map.
        """

        theta = np.linspace(0,2*np.pi, len(self.colors))
        r, res = 1.0, 100
        x0, y0 = 0, 0

        for i in range(len(theta)-1):
            t = (theta[i+1] - theta[i]) * np.linspace(theta[i], theta[i+1], res) + theta[i]
            x = x0 + r*np.cos(t)
            y = y0 + r*np.sin(t)
            plt.plot(x, y, '.', c=self.colors[i])

        plt.xlim(-1.5,1.5)
        plt.ylim(-1.5,1.5)
        plt.show()
