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
            c = cmap(i)[:3]
            colors.append(mpl.colors.rgb2hex(c))

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


    def get_ind(self, x, colors):
        """
        Gets corresponding color to input angle.
        """
        inds = np.zeros(len(x), dtype=int)

        for i in range(len(x)):
            inds[i] = np.where(np.rad2deg(self.theta) >= x[i])[0][0]

        if colors is None:
            return self.colors[inds]
        else:
            return colors[inds]
        
    def complementary(self, x, colors=None):
        """ 
        Pulls complementary color from a given color. 

        Parameters
        ----------
        x : float
           Angle, in radians, to map onto colorwheel.
        """
        ang = self.check(np.abs(x+180))
        hex_vals = self.get_ind([x, ang], colors=colors)
        degrees = [x, ang]
        return hex_vals, degrees


    def split_complementary(self, x, colors=None):
        """ 
        Pulls split complementary colors from a given color. 

        Parameters      
        ----------    
        x : float
           Angle, in degrees, to map onto colorwheel.  
        """
        x1 = self.check(np.abs(x+150))
        x2 = self.check(np.abs(x+210))
        hex_vals = self.get_ind([x, x1, x2], colors=colors)
        degrees = [x, x1, x2]
        return hex_vals, degrees


    def triadic(self, x, colors=None):
        """ 
        Pulls triadic colors from a given color. 

        Parameters
        ----------
        x : float
           Angle, in radians, to map onto colorwheel.
        """
        x1 = self.check(np.abs(x+120))
        x2 = self.check(np.abs(x+240))
        hex_vals = self.get_ind([x, x1, x2], colors=colors)
        degrees = [x, x1, x2]
        return hex_vals, degrees

    
    def tetradic(self, x, colors=None):
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
        hex_vals = self.get_ind(degrees, colors=colors)
        return hex_vals, degrees


    def analagous(self, x, colors=None):
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
        hex_vals = self.get_ind(degrees, colors=colors)
        return hex_vals, degrees


    def rgb_to_hsv(self, r, g, b):
        """ Converts RGB colors to HSV. """
        return colorsys.rgb_to_hsv(r, g, b)

    def hsv_to_rgb(self, h, s, v):
        """ Converts HSV colors to RGB. """
        return colorsys.hsv_to_rgb(h, s, v)

    def rgb_to_hex(self, rgb):
        """ Converts RGB to hex values. """
        return mpl.colors.rgb2hex(rgb)

    def alter_hsv(self, hue=1.0, saturation=1.0, value=1.0):
        """
        Converts RGB to HSV and back to alter values such as hue, saturation, 
        and value. Arguments range between 0 and 1. 

        Parameters
        ----------
        hue : float, optional
           Changes the hue of the colors. Default is 1.
        saturation : float, optional
           Changes the saturation of the colors. Default is 1.
        value : float, optional
           Changes the value of the colors. Default is 1.

        Attributes
        ----------
        new_rgb : np.ndarray
           Array of RGB colors based on new HSV quantities.
        """
        new_rgb = np.zeros(self.rgb.shape)

        for i in range(len(self.rgb)):
            hsv = self.rgb_to_hsv(*self.rgb[i])
            rgb = self.hsv_to_rgb(hsv[0]*hue, 
                                  hsv[1]*saturation, 
                                  hsv[2]*value)
            new_rgb[i] = rgb

            
        cmap_seq = LinearSegmentedColormap.from_list('sequential', new_rgb, N=self.N)

        cmap = cm.get_cmap(cmap_seq, self.N)
        colors = []
        for i in range(cmap.N):
            c = cmap(i)[:3]
            colors.append(mpl.colors.rgb2hex(c))

        self.new_rgb = np.array(colors)


    def make_colormap(self, hex_values, sort=False, add_bnw=False):
        """
        Creates a matplotlib color map from a list of hex values.

        Parameters
        ----------
        hex_values : np.ndarray
           Array of hex values to turn into a matplotlib colormap.
        sort : bool, optional
           Sorts the colors from red to blue. Default is False.
        add_bnw : bool, optional
           Adds black and white to each end of the colormap. 
           Default is False.

        Returns
        -------
        cm : matplotlib.colors.LinearSegmentedColormap
        """
        if sort:
            # implement stuff
            raise ValueError('This lil treat is not yet implemented')
        if add_bnw:
            hex_values = np.append('#FFFFFF', hex_values)
            hex_values = np.append(hex_values, '#000000')

        cm = LinearSegmentedColormap.from_list('sequential',
                                               hex_values,
                                               N=2048)
        return cm
        

    def plot_colorwheel(self, colors=None):
        """
        Plots a color wheel for a given RGB map.

        Parameters
        ----------
        colors : np.ndarray, optional
        """
        if colors is None:
            colors = self.colors

        theta = np.linspace(0,2*np.pi, len(colors))
        r, res = 1.0, 100
        x0, y0 = 0, 0

        for i in range(len(theta)-1):
            t = (theta[i+1] - theta[i]) * np.linspace(theta[i], theta[i+1], res) + theta[i]
            x = x0 + r*np.cos(t)
            y = y0 + r*np.sin(t)
            plt.plot(x, y, '.', c=colors[i])

        plt.xlim(-1.5,1.5)
        plt.ylim(-1.5,1.5)
        plt.show()
