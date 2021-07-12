# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division

import array
from collections import namedtuple
from PIL import Image, ImageFilter

import sys
if sys.version_info[0] <= 2:
    range = xrange
    ARRAY_DATATYPE = b'l'
else:
    ARRAY_DATATYPE = 'l'

Rgb = namedtuple('Rgb', ('r', 'g', 'b'))
Hsl = namedtuple('Hsl', ('h', 's', 'l'))

class Color(object):
    def __init__(self, r, g, b, proportion):
        # linear to sRGB - 2.2 Gamma
        #self.rgb = Rgb( gamma(r), gamma(g), gamma(b) )
        self.rgb = Rgb( r,g,b )
        self.proportion = proportion
    
    def __repr__(self):
        return "<colorgram.py Color: {}, {}%>".format(
            str(self.rgb), str(self.proportion * 100))

    @property
    def hsl(self):
        try:
            return self._hsl
        except AttributeError:
            self._hsl = Hsl(*hsl(*self.rgb))
            return self._hsl

def extract(f, number_of_colors):
    image = f if isinstance(f, Image.Image) else Image.open(f)
    if image.mode not in ('RGB', 'RGBA', 'RGBa'):
        image = image.convert('RGB')
    image = image.filter(ImageFilter.GaussianBlur(radius = 3))    
    samples = sample(image)
    used = pick_used(samples)
    used.sort(key=lambda x: x[0], reverse=True)
    return get_colors(samples, used, number_of_colors)

def linearize(sample):
    x = sample / 255.0
    if x >= 0.0031308:
        x = (1.055) * pow(x, (1.0/2.4)) - 0.055
    else:
        x = 12.92 * x

    return int(round(x * 255.0) )
#    return int( round( pow( sample/255.0, 1.0/2.2) * 255.0))

def gamma(sample):
    x = sample / 255.0
    if  x >= 0.04045:
        x = pow( ((x + 0.055)/(1 + 0.055)), 2.4)
    else:
        x= x / 12.92
    return int(round( x * 255.0) )
#    return int( round( pow( sample/255.0, 2.2) * 255.0))

def sample(image):
    top_two_bits = 0b11000000

    sides = 1 << 2 # Left by the number of bits used.
    cubes = sides ** 7

    samples = array.array(ARRAY_DATATYPE, (0 for _ in range(cubes)))
    width, height = image.size
    
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            # Pack the top two bits of all 6 values into 12 bits.
            # 0bYYhhllrrggbb - luminance, hue, luminosity, red, green, blue.

            r, g, b = pixels[x, y][:3]
            #h, s, l = hsl(r, g, b)

            # Standard constants for converting RGB to relative luminance.
            #Y = int(r * 0.2126 + g * 0.7152 + b * 0.0722)

            #linearize before sampling
            _r = linearize(r)
            _g = linearize(g)
            _b = linearize(b)

            okl, oka, okb = oklab(_r,_g,_b)

            #Y = linearize(Y)
            #h = linearize(h)
            #s = linearize(s)
            #l = linearize(l)
	

            # Everything's shifted into place from the top two
            # bits' original position - that is, bits 7-8.
            packed  = (_r & top_two_bits) << 4
            packed |= (_g & top_two_bits) << 2
            packed |= (_b & top_two_bits) << 0

            # Due to a bug in the original colorgram.js, RGB isn't included.
            # The original author tries using negative bit shifts, while in
            # fact JavaScript has the stupidest possible behavior for those.
            # By uncommenting these lines, "intended" behavior can be
            # restored, but in order to keep result compatibility with the
            # original the "error" exists here too. Add back in if it is
            # ever fixed in colorgram.js.

            packed |= (okl & top_two_bits) >> 2
            packed |= (oka & top_two_bits) >> 4
            packed |= (okb & top_two_bits) >> 6
            # print "Pixel #{}".format(str(y * width + x))
            # print "h: {}, s: {}, l: {}".format(str(h), str(s), str(l))
            # print "R: {}, G: {}, B: {}".format(str(r), str(g), str(b))
            # print "Y: {}".format(str(Y))
            # print "Packed: {}, binary: {}".format(str(packed), bin(packed)[2:])
            # print
            packed *= 4
            samples[packed]     += r
            samples[packed + 1] += g
            samples[packed + 2] += b
            samples[packed + 3] += 1
    return samples

def pick_used(samples):
    used = []
    for i in range(0, len(samples), 4):
        count = samples[i + 3]
        if count:
            used.append((count, i))
    return used

def get_colors(samples, used, number_of_colors):
    pixels = 0
    colors = []
    number_of_colors = min(number_of_colors, len(used))

    for count, index in used[:number_of_colors]:
        pixels += count

        color = Color(
            samples[index]     // count,
            samples[index + 1] // count,
            samples[index + 2] // count,
            count
        )

        colors.append(color)
    for color in colors:
        color.proportion /= pixels
    return colors

def map_range(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

def oklab(r, g, b):
    _r = r / 255.0
    _g = g / 255.0
    _b = b / 255.0

    l = 0.4122214708 * _r + 0.5363325363 * _g + 0.0514459929 * _b
    m = 0.2119034982 * _r + 0.6806995451 * _g + 0.1073969566 * _b
    s = 0.0883024619 * _r + 0.2817188376 * _g + 0.6299787005 * _b

    l_ = pow(l, 1.0/3.0)
    m_ = pow(m, 1.0/3.0)
    s_ = pow(s, 1.0/3.0)

    
    l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    m = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    s = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    l = int( map_range(l, 0.0, 1.0, 0, 255) )
    m = int( map_range(m, -0.233, 0.276, 0, 255) )
    s = int( map_range(s, -0.311, 0.198, 0,  255) )
    return l, m, s 


def hsl(r, g, b):
    # This looks stupid, but it's way faster than min() and max().
    if r > g:
        if b > r:
            most, least = b, g
        elif b > g:
            most, least = r, g
        else:
            most, least = r, b
    else:
        if b > g:
            most, least = b, r
        elif b > r:
            most, least = g, r
        else:
            most, least = g, b

    l = (most + least) >> 1

    if most == least:
        h = s = 0
    else:
        diff = most - least
        if l > 127:
            s = diff * 255 // (510 - most - least)
        else:
            s = diff * 255 // (most + least)
        
        if most == r:
            h = (g - b) * 255 // diff + (1530 if g < b else 0)
        elif most == g:
            h = (b - r) * 255 // diff + 510
        else:
            h = (r - g) * 255 // diff + 1020
        h //= 6
    
    return h, s, l

# Useful snippet for testing values:
# print "Pixel #{}".format(str(y * width + x))
# print "h: {}, s: {}, l: {}".format(str(h), str(s), str(l))
# print "R: {}, G: {}, B: {}".format(str(r), str(g), str(b))
# print "Y: {}".format(str(Y))
# print "Packed: {}, binary: {}".format(str(packed), bin(packed)[2:])
# print

# And on the JS side:
# var Y = ~~(img.data[i] * 0.2126 + img.data[i + 1] * 0.7152 + img.data[i + 2] * 0.0722);
# console.log("Pixel #" + i / img.channels);
# console.log("h: " + h[0] + ", s: " + h[1] + ", l: " + h[2]);
# console.log("R: " + img.data[i] + ", G: " + img.data[i + 1] + ", B: " + img.data[i + 2]);
# console.log("Y: " + Y);
# console.log("Packed: " + v + ", binary: " + (v >>> 0).toString(2));
# console.log();
