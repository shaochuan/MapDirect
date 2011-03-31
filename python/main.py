'''
    A System That Reads Columbia Campus Map for You.

    @date: Mar. 29, 2011
    @author: Shao-Chuan Wang (sw2644 at columbia.edu)
'''
import os
import sys
import cv
import numpy
import im


class ImageWriter(object):
    def __init__(self, output_folder='.'):
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        self.id_ = 1
    def write(self, bgrimg):
        fullpath = self.output_folder + os.sep + '%.3d.png' % (self.id_,)
        cv.SaveImage(fullpath, bgrimg)
        self.id_ += 1

def get_map_filename():
    if len(sys.argv) > 1 and '-map' in sys.argv:
        i = sys.argv.index('-map')
        return sys.argv[i+1]
    else:
        return 'ass3-campus.pgm'

def get_label_filename():
    if len(sys.argv) > 1 and '-label' in sys.argv:
        i = sys.argv.index('-label')
        return sys.argv[i+1]
    else:
        return 'ass3-labeled.pgm'

def get_label_text_filename():
    if len(sys.argv) > 1 and '-labeltext' in sys.argv:
        i = sys.argv.index('-label')
        return sys.argv[i+1]
    else:
        return 'ass3-table.txt'

def print_instructions():
    instructions = '''
    Usage:  
    
    $ python main.py -map <map_filename> -label <label_filename> -labeltext <table_text_filename>
    '''
    if '-h' in sys.argv or '-help' in sys.argv:
        print instructions
        exit(0)

def parse_label_text_map(filename):
    fp = open(filename)
    d = {}
    while True:
        line = fp.readline()
        if not line:
            break
        k,v = line.split('=')
        v = v.strip()
        if v[0] == '"':
            v = v[1:]
        if v[-1] == '"':
            v = v[:-1]
        d[int(k)] = v.strip()
    fp.close()
    return d

import pdb
class WindowManager(object):
    msdelay = 3
    def __init__(self, win_name):
        self.map_filename = get_map_filename()
        self.label_filename = get_label_filename()
        self.label_text_map = parse_label_text_map(get_label_text_filename())
        self.win_name = win_name
        self.proc_win = cv.NamedWindow(win_name, 1)
        self.map_img = cv.LoadImage(self.map_filename, 
                iscolor=cv.CV_LOAD_IMAGE_UNCHANGED)
        self.label_img = cv.LoadImage(self.label_filename,
                iscolor=cv.CV_LOAD_IMAGE_GRAYSCALE)
        self.show_img = cv.CreateImage((self.map_img.width*2,
                self.map_img.height), cv.IPL_DEPTH_8U, 3)
        im.paste(im.clone(self.map_img, "gray2bgr"), self.show_img, 0, 0)
        cv.SetMouseCallback(win_name, self.handle_mouse)

        self.font = im.font.small
        self.fontcolor = im.color.darkgreen


    def draw_selected(self, x, y):
        cv.FloodFill(self.show_img, (x,y), im.color.red)
        b,g,r = im.split3(self.show_img)
        selected = im.newgray(r)
        cv.Sub(r, b, selected)
        contour = im.find_contour(selected)

        X,Y = im.center_of_mass(contour)
        encode_int = cv.Get2D(self.label_img, y, x)[0]
        area = cv.ContourArea(contour)
        bleft, btop, bw, bh = cv.BoundingRect(contour)
        bright = bleft + bw
        bbottom = btop + bh

        # draw
        cv.Rectangle(self.show_img, (bleft, btop), (bright, bbottom),
                color=im.color.yellow,
                thickness=2)
        cv.DrawContours(self.show_img, contour, im.color.blue, im.color.green, 1, thickness=2)

        im.drawtext(self.show_img, 
                    ("(" + "%g, " * 5) % (
                        float(x),float(y),float(encode_int),X,Y,),
                    x+10,y+10,
                    font=self.font,
                    color=self.fontcolor)
        im.drawtext(self.show_img, ("%g, " * 5 + "%s)") % (
                        area, bleft, btop, bright, bbottom,self.label_text_map[int(encode_int)]),
                    x+10,y+40,
                    font=self.font,
                    color=self.fontcolor)

    def dump_image(self):
        i = 1
        while os.path.exists("%d.png" % (i,)):
            i += 1
        cv.SaveImage("%d.png" % (i,), self.show_img)

    def handle_mouse(self, event, x, y, flag, param):
        """ Return True if this function handle the event
                   False otherwise.
        """
        if flag == cv.CV_EVENT_FLAG_LBUTTON and \
                event == cv.CV_EVENT_LBUTTONUP:
            self.show_img = cv.CreateImage((self.map_img.width*2,
                self.map_img.height), cv.IPL_DEPTH_8U, 3)
            im.paste(im.clone(self.map_img, "gray2bgr"), self.show_img, 0, 0)
            if any( cv.Get2D(self.show_img, y, x) ):
                self.draw_selected(x, y)
            else:
                im.drawtext(self.show_img, "(%g, %g, 0)" % (x,y), 
                        x+10, y+10,
                        font=self.font,
                        color=self.fontcolor)
            cv.Circle(self.show_img, (x,y), 2, im.color.green)
            return True
        return False

    def handle_keyboard(self, key):
        ''' return True if handle the key else False.'''
        if key == 'q':
          return True
        return False

    def run(self):
        while True:
            k = cv.WaitKey(self.msdelay)
            k = chr(k) if k > 0 else 0
            if self.handle_keyboard(k):
                break
            cv.ShowImage(self.win_name, self.show_img)

if __name__=='__main__':
    print_instructions()
    win_manager = WindowManager('Main Window')
    win_manager.run()
