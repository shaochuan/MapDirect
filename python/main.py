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
        return ''


def print_instructions():
    instructions = '''
    Usage:  
    
    $ python main.py -map <map_filename>
    '''
    if '-map' not in sys.argv:
        print instructions
        exit(0)


class WindowManager(object):
    msdelay = 3
    def __init__(self, win_name):
        self.map_filename = get_map_filename()
        self.win_name = win_name
        self.proc_win = cv.NamedWindow(win_name, 1)
        self.map_img = cv.LoadImage(self.map_filename, iscolor=cv.CV_LOAD_IMAGE_UNCHANGED)
        self.show_img = self.map_img
        cv.SetMouseCallback(win_name, self.handle_mouse)

    def handle_mouse(self, event, x, y, flag, param):
        """ Return True if this function handle the event
                   False otherwise.
        """
        if flag == cv.CV_EVENT_FLAG_LBUTTON and \
                event == cv.CV_EVENT_LBUTTONUP:
            self.show_img = im.clone(self.map_img, "gray2bgr")
            #print event,x,y,flag,param
       #     mat = cv.GetMat(self.show_img)
            #print cv.Get2D(self.show_img, y, x)
            #cv.Set2D(self.show_img, y, x, 128)
            #cv.Circle(self.show_img, (x,y), 2, (0,0,255))
            if any( cv.Get2D(self.show_img, y, x) ):
                cv.FloodFill(self.show_img, (x,y), (0,0,255))
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
