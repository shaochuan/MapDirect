'''
    A System That Reads Columbia Campus Map for You.

    @date: Mar. 29, 2011
    @author: Shao-Chuan Wang (sw2644 at columbia.edu)
'''
import os
import sys
import cv
import im
import math
import numpy

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

def ishole(cvseq):
    return cvseq[1][1] - cvseq[0][1] < 0

class Building(object):
    label_text_map = parse_label_text_map(get_label_text_filename())
    def __init__(self, bid, contour):
        self.bid = bid
        self.name = self.label_text_map[bid]
        while (ishole(contour)):
            contour = contour.h_next()
        self.contour = contour
        self.area = cv.ContourArea(contour)
        self.center_of_mass = im.center_of_mass(contour)
        l,t,w,h = cv.BoundingRect(contour)
        self.bbox = (l,t,l+w,t+h)
        self.near_list = []
        self.north_list = []
        self.south_list = []
        self.east_list = []
        self.west_list = []

    def __eq__(self, building):
        return self.bid == building.bid

    def __repr__(self):
        return "<Building: %s @ %g, %g>" % (self.name,
                        self.center_of_mass[0],
                        self.center_of_mass[1])

    def __sub__(self, building):
        X, Y = self.center_of_mass
        x, y = building.center_of_mass
        return math.sqrt((X-x) * (X-x) + (Y-y) * (Y-y))

    def is_near_me(self, building):
        w = self.bbox[2]-self.bbox[0]
        h = self.bbox[3]-self.bbox[1]
        X,Y = self.center_of_mass
        l_bound = X - w
        t_bound = Y - h
        r_bound = X + w
        b_bound = Y + h
        for p in list( building.contour ):
            x, y = p
            if l_bound <= x <= r_bound and t_bound <= y <= b_bound:
                return True
        return False

def contours_iterator(cvseq):
    if cvseq and not ishole(cvseq):
        yield cvseq
    while True:
        cvseq = cvseq.h_next()
        if not cvseq:
            break
        if ishole(cvseq):
            continue
        yield cvseq

def init_buildings(buildingIDs, label_img):
    bld_mask = cv.CreateImage((label_img.width, label_img.height),
                    cv.IPL_DEPTH_8U,1)
    buildings = []
    for bid in buildingIDs:
        cv.CmpS(label_img, bid, bld_mask, cv.CV_CMP_EQ)
        buildings.append(Building(bid, im.find_contour(bld_mask)))

    for rb in buildings:
        for sb in buildings:
            if sb.bid == rb.bid:
                continue
            if rb.is_near_me(sb):
                rb.near_list.append(sb)

    return buildings

import pdb
class WindowManager(object):
    msdelay = 3
    def __init__(self, win_name):
        self.map_filename = get_map_filename()
        self.label_filename = get_label_filename()
        self.win_name = win_name
        self.proc_win = cv.NamedWindow(win_name, 1)
        self.map_img = cv.LoadImage(self.map_filename, 
                iscolor=cv.CV_LOAD_IMAGE_UNCHANGED)
        self.label_img = cv.LoadImage(self.label_filename,
                iscolor=cv.CV_LOAD_IMAGE_GRAYSCALE)
        cv.SetMouseCallback(win_name, self.handle_mouse)
        self.refresh()

        self.font = im.font.small
        self.fontcolor = im.color.darkgreen
        self.buildingIDs = [x for x in numpy.unique(
                    numpy.asarray(cv.GetMat(self.label_img))) if x > 0]
        self.buildings = init_buildings(self.buildingIDs, self.label_img)
        self.id_building_dict = dict(zip(self.buildingIDs, self.buildings))

    def refresh(self):
        self.show_img = cv.CreateImage((self.map_img.width*2,
                self.map_img.height), cv.IPL_DEPTH_8U, 3)
        im.paste(im.clone(self.map_img, "gray2bgr"), self.show_img, 0, 0)


    def draw_selected(self, x, y):
        cv.FloodFill(self.show_img, (x,y), im.color.red)
        encode_int = cv.Get2D(self.label_img, y, x)[0]
        building = self.id_building_dict[int(encode_int)]
        X,Y = building.center_of_mass
        area = building.area
        ble, bt, br, bb = building.bbox
        label = building.name

        # draw
        cv.Rectangle(self.show_img, (ble, bt), (br, bb),
                color=im.color.yellow,
                thickness=2)
        cv.DrawContours(self.show_img, building.contour, im.color.blue, im.color.green, 1, thickness=2)

        im.drawtext(self.show_img, 
                    ("(" + "%g, " * 5) % (
                        float(x),float(y),float(encode_int),X,Y,),
                    x+10,y+10,
                    font=self.font,
                    color=self.fontcolor)
        im.drawtext(self.show_img, ("%g, " * 5 + "%s)") % (
                        area, ble, bt, br, bb, label),
                    x+10,y+40,
                    font=self.font,
                    color=self.fontcolor)
        return building


    def dump_image(self):
        i = 1
        while os.path.exists("%d.png" % (i,)):
            i += 1
        cv.SaveImage("%d.png" % (i,), self.show_img)

    def handle_mouse(self, event, x, y, flag, param):
        """ Return True if this function handle the event
                   False otherwise.
        """
        def draw_clicked():
            if any( cv.Get2D(self.show_img, y, x) ):
                return self.draw_selected(x, y)
            else:
                im.drawtext(self.show_img, "(%g, %g, 0)" % (x,y), 
                        x+10, y+10,
                        font=self.font,
                        color=self.fontcolor)
            cv.Circle(self.show_img, (x,y), 2, im.color.green)

        if flag == cv.CV_EVENT_FLAG_LBUTTON and \
                event == cv.CV_EVENT_LBUTTONUP:
            self.refresh()
            bd = draw_clicked()
            if bd:
                for n in bd.near_list:
                    cv.DrawContours(self.show_img, n.contour, im.color.red, im.color.green, 0, thickness=2)
            #bs = self.sort_buildings(x,y)
            #print bs[0]

            return True
        return False

    def handle_keyboard(self, key):
        ''' return True if handle the key else False.'''
        if key == 'q':
          return True
        return False


    def sort_buildings(self, x, y):
        dists = []
        for b in self.buildings:
            X,Y = b.center_of_mass
            d = math.sqrt((X-x)*(X-x) + (Y-y)*(Y-y))
            dists.append((d, b))
        dists.sort()
        return map(lambda x: x[1], dists)

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
