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
import cPickle
import networkx as nx

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


def segment_vline(contour, v):
    ''' return True if contour intersects with vertical line (x = v)
    '''
    for x0, y0 in list( contour ):
        for x1, y1 in list( contour ):
            if x0==x1:
                continue
            alpha = (float(v)-float(x1))/(float(x0)-float(x1))
            if 0 <= alpha <= 1:
                return True
    return False

def segment_hline(contour, h):
    ''' return True if contour intersects with horizontal line (y = h)
    '''
    for x0, y0 in list( contour ):
        for x1, y1 in list( contour ):
            if y0==y1:
                continue
            alpha = (float(h)-float(y1))/(float(y0)-float(y1))
            if 0 <= alpha <= 1:
                return True
    return False


def ns_gen(_type):
    """ generate a method for 'north' or 'south'
        to reduce the code duplication.
    """
    def inner(self, building):
        left, top, right, bottom = self.bbox
        for x, y in list( building.contour ):
            if _type == 'north':
                if y >= self.center_of_mass[1]:
                    return False
            elif _type == 'south':
                if y <= self.center_of_mass[1]:
                    return False
            else:
                raise ValueError("No such direction value.")
        if (left <= building.center_of_mass[0] <= right):
            return True
        if building.center_of_mass[0] < left:
            return segment_vline(building.contour, left)
        else:
            return segment_vline(building.contour, right)
    return inner
def ew_gen(_type):
    """ generate a method for 'east' or 'west'
        to reduce the code duplication.
    """
    def inner(self, building):
        left, top, right, bottom = self.bbox
        for x, y in list( building.contour ):
            if _type == 'east':
                if x <= self.center_of_mass[0]:
                    return False
            elif _type == 'west':
                if x >= self.center_of_mass[0]:
                    return False
            else:
                raise ValueError("No such direction value.")
        if (top <= building.center_of_mass[1] <= bottom):
            return True
        if building.center_of_mass[1] < top:
            return segment_hline(building.contour, top)
        else:
            return segment_hline(building.contour, bottom)
    return inner

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
        self.near_set = set([])
        self.north_set = set([])
        self.south_set = set([])
        self.east_set = set([])
        self.west_set = set([])

    def __eq__(self, building):
        return self.bid == building.bid

    def __repr__(self):
        return "<Building: %s @ %g, %g>" % (self.name,
                        self.center_of_mass[0],
                        self.center_of_mass[1])
    def __hash__(self):
        return self.bid

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
        for x, y in list( building.contour ):
            if l_bound <= x <= r_bound and t_bound <= y <= b_bound:
                return True
        return False

    is_north_me = ns_gen('north')
    is_south_me = ns_gen('south')
    is_east_me = ew_gen('east')
    is_west_me = ew_gen('west')


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
                rb.near_set.add(sb)
            if rb.is_north_me(sb):
                rb.north_set.add(sb)
            if rb.is_south_me(sb):
                rb.south_set.add(sb)
            if rb.is_east_me(sb):
                rb.east_set.add(sb)
            if rb.is_west_me(sb):
                rb.west_set.add(sb)

    return buildings


def prune(buildings, direction):
    if not direction in ('north', 'south', 'east', 'west'):
        raise ValueError('%s No such direction value' % (direction,))
    sa = '%s_set' % (direction,)

    G = nx.DiGraph()
    buildings.sort(cmp=lambda a,b:cmp(len(getattr(a, sa)),
            len(getattr(b, sa))),reverse=True)
    for bd in buildings:
        lst = list(getattr(bd, sa))
        lst.sort(cmp=lambda a,b:cmp(len(getattr(a, sa)), # bestfit
            len(getattr(b, sa))),reverse=True)
        for n in lst:
            if len(getattr(n, sa)) and \
                    getattr(n, sa).issubset(getattr(bd, sa)):
                print n.name
                bd.north_set = bd.north_set - n.north_set
                setattr(bd, sa, getattr(bd, sa) - getattr(n, sa))
                G.add_edge(bd.bid, n.bid)
    return G

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
        self.north_graph = prune(self.buildings, 'north')
        self.south_graph = prune(self.buildings, 'south')
        self.east_graph = prune(self.buildings, 'east')
        self.west_graph = prune(self.buildings, 'west')
        self.dump_graphs()

    def dump_graphs(self):
        for d in ('north', 'south', 'east', 'west'):
            fn = '%s_graph.g' % (d,)
            fout = open(fn, 'wb')
            cPickle.dump(getattr(self, '%s_graph' % (d,)), fout)
            fout.close()

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
                for n in bd.north_set:
                    cv.DrawContours(self.show_img, n.contour, im.color.red, im.color.green, 0, thickness=2)

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
