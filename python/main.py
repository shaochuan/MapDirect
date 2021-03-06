'''
    A System That Reads Columbia Campus Map for You.

    @date: Mar. 29, 2011
    @author: Shao-Chuan Wang (sw2644 at columbia.edu)
'''
import os
import sys
import cv
import im
import it
import cg
import math
import numpy
import cPickle
import itertools
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
    def inner(self, building_or_xy):
        left, top, right, bottom = self.bbox
        if isinstance(building_or_xy, tuple):
            x, y = building_or_xy
            expr = {'north': y < top,
                    'south': y > bottom}[_type]
            if left <= x <= right and expr:
                return True
            return False
        else:
            building = building_or_xy
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
    def inner(self, building_or_xy):
        left, top, right, bottom = self.bbox
        if isinstance(building_or_xy, tuple):
            x, y = building_or_xy
            expr = {'east': x > right,
                    'west': x < left}[_type]
            if top <= y <= bottom and expr:
                return True
            return False
        else:
            building = building_or_xy
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
        self.bid = int(bid)
        self.name = self.label_text_map[bid]
        while (ishole(contour)):
            contour = contour.h_next()
        self.contour = contour
        self.area = cv.ContourArea(contour)
        self.center_of_mass = im.center_of_mass(contour)
        l,t,w,h = cv.BoundingRect(contour)
        self.bbox = (l,t,l+w,t+h)
        self.width = w
        self.height = h
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

    def draw_contour(self, canvas, external_color=im.color.blue,
                     hole_color=im.color.green):
        cv.DrawContours(canvas, self.contour, 
                        external_color, hole_color, 1, thickness=2)

    def fillme(self, canvas, color=im.color.red):
        cv.FillPoly(canvas, [list(self.contour)], color)

    def nearest_dist(self, x, y):
        ''' Return the approximate nearest distance between (x, y) and this
            building. The distance is the minimal distance from contour
            vertices or center of mass to (x, y).
        '''
        points = list(self.contour)
        xc, yc = points[0]
        mind = (xc-x)*(xc-x) + (yc-y)*(yc-y)
        for xc,yc in points:
            d =  (xc-x)*(xc-x) + (yc-y)*(yc-y)
            if d < mind:
                mind = d
        X,Y = self.center_of_mass
        d =  (X-x)*(X-x) + (Y-y)*(Y-y)
        if d < mind:
            mind = d
        return mind

    @property
    def near_region(self):
        w = self.bbox[2]-self.bbox[0]
        h = self.bbox[3]-self.bbox[1]
        X,Y = self.center_of_mass
        l_bound = X - 0.8*w
        t_bound = Y - 0.8*h
        r_bound = X + 0.8*w
        b_bound = Y + 0.8*h
        return l_bound, t_bound, r_bound, b_bound

    @property
    def near_region_poly(self):
        l,t,r,b = self.near_region
        return ((int(l),int(t)),
         (int(r),int(t)),
         (int(r),int(b)),
         (int(l),int(b)))

    def is_near_me(self, building_or_xy):
        l_bound, t_bound, r_bound, b_bound = self.near_region
        if isinstance(building_or_xy, tuple):
            x,y = building_or_xy
            return l_bound <= x <= r_bound and t_bound <= y <= b_bound
        building = building_or_xy
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
                setattr(bd, sa, getattr(bd, sa) - getattr(n, sa))
                G.add_edge(bd.bid, n.bid)
    return G

class BuildingManager(object):
    def __init__(self, label_img):
        self.label_img = label_img
        self.buildingIDs = [x for x in numpy.unique(
                    numpy.asarray(cv.GetMat(self.label_img))) if x > 0]
        self.buildings = init_buildings(self.buildingIDs, self.label_img)
        self.id_building_dict = dict(zip(self.buildingIDs, self.buildings))
        self.north_graph = prune(self.buildings, 'north')
        self.south_graph = prune(self.buildings, 'south')
        self.east_graph = prune(self.buildings, 'east')
        self.west_graph = prune(self.buildings, 'west')
        self.dump_graphs()
        self.empty_img = self.init_empty_img()

    def init_empty_img(self):
        # calculate Union{ near_regions } for all building
        nearby_region_polys = [bd.near_region_poly for bd in self.buildings]
        all_near_regions =cv.CreateImage((self.label_img.width,
                     self.label_img.height),
                     cv.IPL_DEPTH_8U,1)
        cv.FillPoly(all_near_regions, [nearby_region_polys[0]], im.color.blue)
        for poly in nearby_region_polys:
            tmp_canvas = cv.CreateImage((self.label_img.width,
                     self.label_img.height),
                     cv.IPL_DEPTH_8U,1)
            cv.FillPoly(tmp_canvas, [poly], im.color.blue)
            cv.Or(tmp_canvas, all_near_regions, all_near_regions)

        # find the "empty" region
        empty_region = cv.CreateImage((self.label_img.width,
                     self.label_img.height),
                     cv.IPL_DEPTH_8U,1)
        cv.CmpS(all_near_regions, 0, empty_region, cv.CV_CMP_EQ)


        for ele in it.nonzero_indices(cv.GetMat(empty_region)):
            y,x = ele
            y,x = int(y),int(x)
            nearest_bd = self.get_nearest_building(x, y)
            cv.Set2D(empty_region, y, x, nearest_bd.bid)
        return empty_region

    def iter(self, bid, direction='north'):
        ''' Directional tree iterator.
            The returned iterator will traverse the dependency tree 
            and iterate all 'norths' of a building, for example.
        '''
        graph = getattr(self, '%s_graph' % (direction,))
        if graph.has_node(bid):
            for edge in graph.out_edges(bid):
                assert edge[0] == bid
                child_bid = edge[1]
                child_iter = self.iter(child_bid, direction)
                for b in child_iter:
                    yield b
        bd = self.id_building_dict[bid]
        dir_set = getattr(bd, '%s_set' % (direction, ))
        for b in dir_set:
            yield b

    def dump_graphs(self):
        for d in ('north', 'south', 'east', 'west'):
            fn = '%s_graph.g' % (d,)
            fout = open(fn, 'wb')
            cPickle.dump(getattr(self, '%s_graph' % (d,)), fout)
            fout.close()

    def get_equivalent_region_via_nearest(self, bld):
        equ_class_region = cv.CreateImage((self.label_img.width, self.label_img.height),
                          cv.IPL_DEPTH_8U, 1)
        bid = bld.bid
        cv.CmpS(self.empty_img, bid, equ_class_region, cv.CV_CMP_EQ)
        return equ_class_region

    def get_near_region_mask(self, blds):
        regions = [bd.near_region_poly for bd in blds]
        c = cg.rects_intersection(regions, self.label_img.width,
                self.label_img.height)
        # subtract buildings
        equ_class_region = cv.CreateImage((self.label_img.width,
                 self.label_img.height),
                 cv.IPL_DEPTH_8U,1)
        canvas = cv.CloneImage(self.label_img)
        cv.FillPoly(equ_class_region, [c], im.color.blue)
        cv.CmpS(canvas, 0, canvas, cv.CV_CMP_EQ)
        cv.And(equ_class_region, canvas, equ_class_region)

        # subtract near of near's neighbor
        near_of_near = set(self.buildings)
        for bd in blds:
            near_of_near = near_of_near.union( bd.near_set )
        near_of_near.difference_update( set(blds) )
        near_regions = [bd.near_region_poly for bd in near_of_near]
        for reg, bd in zip(near_regions, near_of_near):
            if cg.has_intersection(equ_class_region, reg,
                    self.label_img.width, self.label_img.height):
                cg.sub_intersection(equ_class_region, reg,
                    self.label_img.width, self.label_img.height)
        # equ_class_region -= near of near via nearest
        for bd in near_of_near:
            eq_region = self.get_equivalent_region_via_nearest(bd)
            c = im.find_contour(eq_region)
            while c:
                if cg.has_intersection(equ_class_region, list(c),
                        self.label_img.width, self.label_img.height):
                    cg.sub_intersection(equ_class_region, list(c),
                            self.label_img.width, self.label_img.height)
                c = c.h_next()

        return equ_class_region

    def get_building_by_id(self, bid):
        return self.id_building_dict[bid]

    def is_within_building(self, x, y):
        return any( cv.Get2D(self.label_img, y, x) )

    def get_nearest_building(self, x, y):
        nearests = self.get_nearest_buildings(x, y)
        return nearests[0]

    def get_nearest_buildings(self, x, y, k=3):
        def app_dist((X1,Y1), bld):
            X2,Y2=bld.center_of_mass
            return (X1-X2)*(X1-X2)+(Y1-Y2)*(Y1-Y2)
        dist_blds = [(app_dist((x,y), bld), bld) for bld in self.buildings]
        dist_blds.sort()
        dist_blds = dist_blds[:k+3]
        blds = [d[1] for d in dist_blds]
        lst = []
        for bd in blds:
            lst.append( (bd.nearest_dist(x,y), bd))
        lst.sort()
        return [bd[1] for bd in lst][:k]

    def get_near_buildings(self, x, y, exclude=None):
        candiates = set(self.buildings)
        if exclude:
            candiates.difference_update(set([exclude]))
        return [bd for bd in candiates if bd.is_near_me((x,y))]

    def get_clicked_building(self, x, y):
        encode_int = cv.Get2D(self.label_img, y, x)[0]
        building = self.id_building_dict[int(encode_int)]
        return building

    def draw_selected(self, canvas, x, y):
        building = self.get_clicked_building(x, y)
        encode_int = building.bid
        X,Y = building.center_of_mass
        area = building.area
        ble, bt, br, bb = building.bbox
        label = building.name

        # draw
        cv.Rectangle(canvas, (ble, bt), (br, bb),
                color=im.color.yellow,
                thickness=2)
        building.fillme(canvas)
        building.draw_contour(canvas)

        im.drawtext(canvas,
                    ("(" + "%g, " * 5) % (
                        float(x),float(y),float(encode_int),X,Y,),
                    x+10,y+10,
                    font=im.font.small,
                    color=im.color.darkgreen)
        im.drawtext(canvas, ("%g, " * 5 + "%s)") % (
                        area, ble, bt, br, bb, label),
                    x+10,y+40,
                    font=im.font.small,
                    color=im.color.darkgreen)

class PathGenerator(object):
    def __init__(self, buildingMgr):
        self.buildingMgr = buildingMgr

    def initial_direction(self, from_, to_):
        fx, fy = from_
        tx, ty = to_
        fx, fy = float(fx), float(fy)
        tx, ty = float(tx), float(ty)
        print from_, to_
        if tx-fx == 0.0:
            if ty-fy < 0:
                return 'north'
            else:
                return 'south'
        slope = (ty-fy)/(tx-fx)
        dist = math.sqrt((tx-fx)*(tx-fx)+(ty-fy)*(ty-fy))
        unit_vec = ((tx-fx)/dist, (ty-fy)/dist)
        dot_proc = [(-unit_vec[1], 'north'),
                    (unit_vec[0], 'east'),
                    (unit_vec[1], 'south'),
                    (-unit_vec[0], 'west')]
        dot_proc.sort()
        return dot_proc[-1][1]

    def go_get_building(self, start, start_bd, direction):
        incr = {'north': -1,
                'south': +1,
                'east':  +1,
                'west':  -1,}[direction]
        x, y = start
        if direction in ('north', 'south'):
            while not self.buildingMgr.get_near_buildings(x, y,
                    exclude=start_bd):
                y += incr
        else:
            while not self.buildingMgr.get_near_buildings(x, y,
                    exclude=start_bd):
                x += incr
        bd = self.buildingMgr.get_near_buildings(x, y, exclude=start_bd)[0]

        if direction in ('north', 'south'):
            return (bd, (x,y+incr*bd.height))
        else:
            return (bd, (x+incr*bd.width, y))


    def path_from_to(self, from_, to_):
        from_bld = self.buildingMgr.get_nearest_building(*from_)
        to_bld = self.buildingMgr.get_nearest_building(*to_)

        # the target region
        # the center of mass of the target region (as a guide)
        # get target reference buildings
        

        init_dir = self.initial_direction(from_, to_)
        from_bld, next_p = self.go_get_building(from_, from_bld, init_dir)
        print from_bld, next_p
        while from_bld != to_bld:
            next_dir = self.initial_direction(next_p, to_)
            from_bld, next_p = self.go_get_building(next_p, from_bld, next_dir)
            print from_bld, next_p


    def find_path_relations(self, from_bld, to_bld):
        pass

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

        self.click_state = 'START' # 'FROM_CLICKED', 'TO_CLICKED'
        self.last_state = 'START'
        self.last_clicked = set([])
        self.last_clickedxy = (0,0)

        self.buildingMgr = BuildingManager(self.label_img)
        self.pathGen = PathGenerator(self.buildingMgr)

    def refresh(self):
        self.show_img = cv.CreateImage((int(self.map_img.width*2.8),
                self.map_img.height), cv.IPL_DEPTH_8U, 3)
        im.paste(im.clone(self.map_img, "gray2bgr"), self.show_img, 0, 0)

    def dump_image(self):
        i = 1
        while os.path.exists("%d.png" % (i,)):
            i += 1
        cv.SaveImage("%d.png" % (i,), self.show_img)

    def handle_clicked_building(self, x, y):
        building = self.buildingMgr.get_clicked_building(x, y)
        for bd in self.buildingMgr.iter(building.bid, 'north'):
            cv.DrawContours(self.show_img, bd.contour,
                            im.color.red, im.color.green, 0, thickness=2)
        self.buildingMgr.draw_selected(self.show_img, x, y)

    def generate_position_desc(self, blds):
        def draw_text(text, y, color=im.color.yellow):
            im.drawtext(self.show_img, text, self.map_img.width+30, y,
                    font=im.font.small,
                    color=color)
        if not blds:
            raise ValueError("blds cannot be null.")
            return
        if self.click_state == 'TO_CLICKED':
            anchor = 30 + self.map_img.height/2
            draw_text("To:", anchor, im.color.red)
            anchor += 30
        else:
            anchor = 30
            draw_text("From:", anchor, im.color.red)
            anchor += 30

        if len(blds) == 1:
            ref_bld = blds[0]
            text = 'Near %s' % (ref_bld.name, )
            draw_text(text, anchor)
            anchor += 30
            for direction in ('east', 'west', 'north', 'south'):
                graph = getattr(self.buildingMgr, '%s_graph' % (direction, ))
                if not graph.has_node(ref_bld.bid):
                    continue
                for edge in graph.in_edges(ref_bld.bid):
                    node = edge[0]
                    b = self.buildingMgr.get_building_by_id(node)
                    text = '%s of %s ' % (direction.capitalize(), b.name)
                    print text
                    draw_text(text, anchor)
                    anchor += 30
        else:
            s = 'Between '
            draw_text(s, anchor)
            anchor += 30
            for bd in blds[:-1]:
                draw_text('%s, ' % (bd.name,), anchor)
                anchor += 30
            draw_text('and %s' % (blds[-1].name,), anchor)


    def handle_clicked_nonbuilding(self, x, y):
        def equivalent_region_nearest(nearest_building):
            ''' find the equivalent class region via nearest building
            '''
            equ_class_region = cv.CreateImage((self.map_img.width, self.map_img.height),
                          cv.IPL_DEPTH_8U, 1)
            bid = nearest_building.bid
            cv.CmpS(self.buildingMgr.empty_img, bid, equ_class_region, cv.CV_CMP_EQ)
            return equ_class_region
        def draw_region(equ_class_region, color=im.color.red):
            c = im.find_contour(equ_class_region)
            while c:
                cv.FillPoly(self.show_img, [list(c)], color)
                c = c.h_next()

        blds = self.buildingMgr.get_near_buildings(x,y)
        if not blds:
            blds = [self.buildingMgr.get_nearest_building(x, y)]
        # update clicked states
        if self.click_state == 'START':
            self.click_state = 'FROM_CLICKED'
        elif self.click_state == 'FROM_CLICKED':
            if self.last_clicked != set(blds):
                self.click_state = 'TO_CLICKED'
        else: # click_state == 'TO_CLICKED'
            self.click_state = 'FROM_CLICKED'

        reg_color = {'FROM_CLICKED' : im.color.red,
                     'TO_CLICKED'   : im.color.blue}[self.click_state]
        fill_bld_color = {'FROM_CLICKED' :im.color.pink,
                          'TO_CLICKED':im.color.lightblue}[self.click_state]

        # equivalent class region 
        equ_class_region = self.buildingMgr.get_near_region_mask(blds)
        if len(blds)==1:
            cv.Or(equ_class_region, equivalent_region_nearest(blds[0]),
                equ_class_region)
        draw_region(equ_class_region, color=reg_color)
        # fill nearby buildings
        for bd in blds:
            bd.fillme(self.show_img, fill_bld_color)
        self.generate_position_desc(blds)

        im.drawtext(self.show_img, "(%g, %g, 0)" % (x,y), 
                x+10, y+10,
                font=self.font,
                color=self.fontcolor)
        self.last_clicked = set(blds)


    def handle_mouse(self, event, x, y, flag, param):
        """ Return True if this function handle the event
                   False otherwise.
        """
        def screen_shot():
            #cv.SetImageROI(self.show_img, (0,0,self.map_img.width,self.map_img.height))
            i = 1
            while os.path.exists("%d.png" % (i,)):
                i += 1
            cv.SaveImage('%d.png' % (i,), self.show_img)

        is_click = (flag == cv.CV_EVENT_FLAG_LBUTTON and \
                event == cv.CV_EVENT_LBUTTONUP)
        if is_click:
            if self.click_state != 'FROM_CLICKED':
                self.refresh()
            if self.buildingMgr.is_within_building(x, y):
                self.handle_clicked_building(x, y)
            else:
                self.handle_clicked_nonbuilding(x, y)
            if self.click_state == 'TO_CLICKED':
                self.pathGen.path_from_to(self.last_clickedxy, (x,y))
                screen_shot()
            self.last_clickedxy = (x,y)

            # mark the clicked position.
            cv.Circle(self.show_img, (x,y), 2, im.color.green)
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
