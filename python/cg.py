
import cv
import math
import im

def circumcenter(x0,y0,x1,y1,x2,y2):
    x0=float(x0)
    x1=float(x1)
    x2=float(x2)
    y0=float(y0)
    y1=float(y1)
    y2=float(y2)
    nu = (y0*x2**2 + y0*y2**2 + y1*x0**2 + \
          y1*y0**2 + y2*x1**2 + y2*y1**2 - \
          y0*x1**2 - y0*y1**2 - y1*x2**2 - \
          y1*y2**2 - y2*x0**2 - y2*y0**2)
    de = (-2*x0*y2 - 2*x1*y0 - 2*x2*y1 + 2*x0*y1 + 2*x1*y2 + 2*x2*y0)
    assert de != 0.0
    X = nu/de

    nu = (x0*x1**2 + x0*y1**2 + x1*x2**2 + \
          x1*y2**2 + x2*x0**2 + x2*y0**2 - \
          x0*x2**2 - x0*y2**2 - x1*x0**2 - \
          x1*y0**2 - x2*x1**2 - x2*y1**2)
    de = (-2*x0*y2 - 2*x1*y0 - 2*x2*y1 + 2*x0*y1 + 2*x1*y2 + 2*x2*y0)
    assert de != 0.0
    Y = nu/de
    return X,Y

def within_rect(rect, x, y):
    l,t,r,b = rect
    return l < x < r and t < y < b

def rects_intersection(rects, maxwidth, maxheight):
    if not rects:
        return
    intersection = cv.CreateImage((maxwidth,
                         maxheight),
                         cv.IPL_DEPTH_8U,1)
    cv.FillPoly(intersection, [rects[0]], im.color.blue)
    for r in rects:
        canvas = cv.CreateImage((maxwidth,
                         maxheight),
                         cv.IPL_DEPTH_8U,1)
        cv.FillPoly(canvas, [r], im.color.blue)
        cv.And(canvas, intersection, intersection)
    return im.find_contour(intersection)

def has_intersection(amap, apoly, maxwidth, maxheight):
    polymap = cv.CreateImage((maxwidth,
                         maxheight),
                         cv.IPL_DEPTH_8U,1)
    cv.FillPoly(polymap, [apoly], im.color.blue)
    intersection = cv.CreateImage((maxwidth,
                         maxheight),
                         cv.IPL_DEPTH_8U,1)
    cv.And(polymap, amap, intersection)
    m=cv.Moments(cv.GetMat(intersection), True)
    return bool(cv.GetSpatialMoment(m, 0, 0))

def sub_intersection(amap, apoly, maxwidth, maxheight):
    polymap = cv.CreateImage((maxwidth,
                         maxheight),
                         cv.IPL_DEPTH_8U,1)
    cv.FillPoly(polymap, [apoly], im.color.blue)
    intersection = cv.CreateImage((maxwidth,
                         maxheight),
                         cv.IPL_DEPTH_8U,1)
    cv.And(polymap, amap, intersection)
    cv.Sub(amap, intersection, amap)

if __name__ == '__main__':
    print circumcenter(0.0,0.0,3.0,0.0,0.0,4.0)
    print circumcenter(1.0,0.0,-1.0,0.0,0.0,math.sqrt(3.0))
    print circumcenter(0.0,0.0,0.0,8.0,4.0,2.0)
