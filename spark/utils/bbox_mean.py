def bbox_mean(bbox):
    return (bbox[0]+bbox[1]+bbox[2]+bbox[3])/4

def bbox_geometric_center(bbox):
    y = (bbox[0]+bbox[2])/2
    x = (bbox[1]+bbox[3])/2
    return x,y

def averaged_bbox(bbox1,bbox2):
    y_min = (bbox1[0]+bbox2[0])/2
    x_min = (bbox1[1]+bbox2[1])/2
    y_max = (bbox1[2]+bbox2[2])/2
    x_max = (bbox1[3]+bbox2[3])/2
    return [y_min,x_min,y_max,x_max]
