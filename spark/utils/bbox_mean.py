def bbox_mean(bbox):
    return (bbox[0]+bbox[1]+bbox[2]+bbox[3])/4

def bbox_geometric_center(bbox):
    y = (bbox[0]+bbox[2])/2
    x = (bbox[1]+bbox[3])/2
    return x,y