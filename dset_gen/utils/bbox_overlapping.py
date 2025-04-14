def calculate_volume(bbox):
    """ Calculate the volume of a bounding box """
    width = bbox[1][0] - bbox[0][0]
    height = bbox[1][1] - bbox[0][1]
    depth = bbox[1][2] - bbox[0][2]
    return max(width, 0) * max(height, 0) * max(depth, 0)

def calculate_intersection(bbox1, bbox2):
    """ Calculate the intersection of two bounding boxes """
    xmin = max(bbox1[0][0], bbox2[0][0])
    ymin = max(bbox1[0][1], bbox2[0][1])
    zmin = max(bbox1[0][2], bbox2[0][2])
    xmax = min(bbox1[1][0], bbox2[1][0])
    ymax = min(bbox1[1][1], bbox2[1][1])
    zmax = min(bbox1[1][2], bbox2[1][2])
    return [[xmin, ymin, zmin], [xmax, ymax, zmax]]

def has_significant_overlap(bbox1, bbox2, threshold=0.8):
    """ Check if two bounding boxes have significant overlap """
    intersection = calculate_intersection(bbox1, bbox2)
    volume_intersection = calculate_volume(intersection)
    volume_union = calculate_volume(bbox1) + calculate_volume(bbox2) - volume_intersection

    # Check for zero division and calculate overlap
    if volume_union == 0:
        return False, None

    overlap = volume_intersection / volume_union

    if overlap > threshold:
        return True, bbox1 if calculate_volume(bbox1) > calculate_volume(bbox2) else bbox2
    else:
        return False, None