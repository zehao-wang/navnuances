def on_segment(p, q, r):
    """Given three colinear points p, q, and r, check if point q lies on line segment 'pr'"""
    if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
        q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
        return True
    return False

def orientation(p, q, r):
    """Find orientation of ordered triplet (p, q, r).
    Returns 0 if p, q and r are colinear, 1 if clockwise, 2 if counterclockwise"""
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0: return 0  # colinear
    return 1 if val > 0 else 2  # clock or counterclock wise

def do_intersect(p1, q1, p2, q2):
    """Check if line segments 'p1q1' and 'p2q2' intersect."""
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if (o1 != o2 and o3 != o4):
        return True

    # Special Cases
    # p1, q1 and p2 are colinear and p2 lies on segment p1q1
    if (o1 == 0 and on_segment(p1, p2, q1)): return True
    # p1, q1 and q2 are colinear and q2 lies on segment p1q1
    if (o2 == 0 and on_segment(p1, q2, q1)): return True
    # p2, q2 and p1 are colinear and p1 lies on segment p2q2
    if (o3 == 0 and on_segment(p2, p1, q2)): return True
    # p2, q2 and q1 are colinear and q1 lies on segment p2q2
    if (o4 == 0 and on_segment(p2, q1, q2)): return True

    return False  # Doesn't fall in any of the above cases

def is_inside_box(point, box):
    """Check if a point is inside a given box defined by two corner points."""
    if (point[0] >= min(box[0][0], box[1][0]) and point[0] <= max(box[0][0], box[1][0]) and
        point[1] >= min(box[0][1], box[1][1]) and point[1] <= max(box[0][1], box[1][1])):
        return True
    return False

def check_intersection(line, box):
    """Check if a line segment intersects with a given box."""
    # Check if either of the endpoints of the line is inside the box
    line_pt0 = line[0][:2]
    line_pt1 = line[1][:2]
    line_height_center = (line[0][2] + line[1][2]) / 2
    box_height_center = (box[0][2] + box[1][2]) / 2

    # NOTE: check bbox height and line center height not exceeding a threshold
    if abs(line_height_center - box_height_center) > 2:
        return False
    
    # NOTE: if the point inside the door bounding box, also consider as has intersection with the path
    # if is_inside_box(line_pt0, box) or is_inside_box(line_pt1, box):
    #     return True

    # Coordinates of the bounding box
    box_top_left = box[0][:2]
    box_bottom_right = box[1][:2]
    box_top_right = [box_bottom_right[0], box_top_left[1]]
    box_bottom_left = [box_top_left[0], box_bottom_right[1]]

    # TODO: intersect even if intersect with two face.
    faces_intersect = [False, False, False, False]

    # Check for intersection with each side of the bounding box
    if do_intersect(line_pt0, line_pt1, box_top_left, box_top_right): 
        faces_intersect[0] = True
    if do_intersect(line_pt0, line_pt1, box_top_right, box_bottom_right): 
        faces_intersect[1] = True
    if do_intersect(line_pt0, line_pt1, box_bottom_right, box_bottom_left):
        faces_intersect[2] = True
    if do_intersect(line_pt0, line_pt1, box_bottom_left, box_top_left): 
        faces_intersect[3] = True
    
    if faces_intersect == [True, False, True, False]:
        return True

    if faces_intersect == [False, True, False, True]:
        return True

    return False  # No intersection with any sides