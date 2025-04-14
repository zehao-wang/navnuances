import os
import json
from utils.nav_graph_loader import load_nav_graphs
from utils.line_intersect_box import check_intersection
from utils.bbox_overlapping import has_significant_overlap
from utils.sim_interactive_vm import SimInteractive
sim = SimInteractive()

OBJ_INFO_PATH_FORMATTER = "./objects/{scene_id}.json"
OUTPUT_FORMAT = "./out/jsons/vertical_movement/{scene_id}_region{region_id}_vm.json"
os.makedirs("/".join(OUTPUT_FORMAT.split('/')[:-1]), exist_ok=True)

WIDTH = 512
HEIGHT = 512

scans = os.listdir('./data/v1/scans')
graphs, paths, distances = load_nav_graphs(scans)

def filter_by_labels(G, poses, objs_meta, region_kept, obj_kept):
    object_list = []
    region_list = []
    for k,v in objs_meta.items():
        is_region = k.startswith('region-')
        cat_label = k.split('-')[-1]
        if is_region:
            if cat_label in region_kept:
                for item in v:
                    aabb = item['aabb']
                    region_list.append(
                        {
                            "color": 'blue',
                            "label": cat_label,
                            "aabb": aabb
                        }
                    )
        else:
            if cat_label in obj_kept:
                kept_list = []
                for item in v:
                    aabb = item['aabb']
                    
                    find_overlap = False
                    bbox1 = [
                        [aabb[0][0], aabb[1][0], aabb[2][0]], 
                        [aabb[0][1], aabb[1][1], aabb[2][1]], 
                    ]

                    flag = False
                    intersect_edges = set()
                    for edge in G.edges:
                        n0, n1 = edge
                        if (n0, n1) in intersect_edges or (n1, n0) in intersect_edges:
                            continue

                        has_intersection = check_intersection(
                            [list(poses[n0]), list(poses[n1])],
                            bbox1, 
                        )
                        if has_intersection:
                            intersect_edges.add((n0, n1))
                            flag = True
                    
                    if not flag:
                        continue

                    for i in range(len(kept_list)):
                        bbox2 = [
                            [kept_list[i]['aabb'][0][0], kept_list[i]['aabb'][1][0], kept_list[i]['aabb'][2][0]], 
                            [kept_list[i]['aabb'][0][1], kept_list[i]['aabb'][1][1], kept_list[i]['aabb'][2][1]], 
                        ]
                        overlap, larger_bbox = has_significant_overlap(bbox1, bbox2, threshold=0.1)
                        if overlap:
                            print('\033[1;32m [INFO]\033[0m find overlap')
                            find_overlap = True
                            kept_list[i]['aabb'] = [
                                [larger_bbox[0][0], larger_bbox[1][0]],
                                [larger_bbox[0][1], larger_bbox[1][1]],
                                [larger_bbox[0][2], larger_bbox[1][2]],
                            ]
                            kept_list[i]['intersect_edge'] = intersect_edges
                            kept_list[i]['obj_id'] = item['instance_id']
                            break

                    if not find_overlap:
                        kept_list.append({
                                "color": 'green',
                                "label": cat_label,
                                "aabb": aabb,
                                "intersect_edge": intersect_edges,
                                "obj_id": item['instance_id']
                            }
                        )

                object_list += kept_list
    return object_list, region_list

def _is_inside(pose, region_aabb):
    if (region_aabb[0][0] < pose[0] < region_aabb[0][1]) \
        and (region_aabb[1][0] < pose[1] < region_aabb[1][1]) \
        and (region_aabb[2][0] < pose[2]+1.5 < region_aabb[2][1]):
        return True

def get_vp_inside(poses, region, scene_id):
    aabb = region['aabb']
    kept_pts = []
    for vp, pose in poses.items():
        if _is_inside(pose, aabb):
            kept_pts.append((pose[0], pose[1], pose[2], vp))
            return vp
    return None

def call_interactive(scene_id, vp):
    sim.reset(scene_id=scene_id, start_vp = vp)
    state = sim.start()
    results = sim.annt
    return results, state

def gen_path_from_scan(scene_id):
    print(scene_id)
    objs_meta = json.load(open(OBJ_INFO_PATH_FORMATTER.format(scene_id=scene_id)))
    G = graphs[scene_id]
    poses = {n: (d["position"][0], d["position"][1], d["position"][2])  for n, d in G.nodes.items()}

    region_kept = ['stairs']
    obj_kept = ['stairs']

    object_list, region_list = filter_by_labels(G, poses, objs_meta=objs_meta, region_kept=region_kept, obj_kept=obj_kept)
    for i, region in enumerate(region_list):
        if os.path.exists(OUTPUT_FORMAT.format(scene_id=scene_id, region_id=i)):
            continue
        
        path_collection = [] 
        vp = get_vp_inside(poses, region, scene_id)
        if vp is None:
            continue

        path_info, flag = call_interactive(scene_id, vp)
        if flag > 0: # if want to drop the annotation
            if len(path_info['paths']) > 0:
                path_collection.append(path_info)

        if len(path_collection) > 0:
            json.dump(path_collection, open(OUTPUT_FORMAT.format(scene_id=scene_id, region_id=i), 'w'), indent=2)
        
def main():
    for scene_id in scans:
        if os.environ.get('DEBUG', False):
            if scene_id != '8WUmhLawc2A':
                # scene_id != 'pa4otMbVnkk'
                continue
        gen_path_from_scan(scene_id=scene_id)

if __name__ == '__main__':
    main()