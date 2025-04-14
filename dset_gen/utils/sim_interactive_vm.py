
import MatterSim
import math
import cv2
import numpy as np
import networkx as nx
import json
import pprint
pp = pprint.PrettyPrinter(indent=4)

WIDTH = 512
HEIGHT = 512
VFOV = math.radians(90)
HFOV = VFOV*WIDTH/HEIGHT
TEXT_COLOR = [230, 40, 40]


ANGLEDELTA = 5 * math.pi / 180

def load_nav_graphs(scans, scene_name):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        if scan != scene_name:
            continue

        with open('connectivity/%s_connectivity.json' % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs

def state_eq(state1, state2):
    return (state1[1] == state2.heading) \
        and (state1[0] == state2.location.viewpointId) \
        and (state1[2] == state2.elevation)

class SimInteractive(object):
    def __init__(self) -> None:
        sim = MatterSim.Simulator()
        sim.setCameraResolution(WIDTH, HEIGHT)
        # sim.setDiscretizedViewingAngles(True)
        sim.setCameraVFOV(VFOV)
        sim.setDepthEnabled(False) # Turn on depth only after running ./scripts/depth_to_skybox.py (see README.md)
        sim.initialize()
        self.sim = sim
        
    def reset(self, scene_id, start_vp):
        cv2.namedWindow('Python RGB')
        self.sim.newEpisode([scene_id], [start_vp], [0], [0])
        self.path = None
        self.scene_id = scene_id
        self.annt = {
            "scene_name": scene_id,
            "paths": [] # multiple paths
        }
    
    def start(self):
        view_pts = []
        heading = 0
        elevation = 0
        location = 0
        last_state = None
        last_view = None
        while True:
            self.sim.makeAction([location], [heading], [elevation])
            if self.path is not None:
                self.path.append((state.location.viewpointId, state.heading, state.elevation))

            location = 0
            heading = 0
            elevation = 0

            state = self.sim.getState()[0]
            locations = state.navigableLocations
            rgb = np.array(state.rgb, copy=True)

            ava_paths = {}
            for idx, loc in enumerate(locations[1:]):
                # Draw actions on the screen
                fontScale = 3.0/loc.rel_distance
                x = int(WIDTH/2 + loc.rel_heading/HFOV*WIDTH)
                y = int(HEIGHT/2 - loc.rel_elevation/VFOV*HEIGHT)
                cv2.putText(rgb, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale, TEXT_COLOR, thickness=3)
                ava_paths[idx + 1] = (loc.viewpointId, loc.rel_distance)
            
            if (last_state is not None) and (not state_eq(last_state, state)):
                if last_view != state.location.viewpointId:
                    last_view = state.location.viewpointId
                view_pts.append(last_state)

            if last_state is None:
                print(f"\033[1;32m [INFO]\033[0m Current node: {state.location.viewpointId}")
                for path_idx,v in ava_paths.items():
                    print(f"PATH {path_idx}", v[0])
                print('-' * 20)
                last_state = (state.location.viewpointId, state.heading, state.elevation)
                last_view = last_state[0]

            cv2.imshow('Python RGB', rgb)

            # depth = np.array(state.depth, copy=False)
            # cv2.imshow('Python Depth', depth)
            k = cv2.waitKey(1)
            if k == -1:
                continue
            else:
                k = (k & 255)
            if k == ord('p'):
                break
            elif ord('1') <= k <= ord('9'):
                location = k - ord('0')
                if location >= len(locations):
                    location = 0
            elif k == 81 or k == ord('a'):
                heading = -ANGLEDELTA
                # heading = -1
            elif k == 82 or k == ord('w'):
                elevation = ANGLEDELTA
                # heading = -6
            elif k == 83 or k == ord('d'):
                heading = ANGLEDELTA
                # heading = 1
            elif k == 84 or k == ord('s'):
                elevation = -ANGLEDELTA
                # heading = -6
            elif k == ord('u'):
                ans = input("You choose to drop this data, are u sure [Y/n]? ")
                if ans.strip() == 'Y':
                    return 0
            elif k == ord('n'):
                # NOTE: create a new path
                self.path = []
                self.path.append((state.location.viewpointId, state.heading, state.elevation))
                print('\033[1;32m [INFO]\033[0m Starting a new annotation')
            elif k == ord('b'):
                text = input(f"Please annotate [up/down]: ").strip()
                while text not in ['up', 'down']:
                    text = input(f"Invalid {text}, Please annotate [up/down]: ").strip()
                if self.path is not None:
                    self.annt['paths'].append(
                        {
                            "path": self.path,
                            "label": text+"stairs",
                        }
                    )
                
                # NOTE: stop current annotation
                print('\033[1;32m [INFO]\033[0m Stop current annotation, reset to start')
                self.sim.newEpisode([self.scene_id], [self.path[0][0]], [self.path[0][1]], [self.path[0][2]])
                self.path = None
            
        return 1

    def close(self):
        self.sim.close()
                












