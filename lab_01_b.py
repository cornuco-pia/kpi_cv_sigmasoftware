## WARNING:
# i KNOW that this code is very hard to modify, track or debug.
# it was written as-is and unfortunately i have no time right now to refactor it
# nevertheless, it WORKS!
# the main task was to understand and apply 3d transformations
# and i managed to capture the material in our lectures quite well,
# even if the code structure is garbage :)

import pygame
import numpy as np

#configs

SCREEN_SIZE = (1000, 1000)

STARTING_POINTS = np.array([
    (0.0, 0.0, 25.0),
    (-20.0, -10.0, 0.0),
    (20.0, -10.0, 0.0),
    (0.0, 0.0, 0.0)
    
], dtype=np.float64) #relative to the barycenter


POSITION_WAYPOINTS = np.array(((150, 150, 10),(150, 350, 100),(350, 350, 100),(350, 150, 10)))
SCALE_WAYPOINTS = (10, 30, 60, 40)
ROTATION_AXIS = np.array([0.0, 0.0, 1.0])
ROTATION_AXIS = ROTATION_AXIS / np.linalg.norm(ROTATION_AXIS)
ROTATION_MEASURE_VERTEX = 1 #which vertex to use to measure rotation
#make sure it's not on the rotation axis!
ROT_WAYPOINTS = (0, 90, 180, 300)

STARTING_CENTER = POSITION_WAYPOINTS[3]

FACES = [
    [0, 1, 2],
    [0, 1, 3],
    [1, 2, 3],
    [2, 0, 3]
]

def project_onto_plane(v, axis):
    return v - axis * np.dot(v, axis)

def to_hmg(points):
    """
    points - np.array of shape [num_points, 3]
    returns [num_points, 4]
    transforms the points to homogenous coordinates (simply adds a dimension along the second axis)
    """
    return np.hstack([points, np.ones((points.shape[0], 1))])

def from_hmg(points):
    """
    points - np.array of shape [num_points, 4]
    returns [num_points, 3]
    transforms the points from homogenous coordinates
    """
    return points[:, :3] / points[:, 3].reshape(-1, 1)

def project_3d_to_2d(points, focal_length=300, camera_distance=100):
    #TODO
    projected = np.zeros((points.shape[0], 2))
    points_z = points[:, 2] + camera_distance
    points_z[points_z <= 0] = 0.1
    projected[:, 0] = (points[:, 0] * focal_length) / points_z
    projected[:, 1] = (points[:, 1] * focal_length) / points_z
    return projected

def rot_matrix(axis, angle_deg):
    #uses the rodrigues formula
    axis = axis / np.linalg.norm(axis) 
    angle = np.radians(angle_deg)
    a_cos = np.cos(angle)
    a_sin = np.sin(angle)

    k = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    v_rot = a_cos * np.eye(3) + a_sin * k + (1 - a_cos) * np.outer(axis, axis)
    v_rot_hmg = np.eye(4)
    v_rot_hmg[:3, :3] = v_rot
    return v_rot_hmg
    
def scale_matrix(coef):
    return np.array([
        [coef, 0,    0,    0],
        [0,    coef, 0,    0],
        [0,    0,    coef, 0],
        [0,    0,    0,    1]
    ])

def trans_matrix(dx, dy, dz=0):
    return np.array([
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1]
    ])

def trans_to(figure_center, destination, velocity=1, epsilon=1):
    if not isinstance(destination, np.ndarray):
        destination = np.array(destination)

    direction = destination - figure_center[:3]
    if np.linalg.norm(direction) < epsilon:
        return figure_center, True
    direction = direction / np.linalg.norm(direction)
    pixel_velocity = velocity * direction
    new_center = (trans_matrix(*pixel_velocity) @ figure_center.T).T
    return new_center, False

def scale_to(points, goal, velocity=1, epsilon=0.1):
    current = np.mean(np.linalg.norm(points[:, :3], axis=1))
    if np.abs(current-goal) < epsilon:
        return points, True
    print("scale diff:", np.abs(current-goal))
    if current > goal:
        velocity *= -1
    new_points = (scale_matrix(1+velocity/240) @ points.T).T
    return new_points, False

def rot_to(points, goal, velocity=0.5, epsilon=1.0):
    current_total_rotation = get_current_rotation(points)
    diff = goal - current_total_rotation
    diff = ((diff+180) % 360) - 180 #normalize to -180 180
    if np.abs(diff) < epsilon:
        return points, True
    print("angle diff:", np.abs(diff))
    if diff < 0:
        velocity *= -1

    new_points = (rot_matrix(ROTATION_AXIS, velocity) @ points.T).T
    return new_points, False

def color_to(current, target, velocity=5):
    diff = target - current
    step = np.clip(diff, -velocity, velocity)
    new = current + step
    return new, np.all(np.abs(diff) <= velocity)

def get_current_rotation(points):
    """
    this function calculates the angle between
    vector of center and first vertex
    and the same vector at the start of the program
    by doing the common (dot product / norms product), taking the arccos.
    it checks for the half plane using the cross product's sign
    relative to the rotation axis
    """
    reference_vector = points[ROTATION_MEASURE_VERTEX][:3]
    initial_reference = STARTING_POINTS[ROTATION_MEASURE_VERTEX]

    v0 = project_onto_plane(initial_reference, ROTATION_AXIS)
    v1 = project_onto_plane(reference_vector, ROTATION_AXIS)

    n0 = np.linalg.norm(v0)
    n1 = np.linalg.norm(v1)

    v0 /= n0
    v1 /= n1
    
    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    angle_rad = np.arccos(dot)

    # sign with cross product
    cross = np.cross(v0, v1)
    if np.dot(cross, ROTATION_AXIS) < 0:
        angle_rad = -angle_rad

    return np.degrees(angle_rad) % 360

def main():
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    clock = pygame.time.Clock()

    points_hmg = to_hmg(STARTING_POINTS)
    figure_center = np.append(STARTING_CENTER, 1)

    #ANIMATION STATE VARS
    animation_phase = 0 # 0 for translation and 1 for rot + scale
    current_waypoint = 0 # 0 to len(POSITION_WAYPOINTS) - 1
    visible = True
    visibility_time_passed = 0
    visible_period = 8000 #ms
    unvisible_period = 2000
    current_border_color = np.array((100, 255, 150))
    target_border_color = np.array((200, 200, 200))

    current_bg_color = np.array((0, 155, 50))
    target_bg_color = np.array((0, 155, 50))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        screen.fill((30, 30, 30))

        #TRANSLATE
        if animation_phase == 0:
            figure_center, reached_waypoint = trans_to(figure_center, POSITION_WAYPOINTS[current_waypoint])
            if reached_waypoint:
                current_waypoint += 1
                current_waypoint = current_waypoint % (POSITION_WAYPOINTS.shape[0])
                reached_waypoint = False
                animation_phase = 1
                print("reached pos goal")
        else:
            reached_scale_waypoint = True
            #points_hmg, reached_scale_waypoint = scale_to(points_hmg, SCALE_WAYPOINTS[current_waypoint])
            points_hmg, reached_rot_waypoint = rot_to(points_hmg, ROT_WAYPOINTS[current_waypoint])
            if reached_scale_waypoint and reached_rot_waypoint:
                animation_phase = 0
                reached_rot_waypoint = False
                reached_scale_waypoint = False
                #print("reached scale goal:", SCALE_WAYPOINTS[current_waypoint])
                print("reached rot goal:", ROT_WAYPOINTS[current_waypoint])

        #RECOLOR
        current_border_color, border_goal_reached = color_to(current_border_color, target_border_color)
        current_bg_color, bgn_goal_reached = color_to(current_bg_color, target_bg_color)

        if border_goal_reached and bgn_goal_reached:
            target_border_color = np.array(np.random.randint(0, 256, size=3), dtype=np.float32)
            target_bg_color = np.array(np.random.randint(0, 256, size=3), dtype=np.float32)

        visibility_time_passed += 60
        if (visible and visibility_time_passed >= visible_period) \
        or (not visible and visibility_time_passed >= unvisible_period):
            visible = not visible
            print("switched visibility")
            visibility_time_passed = 0

        #drawing in 3d is different
        if visible:
            world_points = points_hmg + figure_center
            world_points[:, 3] = 1
            projected_2d = project_3d_to_2d(world_points)
            for face in FACES: #had to separate the loops for correct order of drawing & didn't have energy to figure out a more efficient way
                face_points = [projected_2d[i] for i in face]
                pygame.draw.polygon(screen, current_bg_color, face_points, 0)
            for face in FACES:
                face_points = [projected_2d[i] for i in face]
                pygame.draw.polygon(screen, current_border_color, face_points, 3)

        pygame.display.flip()
        clock.tick(60)
        

    pygame.quit()

if __name__ == "__main__":
    main()