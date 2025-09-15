## WARNING:
# i KNOW that this code is very hard to modify, track or debug.
# it was written as-is and unfortunately i have no time right now to refactor it
# nevertheless, it WORKS!
# the main task was to understand and apply 2d transformations
# and i managed to capture the material in our lectures quite well,
# even if the code structure is garbage :)

import pygame
import numpy as np

#configs

SCREEN_SIZE = (500, 500)
COLOR = (100, 255, 150)
FADED_COLOR = (0, 155, 50)

STARTING_POINTS = np.array([
    (-20.0, 30.0),
    (-5.0, 15.0),
    (20.0, 30.0)
    
], dtype=np.float64) #relative to the barycenter


POSITION_WAYPOINTS = ((150, 150),(150, 350),(350, 350),(350, 150))
SCALE_WAYPOINTS = (10, 30, 60, 40)
ROT_WAYPOINTS = (0, 90, 180, 300)

STARTING_CENTER = np.array(POSITION_WAYPOINTS[3])

# def get_barycenter(points):
#     """
#     points: np.array of shape [num_points, 2] or [num_points, 3]
#     returns: np.array of shape [2] or [3] representing (x, y(, z)) of centroid pf the points
#     """
#     return np.mean(points, axis=0)


def to_hmg(points):
    """
    points - np.array of shape [num_points, 2]
    returns [num_points, 3]
    transforms the points to homogenous coordinates (simply adds a dimension along the second axis)
    """
    return np.hstack([points, np.ones((points.shape[0], 1))])

def from_hmg(points):
    """
    points - np.array of shape [num_points, 3]
    returns [num_points, 2]
    transforms the points from homogenous coordinates
    """
    return points[:, :2] / points[:, 2].reshape(-1, 1)


def rot_matrix(deg, dir=1):
    """
     creates a 2d rotation matrix for n degrees in a certain direction
     dir == 1 is anti clockwise and -1 is clockwise
     returns: np.array of shape (3, 3)
    """
    assert (dir == 1 or dir == -1), "incorrect rot_matrix call"
    angle = np.radians(deg)
    angle = angle * dir
    a_cos = np.cos(angle)
    a_sin = np.sin(angle)
    return np.array([
        [a_cos, -a_sin, 0],
        [a_sin,  a_cos, 0],
        [0,      0,     1]
    ])

def scale_matrix(coef):
    """
    returns a (3, 3) np.array with coef on (0,0) and (1,1), 1 on (2,2) and zeros elsewhere
    """
    return np.array([
        [coef, 0,    0],
        [0,    coef, 0],
        [0,    0,    1]
    ])

def trans_matrix(dx, dy):
    """
    returns a 2d translation matrix for homogeneous coordinates
    """
    return np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1]
    ])

def trans_to(figure_center, destination, velocity = 1, epsilon = 1):
    """
    a wrapper for translation with defined destination and velocity
    returns new points coordinates and whether the destination has been reached
    points: np.array [num_points, 3]
    destination: tuple or any array of len 2
    velocity: scalar, pixels per step (60 fps default)
    """
    if not isinstance(destination, np.ndarray):
        destination = np.array(destination)

    #print(figure_center)
    direction = destination - figure_center[:2]
    if np.linalg.norm(direction) < epsilon:
        return figure_center, True
    direction = direction / np.linalg.norm(direction)
    pixel_velocity = velocity * direction
    new_center = (trans_matrix(*pixel_velocity) @ figure_center.T).T
    return new_center, False

def scale_to(points, goal, velocity = 1, epsilon = 0.1):
    """
    a wrapper for scaling with defined goal and velocity
    goal is the mean distance from barycenter to verteces
    points: np.array [num_points, 3]
    goal: scalar
    velocity: scaling is happening with a coef of 1 - direction * velocity/240
    returns new points and whether the goal has been reached
    """
    current = np.mean(np.linalg.norm(points[:, :2], axis=1))
    #print("current scale:", current)
    if np.abs(current - goal) < epsilon:
        return points, True
    if current > goal:
        velocity *= -1
    new_points = (scale_matrix(1 + velocity/240) @ (points).T).T
    return new_points, False

def rot_to(points, goal, velocity = 0.5, epsilon = 0.01):
    """
    a wrapper for rotation with defined goal and velocity
    goal is angle the vector consisting of first vertex and center is pointing towards
    points: np.array [num_points, 3]
    goal: scalar in degrees
    velocity: deg per step
    returns new points and whether the goal has been reached
    """
    angle_vector = points[0][:2]
    current = np.arctan2(angle_vector[1], angle_vector[0])
    goal_rad = np.radians(goal)
    
    diff = goal_rad - current
    diff = ((diff + np.pi) % (2 * np.pi)) - np.pi  # normalizing to [-π, π]
    
    if np.abs(diff) < epsilon:
        return points, True
    
    if diff < 0:
        velocity *= -1
        
    new_points = (rot_matrix(velocity, 1) @ (points).T).T
    return new_points, False

def main():
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    clock = pygame.time.Clock()

    points_hmg = to_hmg(STARTING_POINTS)
    figure_center = np.append(STARTING_CENTER, 1)

    #ANIMATION STATE VARS
    animation_phase = 0 # 0 for translation and 1 for rot + scale
    current_waypoint = 0 # 0 to len(POSITION_WAYPOINTS) - 1

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        #screen.fill((30, 30, 30))
        #TODO: trace the matrix operations carefully and maybe transpose where needed

        #figure_center = get_barycenter(points_hmg)
        #points_hmg = (scale_matrix(1 + 1/240) @ (points_hmg).T).T
        #points_hmg = (rot_matrix(0.5, 1) @ (points_hmg).T).T
        #points_hmg = (trans_matrix(1, 1) @ points_hmg.T).T

        #TRANSLATE
        if animation_phase == 0:
            figure_center, reached_waypoint = trans_to(figure_center, POSITION_WAYPOINTS[current_waypoint])
            if reached_waypoint:
                current_waypoint += 1
                current_waypoint = current_waypoint % (len(POSITION_WAYPOINTS))
                reached_waypoint = False
                animation_phase = 1
                print("reached pos goal")
        else:
            points_hmg, reached_scale_waypoint = scale_to(points_hmg, SCALE_WAYPOINTS[current_waypoint])
            points_hmg, reached_rot_waypoint = rot_to(points_hmg, ROT_WAYPOINTS[current_waypoint])
            if reached_scale_waypoint and reached_rot_waypoint:
                animation_phase = 0
                reached_rot_waypoint = False
                reached_scale_waypoint = False
                #print("reached scale goal:", SCALE_WAYPOINTS[current_waypoint])
                print("reached rot goal:", ROT_WAYPOINTS[current_waypoint])


        pygame.draw.polygon(screen, COLOR, points_hmg[:, :2] + figure_center[:2], 2)
        pygame.display.flip()
        pygame.draw.polygon(screen, FADED_COLOR, points_hmg[:, :2] + figure_center[:2], 2)
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()