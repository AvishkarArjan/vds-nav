import pybullet as p
import pybullet_data
import time

def create_box(center, size, color=[0.7, 0.7, 0.7, 1]):
    half_extents = [size[0]/2, size[1]/2, size[2]/2]
    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=center
    )

def main():
    # Start PyBullet with GUI
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Basic lighting and gravity
    p.setGravity(0, 0, -9.8)

    # Hallway parameters from specification
    hallway_length = 10.0
    hallway_width = 1.0
    hallway_height = 2.0

    print("Building hallway...")

    # 1. Floor (Greenish)
    create_box(
        center=[hallway_length/2, 0.0, 0.0],
        size=[hallway_length, hallway_width, 0.1],
        color=[0.3, 0.8, 0.3, 1.0] 
    )
    
    # 2. Left Wall (Blue)
    create_box(
        center=[hallway_length/2, -hallway_width/2 - 0.05, hallway_height/2],
        size=[hallway_length, 0.1, hallway_height],
        color=[0.2, 0.2, 0.8, 1.0] 
    )
    
    # 3. Right Wall (Blue)
    create_box(
        center=[hallway_length/2, hallway_width/2 + 0.05, hallway_height/2],
        size=[hallway_length, 0.1, hallway_height],
        color=[0.2, 0.2, 0.8, 1.0] 
    )
    
    # 4. Obstacles (Red)
    obstacles = [
        {'pos': [2.0, 0.0, 1.0], 'size': [0.2, 0.2, 2.0]},
        {'pos': [4.5, -0.2, 1.2], 'size': [0.3, 0.3, 2.0]},
        {'pos': [7.0, 0.3, 0.8], 'size': [0.4, 0.4, 2.0]}
    ]
    for obs in obstacles:
        create_box(
            center=obs['pos'], 
            size=obs['size'], 
            color=[0.9, 0.1, 0.1, 1.0]
        )

    # Adjust Camera so the user can see it perfectly
    p.resetDebugVisualizerCamera(
        cameraDistance=5.0, 
        cameraYaw=-45.0, 
        cameraPitch=-30.0, 
        cameraTargetPosition=[hallway_length/2, 0, hallway_height/2]
    )

    print("Hallway generated! You can spin the camera to look around. Close the Pybullet window to exit.")
    
    # Keep simulation running
    while p.isConnected():
        try:
            p.stepSimulation()
            time.sleep(1./240.)
        except pybullet.error:
            break
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
