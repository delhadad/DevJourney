from controller import Robot

# Get reference to the robot
robot = Robot()

# Get simulation step length
timeStep = int(robot.getBasicTimeStep())

# Constants
maxMotorVelocity = 4.2
num_left_dist_sensors = 4
num_right_dist_sensors = 4
right_threshold = [75, 75, 75, 75]
left_threshold = [75, 75, 75, 75]

# Get left and right wheel motors using getDevice
leftMotor = robot.getDevice("left wheel motor")
rightMotor = robot.getDevice("right wheel motor")

# Get frontal distance sensors using getDevice
dist_left_sensors = [robot.getDevice('ps' + str(x)) for x in range(num_left_dist_sensors)]
dist_right_sensors = [robot.getDevice('ps' + str(x)) for x in range(num_left_dist_sensors, 8)]

# Enable all distance sensors
for sensor in dist_left_sensors + dist_right_sensors:
    sensor.enable(timeStep)

# Disable motor PID control mode
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))

# Set ideal motor velocity
initialVelocity = 0.7 * maxMotorVelocity

# Set the initial velocity
leftMotor.setVelocity(initialVelocity)
rightMotor.setVelocity(initialVelocity)

while robot.step(timeStep) != -1:
    left_dist_sensor_values = [sensor.getValue() for sensor in dist_left_sensors]
    right_dist_sensor_values = [sensor.getValue() for sensor in dist_right_sensors]
    
    left_obstacle = any(x > y for x, y in zip(left_dist_sensor_values, left_threshold))
    right_obstacle = any(x > y for x, y in zip(right_dist_sensor_values, right_threshold))
 
    if left_obstacle:
        leftMotor.setVelocity(initialVelocity - (0.5 * initialVelocity))
        rightMotor.setVelocity(initialVelocity + (0.5 * initialVelocity))
    elif right_obstacle:
        leftMotor.setVelocity(initialVelocity + (0.5 * initialVelocity))
        rightMotor.setVelocity(initialVelocity - (0.5 * initialVelocity))
    else:
        leftMotor.setVelocity(initialVelocity)
        rightMotor.setVelocity(initialVelocity)
