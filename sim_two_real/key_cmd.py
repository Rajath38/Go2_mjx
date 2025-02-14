import pygame
import numpy as np
from inter_process_com import publisher as pub

# Initialize publisher
PJ = pub.publish_cmd()

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((400, 300))  # Window to capture input
pygame.display.set_caption("Velocity Controller")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)

# Velocity limits
X_LIMIT = 1.0
Y_LIMIT = 1.0
YAW_LIMIT = 1.0

# Velocity increments
X_STEP = 0.05
Y_STEP = 0.05
YAW_STEP = 0.05

# Initialize velocities
x, y, yaw = 0.0, 0.0, 0.0

running = True
while running:
    screen.fill((0, 0, 0))  # Clear screen
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    # Update x velocity
    if keys[pygame.K_UP]:
        x = min(x + X_STEP, X_LIMIT)
    if keys[pygame.K_DOWN]:
        x = max(x - X_STEP, -X_LIMIT)
    
    # Update y velocity
    if keys[pygame.K_RIGHT]:
        y = min(y + Y_STEP, Y_LIMIT)
    if keys[pygame.K_LEFT]:
        y = max(y - Y_STEP, -Y_LIMIT)
    
    # Update yaw velocity
    if keys[pygame.K_a]:
        yaw = max(yaw - YAW_STEP, -YAW_LIMIT)
    if keys[pygame.K_d]:
        yaw = min(yaw + YAW_STEP, YAW_LIMIT)
    
    # Publish command
    PJ.set(np.array([x, y, yaw]))
    
    # Render text on screen
    text = font.render(f"X: {x:.2f}  Y: {y:.2f}  Yaw: {yaw:.2f}", True, (255, 255, 255))
    screen.blit(text, (50, 130))
    
    pygame.display.flip()  # Update display
    
    # Control update rate
    clock.tick(30)  # 30 FPS

pygame.quit()