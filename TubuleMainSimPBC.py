import sys
import pygame
import TubulePhysicsPBC

(width, height) = (400, 400)
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Tubule demo')

substrate = TubulePhysicsPBC.Substrate(width, height)

# Adds anonymous functions for the main sim loop
substrate.addFunctions(['moveJunction',
                        'updateTubule',
                        'growTubule',
                        'contractTubule',
                        'boundSubstrate',
                        'mergeTubule',
                        'wrapSubstrate'
                        ])

# Populates initial conditions
substrate.initSubstrate()

pause = False

while True:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                pause = (True, False)[pause]
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Click LMB to split a tube
            if event.button == 1:
                substrate.splitTubule()
            # Click RMB to check how many tubules
            # are in the system
            elif event.button == 3:
                print(len(substrate.tubules))

    screen.fill((255, 255, 255))

    # Removes junctions/tubules marked for deletion
    # from the system
    junctions_to_remove = []
    for j in substrate.junctions:
        if 'to_remove' in j.__dict__:
            junctions_to_remove.append(j.to_remove)
            del(j.__dict__['to_remove'])

    tubules_to_remove = []
    for t in substrate.tubules:
        if 'to_remove' in t.__dict__:
            tubules_to_remove.append(t.to_remove)
            del(t.__dict__['to_remove'])

    for j in junctions_to_remove:
        if j in substrate.junctions:
            substrate.junctions.remove(j)

    for t in tubules_to_remove:
        if t in substrate.tubules:
            substrate.tubules.remove(t)

    # Drawing... Comment out this first block if you
    # don't want to see the junctions
    for j in substrate.junctions:
        pygame.draw.circle(screen, (0, 0, 255),
                           (int(j.position[0]) % width,
                            int(j.position[1])), 3, 1)

    for t in substrate.tubules:

        if t.crossover == 0:

            pygame.draw.aaline(screen, (0, 0, 0),
                               (int(t.j1.position[0]),
                                int(t.j1.position[1])),
                               (int(t.j2.position[0]),
                                int(t.j2.position[1])))

        else:

            # Finds the correct y coordinate where the
            # tube crosses the x bound and draws two
            # lines
            #
            # There are sometimes drawing errors (not sure
            # where they're coming from) where tubules
            # will flicker across the screen, but this
            # doesn't reflect the system physics
            x_min = min(t.j1.position[0], t.j2.position[0])
            x_max = max(t.j1.position[0], t.j2.position[0])

            if x_min == t.j1.position[0]:
                y_min = t.j1.position[1]
                y_max = t.j2.position[1]
            else:
                y_min = t.j2.position[1]
                y_max = t.j1.position[1]

            dx = x_max - x_min
            dy = y_max - y_min
            f_x1 = x_min / (width - dx)
            f_x2 = (width - x_max) / (width - dx)

            f_y = y_max * f_x1 + y_min * f_x2

            pygame.draw.aaline(
                screen, (0, 0, 0),
                (0, int(f_y)), (int(x_min), int(y_min)))

            pygame.draw.aaline(
                screen, (0, 0, 0),
                (int(x_max), int(y_max)), (width, int(f_y)))

    if not pause:
        substrate.updateSubstrate()

    pygame.display.flip()
