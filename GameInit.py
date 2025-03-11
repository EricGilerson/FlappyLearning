import pygame


def createPyGame():
    background_colour = (0,0,0)
    screen = pygame.display.set_mode((900, 900))
    pygame.display.set_caption('Flappy Bird')
    screen.fill(background_colour)
    pygame.display.flip()
    running = True
    pipes = pygame.image.load('Images/pipes.png').convert_alpha()

    # game loop
    while running:

        # for loop through the event queue
        for event in pygame.event.get():
            screen.blit(pipes, (0, 0))
            pygame.display.flip()

            # Check for QUIT event
            if event.type == pygame.QUIT:
                running = False