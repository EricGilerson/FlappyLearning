import numpy as np
import pygame, sys, random

from train import _reward, _ensure_pipes, _PIPE_IMG


def get_next_pipes_info(bird_rect, pipe_list):
    # Sort pipes by x position
    upcoming_pipes = []
    for top, bot in pipe_list:
        if top.left > bird_rect.right:
            upcoming_pipes.append((top, bot))

    upcoming_pipes.sort(key=lambda pipe_pair: pipe_pair[0].centerx)

    result = []
    for i in range(min(2, len(upcoming_pipes))):
        top, bot = upcoming_pipes[i]
        x_distance = top.centerx - bird_rect.centerx

        gap_center = (top.bottom + bot.top) / 2
        y_distance = gap_center - bird_rect.centery

        result.append((x_distance, y_distance))

    while len(result) < 2:
        result.append((None, None))

    return result


def flappy_bird():
    pygame.init()

    bg_raw = pygame.image.load("Images/background-day.png")
    floor_raw = pygame.image.load("Images/base.png")
    bird_raw = pygame.image.load("Images/dev.png")
    pipe_raw = pygame.image.load("Images/pipe-green.png")

    SCALE = 3
    raw_w, raw_h = bg_raw.get_size()
    SCREEN_WIDTH = raw_w * SCALE * 2
    SCREEN_HEIGHT = raw_h * SCALE

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    def sc(img):
        return pygame.transform.scale(
            img, (img.get_width() * SCALE, img.get_height() * SCALE)
        )

    bg_img = sc(bg_raw).convert()
    floor_img = sc(floor_raw).convert()
    bird_img = sc(bird_raw).convert_alpha()
    pipe_img = sc(pipe_raw).convert_alpha()

    BG_W, BG_H = bg_img.get_size()
    FLOOR_W, FLOOR_H = floor_img.get_size()
    floor_y = SCREEN_HEIGHT - FLOOR_H

    gravity = 0.05 * SCALE
    jump_impulse = 2 * SCALE
    pipe_speed = 1.5 * SCALE
    pipe_gap = 100 * SCALE

    pipe_horizontal_spacing = 200 * SCALE

    bird_vel = 0
    bird_rect = bird_img.get_rect(center=(50 * SCALE, SCREEN_HEIGHT // 3))

    pipe_list = []
    scored_pipes = []
    score = 0
    floor_x = 0

    for i in range(3):
        x_pos = SCREEN_WIDTH + (i * pipe_horizontal_spacing)
        min_c = pipe_gap // 2 + 10 * SCALE
        max_c = floor_y - pipe_gap // 2 - 10 * SCALE
        cy = random.randint(min_c, max_c)
        top = pipe_img.get_rect(midbottom=(x_pos, cy - pipe_gap // 2))
        bot = pipe_img.get_rect(midtop=(x_pos, cy + pipe_gap // 2))
        pipe_list.append((top, bot))

    def create_pipe(x_position):
        min_c = pipe_gap // 2 + 10 * SCALE
        max_c = floor_y - pipe_gap // 2 - 10 * SCALE
        cy = random.randint(min_c, max_c)
        top = pipe_img.get_rect(midbottom=(x_position, cy - pipe_gap // 2))
        bot = pipe_img.get_rect(midtop=(x_position, cy + pipe_gap // 2))
        return top, bot

    def draw_pipes():
        for top, bot in pipe_list:
            screen.blit(pipe_img, bot)
            screen.blit(pygame.transform.flip(pipe_img, False, True), top)

    def ensure_consistent_pipes():
        rightmost_x = SCREEN_WIDTH

        if pipe_list:
            rightmost_pipe = max(pipe_list, key=lambda p: p[0].centerx)
            rightmost_x = rightmost_pipe[0].centerx

        if rightmost_x < SCREEN_WIDTH + pipe_horizontal_spacing:
            new_pipe = create_pipe(rightmost_x + pipe_horizontal_spacing)
            pipe_list.append(new_pipe)

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit();
                sys.exit()
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                bird_vel = -jump_impulse

        bird_vel += gravity
        bird_rect.centery += bird_vel

        for top, bot in pipe_list:
            top.x -= pipe_speed
            bot.x -= pipe_speed

        new_pipe_list = []
        for pipe_pair in pipe_list:
            top, bot = pipe_pair
            if top.right > -50 * SCALE:
                new_pipe_list.append(pipe_pair)
            else:
                if pipe_pair in scored_pipes:
                    scored_pipes.remove(pipe_pair)

        pipe_list = new_pipe_list

        ensure_consistent_pipes()

        if bird_rect.top <= 0 or bird_rect.bottom >= floor_y:
            print("Game Over – score:", score)
            pygame.quit();
            sys.exit()

        for top, bot in pipe_list:
            if bird_rect.colliderect(top) or bird_rect.colliderect(bot):
                print("Game Over – score:", score)
                pygame.quit();
                sys.exit()

        for pipe_pair in pipe_list:
            top, bot = pipe_pair
            if pipe_pair not in scored_pipes and bird_rect.centerx > top.centerx:
                score += 1
                scored_pipes.append(pipe_pair)

        screen.blit(bg_img, (0, 0))
        screen.blit(bg_img, (BG_W, 0))
        draw_pipes()
        screen.blit(bird_img, bird_rect)

        floor_x -= pipe_speed
        for i in range(3):
            screen.blit(floor_img, (floor_x + i * FLOOR_W, floor_y))
        if floor_x <= -FLOOR_W:
            floor_x += FLOOR_W

        txt = pygame.font.SysFont(None, 48).render(str(score), True, (255, 255, 255))
        screen.blit(txt, (10, 10))

        pipe_info = get_next_pipes_info(bird_rect, pipe_list)
        y_offset = 50
        for i, (x_dist, y_dist) in enumerate(pipe_info):
            if x_dist is not None:
                info_text = f"Pipe {i + 1}: x={int(x_dist)}, y={int(y_dist)}"
            else:
                info_text = f"Pipe {i + 1}: Not available"
            pipe_txt = pygame.font.SysFont(None, 36).render(info_text, True, (255, 255, 255))
            screen.blit(pipe_txt, (10, y_offset))
            y_offset += 30

        pygame.display.update()
        clock.tick(120)


class FlappyBirdEnv:
    ACTION_SPACE = 2  # 0 = no-flap, 1 = flap
    STATE_SIZE = 4  # [dist_top, dist_bottom, next_pipe_x, next_pipe_y]

    def __init__(self, *, render: bool = False, frame_skip: int = 10):
        pygame.init()
        self.render_enabled = render
        self.frame_skip = frame_skip

        self.bg_raw = pygame.image.load("Images/background-day.png")
        self.floor_raw = pygame.image.load("Images/base.png")
        self.bird_raw = pygame.image.load("Images/dev.png")
        self.pipe_raw = pygame.image.load("Images/pipe-green.png")

        self.SCALE = 3
        raw_w, raw_h = self.bg_raw.get_size()
        self.SCREEN_WIDTH = raw_w * self.SCALE * 2
        self.SCREEN_HEIGHT = raw_h * self.SCALE

        if self.render_enabled:
            self.screen = pygame.display.set_mode(
                (self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()

        def sc(img):
            return pygame.transform.scale(
                img, (img.get_width() * self.SCALE, img.get_height() * self.SCALE)
            )

        self.bg_img = sc(self.bg_raw).convert()
        self.floor_img = sc(self.floor_raw).convert()
        self.bird_img = sc(self.bird_raw).convert_alpha()
        self.pipe_img = sc(self.pipe_raw).convert_alpha()

        self.gravity = 0.05 * self.SCALE
        self.jump_impulse = 2.0 * self.SCALE
        self.pipe_speed = 1.5 * self.SCALE
        self.pipe_gap = 120 * self.SCALE
        self.pipe_spacing = 200 * self.SCALE
        self.floor_y = self.SCREEN_HEIGHT - self.floor_img.get_height()

        self.reset()

    def reset(self):
        self.bird_vel = 0.0
        self.bird_rect = self.bird_img.get_rect(
            center=(50 * self.SCALE, self.SCREEN_HEIGHT // 3))
        self.pipe_list = []
        self.scored_pipes = []
        self.score = 0
        self.floor_x = 0
        self.frame_idx = 0
        self.done = False

        for i in range(3):
            self._spawn_pipe(self.SCREEN_WIDTH + i * self.pipe_spacing)

        return self._get_state()

    def step(self, action: int):
        reward = 0
        for _ in range(self.frame_skip):
            self._advance_physics(action)
            prev_bird = self.bird_rect.copy()
            prev_pipes = list(self.pipe_list)
            reward += self._handle_collisions_and_score(prev_bird, prev_pipes, action)
            self._maybe_spawn_pipe()
            self.frame_idx += 1
            if self.done:
                break

        if self.render_enabled:
            self._render()

        return self._get_state(), reward, self.done

    def set_pipe_gap(self, gap_size):
        self.pipe_gap = gap_size

    def _spawn_pipe(self, x_pos):
        min_c = self.pipe_gap // 2 + 10 * self.SCALE
        max_c = self.floor_y - self.pipe_gap // 2 - 10 * self.SCALE
        cy = random.randint(min_c, max_c)
        top = self.pipe_img.get_rect(midbottom=(x_pos, cy - self.pipe_gap // 2))
        bot = self.pipe_img.get_rect(midtop=(x_pos, cy + self.pipe_gap // 2))
        self.pipe_list.append((top, bot))

    def _maybe_spawn_pipe(self):
        rightmost = max(self.pipe_list, key=lambda p: p[0].centerx)[0].centerx
        if rightmost < self.SCREEN_WIDTH + self.pipe_spacing:
            self._spawn_pipe(rightmost + self.pipe_spacing)

    def _advance_physics(self, action):
        if action == 1:
            self.bird_vel = -self.jump_impulse
        self.bird_vel += self.gravity
        self.bird_rect.centery += self.bird_vel

        for top, bot in self.pipe_list:
            top.x -= self.pipe_speed
            bot.x -= self.pipe_speed
        self.pipe_list = [p for p in self.pipe_list if p[0].right > -50 * self.SCALE]

    def _handle_collisions_and_score(self,
                                     prev_bird_rect: pygame.Rect,
                                     prev_pipe_list: list[tuple[pygame.Rect, pygame.Rect]],
                                     action: int) -> float:
        died = False
        scored = False

        if self.bird_rect.top <= 0 or self.bird_rect.bottom >= self.floor_y:
            self.done = True
            died = True

        else:
            for pair in self.pipe_list:
                top, bot = pair
                if self.bird_rect.colliderect(top) or self.bird_rect.colliderect(bot):
                    self.done = True
                    died = True
                    break

                if pair not in self.scored_pipes and self.bird_rect.centerx > top.centerx:
                    self.scored_pipes.append(pair)
                    self.score += 1
                    scored = True
                    break

        return _reward(
            bird=self.bird_rect,
            prev=prev_bird_rect,
            act=action,
            pipes=prev_pipe_list,
            scored=scored,
            died=died,
            w=self.SCREEN_WIDTH,
            h=self.SCREEN_HEIGHT,
            floor_y=self.floor_y
        )

    def _get_state(self):
        # Get the closest pipe that the bird hasn't passed yet
        next_pipe = None
        for top, bot in self.pipe_list:
            if top.right > self.bird_rect.left:
                next_pipe = (top, bot)
                break

        # Distance to top and bottom boundaries
        dist_top = self.bird_rect.top / self.SCREEN_HEIGHT
        dist_bottom = (self.floor_y - self.bird_rect.bottom) / self.SCREEN_HEIGHT

        # Distance to next pipe (x and y)
        if next_pipe:
            top_pipe, bot_pipe = next_pipe
            pipe_x = (top_pipe.centerx - self.bird_rect.centerx) / self.SCREEN_WIDTH

            gap_center = (top_pipe.bottom + bot_pipe.top) / 2
            pipe_y = (self.bird_rect.centery - gap_center) / self.SCREEN_HEIGHT
        else:
            pipe_x = 1.0
            pipe_y = 0.0

        return np.array([
            dist_top,
            dist_bottom,
            pipe_x,
            pipe_y
        ], dtype=np.float32)

    def _render(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit();
                sys.exit()

        self.screen.blit(self.bg_img, (0, 0))
        self.screen.blit(self.bg_img, (self.bg_img.get_width(), 0))
        for top, bot in self.pipe_list:
            self.screen.blit(self.pipe_img, bot)
            self.screen.blit(pygame.transform.flip(self.pipe_img, False, True), top)
        self.screen.blit(self.bird_img, self.bird_rect)

        self.floor_x -= self.pipe_speed
        FLOOR_W = self.floor_img.get_width()
        for i in range(3):
            self.screen.blit(self.floor_img, (self.floor_x + i * FLOOR_W, self.floor_y))
        if self.floor_x <= -FLOOR_W:
            self.floor_x += FLOOR_W

        txt = pygame.font.SysFont(None, 48).render(str(self.score), True, (255, 255, 255))
        self.screen.blit(txt, (10, 10))

        pygame.display.update()
        self.clock.tick(120)

    def close(self):
        if getattr(self, "render_enabled", False):
            pygame.quit()