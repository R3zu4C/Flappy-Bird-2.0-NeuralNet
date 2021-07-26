import pygame
import os
import random
import neat
import pickle

pygame.display.set_caption("Flappy Bird")
pygame.init()

font = pygame.font.SysFont('Comic Sans MS', 30)

RES = (575, 800)

screen = pygame.display.set_mode(RES)

clock = pygame.time.Clock()

bird_img = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png")).convert_alpha()), pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png")).convert_alpha()), pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png")).convert_alpha())]
pipe_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")).convert_alpha())
base_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")).convert_alpha())
bg = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")).convert())

gen = 0

class Bird:
    IMGS = bird_img
    ANIMATION_TIME = 10

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tick = 0
        self.vel = 0
        self.img_cnt = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.y -= 5
        self.vel = -5
        self.tick = 0
    
    def move(self):
        self.tick += 1
        disp = (self.vel*self.tick) + 0.5*(self.tick)**2 

        if disp >= 15:
            disp = 15

        self.y = self.y + disp

    def draw(self, screen):
        self.img_cnt += 1

        if self.img_cnt <= self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_cnt <= self.ANIMATION_TIME * 2:
            self.img = self.IMGS[1]
        elif self.img_cnt <= self.ANIMATION_TIME * 3:
            self.img = self.IMGS[2]
        elif self.img_cnt <= self.ANIMATION_TIME * 4:
            self.img = self.IMGS[1]
        else:
            self.img = self.IMGS[0]
            self.img_cnt = 0

        screen.blit(self.img, (self.x, self.y))


class Pipe:

    def __init__(self):
        self.top_img = pygame.transform.flip(pipe_img, False, True)
        self.bot_img = pipe_img
        self.height = random.randrange(10, 500)
        self.top = self.height - self.top_img.get_height()
        self.bot = self.height + 200
        self.x = 575
        self.passed = False

    def move(self, up):
        self.x -= 5
        
        if up:
            self.top -= 3
            self.bot -= 3
        if not up:
            self.top += 3
            self.bot += 3

    def collision(self, bird):
        bird_mask = pygame.mask.from_surface(bird.img)
        top_mask = pygame.mask.from_surface(self.top_img)
        bot_mask = pygame.mask.from_surface(self.bot_img)

        if top_mask.overlap(bird_mask, (bird.x - self.x, round(bird.y) - self.top)) or bot_mask.overlap(bird_mask, (bird.x - self.x, round(bird.y) - self.bot)):
            return True
        
        return False

    def draw(self, screen):
        screen.blit(self.top_img, (self.x, self.top))
        screen.blit(self.bot_img, (self.x, self.bot))

class Base:

    def __init__(self, x):
        self.y = 730
        self.x = x
        self.img = base_img

    def move(self):
        self.x -= 5

    def collision(self, bird):
        bird_mask = pygame.mask.from_surface(bird.img)
        base_mask = pygame.mask.from_surface(self.img)

        if base_mask.overlap(bird_mask, (0, round(bird.y) - self.y)):
            return True
        
        return False


    def draw(self, screen):
        screen.blit(self.img, (self.x, self.y))

def draw_screen(screen, birds, pipes, bases, score, gen):

    if gen == 0:
        gen = 1

    screen.blit(bg, (0,0))

    for pipe in pipes:
        pipe.draw(screen)

    for bird in birds:
        bird.draw(screen)

    for base in bases:
        base.draw(screen)
    
    score_card = font.render('Score :' + str(score), False, (0, 0, 0))
    screen.blit(score_card,(0,0))
    gen_card = font.render('Gen :' + str(gen), False, (0, 0, 0))
    screen.blit(gen_card, (0, 30))
    alive_card = font.render('Alive :' + str(len(birds)), False, (0, 0, 0))
    screen.blit(alive_card, (0, 60))

    pygame.display.update()


def main(genomes, config):

    global gen
    gen += 1

    up = True

    pipes = [Pipe()]
    bases = [Base(0)]
    birds = []
    
    nets = []
    ge = []
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(50,250))
        ge.append(genome)

    score = 0

    run = True

    while run and len(birds) > 0:
        clock.tick(60)

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
        
        
        draw_screen(screen, birds, pipes, bases, score, gen) 
             
        rem_b = []
        rem_p = []
        add_p = False
        add_b = False

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].top_img.get_width():  
                pipe_ind = 1                                                                 

        for x, bird in enumerate(birds):
            ge[x].fitness += 0.1
            bird.move()

            output = nets[birds.index(bird)].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bot)))

            if output[0] > 0.5:
                bird.jump()


        for bird in birds:

            if bird.y < 0:
                nets.pop(birds.index(bird))
                ge.pop(birds.index(bird))
                birds.pop(birds.index(bird))


        for pipe in pipes:

            if pipe.bot > 730 and not up:
                up = True

            if pipe.bot - 200 < 10 and up:
                up = False

            pipe.move(up)

            for bird in birds:

                if pipe.collision(bird):
                    ge[birds.index(bird)].fitness -= 1
                    nets.pop(birds.index(bird))
                    ge.pop(birds.index(bird))
                    birds.pop(birds.index(bird))

            if pipe.x + pipe.top_img.get_width() < 0:
                rem_p.append(pipe)

            if pipe.x < bird.x and not pipe.passed:
                add_p = True
                pipe.passed = True 
                score += 1

        if add_p:
            score += 1

            for genome in ge:
                genome.fitness += 5

            pipes.append(Pipe())

        for rp in rem_p:
            pipes.remove(rp)

        for base in bases:

            base.move()

            for bird in birds:
                if base.collision(bird):
                    ge[birds.index(bird)].fitness -= 1
                    nets.pop(birds.index(bird))
                    ge.pop(birds.index(bird))
                    birds.pop(birds.index(bird))

            if base.x + base.img.get_width() <= 0:
                rem_b.append(base)

            if base.x <= 0 and len(bases) < 2:
                add_b = True

        if add_b:
            bases.append(Base(bases[0].x + bases[0].img.get_width()))

        for rb in rem_b:
            bases.remove(rb)


def run(config_file):
    
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main, 50)

    print('\nBest genome:\n{!s}'.format(winner))


run('feedForwardConfig.txt')