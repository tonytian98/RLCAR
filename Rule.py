import keyboard
from abc import ABC


class Rule(ABC):
    def __init__(self, car):
        self.car = car

    def event_listener(self, input=None):
        pass


class RuleKeyboard(Rule):
    def event_listener(self, input=None):
        event = keyboard.read_event()

        if event.event_type == keyboard.KEY_DOWN and event.name == "up":
            self.car.accelerate()

        if event.event_type == keyboard.KEY_DOWN and event.name == "down":
            self.car.decelerate()

        if event.event_type == keyboard.KEY_DOWN and event.name == "left":
            self.car.turn_left()

        if event.event_type == keyboard.KEY_DOWN and event.name == "right":
            self.car.turn_right()
