#!/usr/bin/env python3
#!c:/Python35/python3.exe -u
import asyncio
import sys
import cv2
import numpy as np
import cozmo
import cozmo.util as u
import time
import os
import enum
from glob import glob

from find_cube import *

try:
    from PIL import ImageDraw, ImageFont
except ImportError:
    sys.exit('run `pip3 install --user Pillow numpy` to run this example')
def nothing(x):
    pass

YELLOW_LOWER = np.array([9, 115, 151])
YELLOW_UPPER = np.array([179, 215, 255])

GREEN_LOWER = np.array([0,0,0])
GREEN_UPPER = np.array([179, 255, 60])

# Define a decorator as a subclass of Annotator; displays the keypoint
class BoxAnnotator(cozmo.annotate.Annotator):

    cube = None

    def apply(self, image, scale):
        d = ImageDraw.Draw(image)
        bounds = (0, 0, image.width, image.height)

        if BoxAnnotator.cube is not None:

            #double size of bounding box to match size of rendered image
            BoxAnnotator.cube = np.multiply(BoxAnnotator.cube,2)

            #define and display bounding box with params:
            #msg.img_topLeft_x, msg.img_topLeft_y, msg.img_width, msg.img_height
            box = cozmo.util.ImageBox(BoxAnnotator.cube[0]-BoxAnnotator.cube[2]/2,
                                      BoxAnnotator.cube[1]-BoxAnnotator.cube[2]/2,
                                      BoxAnnotator.cube[2], BoxAnnotator.cube[2])
            cozmo.annotate.add_img_box_to_image(image, box, "green", text=None)

            BoxAnnotator.cube = None

class State(enum.IntEnum):
    IDLE = 0
    SEARCHING = 1
    ORIENTING = 2
    APPROACHING = 3
    ATCUBE = 4
    WANDERING = 5

class Event(enum.IntEnum):
    NONE = 0
    INIT_COMPLETE = 19
    FOUND_CUBE = 1
    LOST_CUBE = 2
    CUBE_REACHED = 7
    CUBE_IS_CENTERED = 8

class CubeSeeker:
    
    def __init__(self, robot : cozmo.robot.Robot):
        self.__robot = robot
        self.__state = 0
        self.__events = []
        self.__cubelog = []
        self.__center_margin = 10
        self.__width = 320
        self.__height = 200
        self.__enough_cubes = 10
        self.__turn_degrees = 24
        self.__move_inches = 1
        self.__head_speed = 10
        self.__approach_speed = 50
        self.__close_enough_cube_size = 70   
        self.__max_cubes = 20
                    
    def receive_input(self, **kwargs):
        if 'cube' in kwargs:
            self.__process_cube(kwargs['cube'])

    def step(self):
        self.__update_state()

    def __approximate_cube(self):
        
        if len(self.__cubelog) < 1:
            return None
        
        return np.array(self.__cubelog).mean(axis=0).tolist()

    def __process_cube(self, cube):
        
        if cube is None:
            if len(self.__cubelog) < 1:
                self.__raise_event(Event.LOST_CUBE)
            else:
                self.__cubelog.pop()
                print(len(self.__cubelog))
            return
        
        if len(self.__cubelog) > self.__max_cubes:
            self.__cubelog.pop()
        
        self.__cubelog.insert(0, cube)
        print(len(self.__cubelog))
        
        if len(self.__cubelog) > self.__enough_cubes:
            self.__raise_event(Event.FOUND_CUBE)
    
    def __change_state(self, state):
        if self.__robot.has_in_progress_actions:
            self.__robot.abort_all_actions()
        self.__state = state
        print("changed state to {0}".format(state))

    def __do_action(self, action):
        if not self.__robot.has_in_progress_actions:
            action()
        
    def __raise_event(self, evt : Event):
        self.__events.insert(0, evt)
        
    def __update_state(self, **input_kwargs):
        event = None
                
        if len(self.__events) > 0:
            event = self.__events.pop()
            
        apx_cube = self.__approximate_cube()
        
        ## this is essentially an empty state, s0
        if self.__state == State.IDLE:
            
            if event is not None and event == Event.INIT_COMPLETE:
                self.__change_state(State.SEARCHING)
                return
            
            self.__do_action(
                lambda : self.__robot.set_head_angle(u.Angle(radians=0)))
            
            if abs(self.__robot.head_angle) < 1:
                self.__raise_event(Event.INIT_COMPLETE)
                    
        elif self.__state == State.SEARCHING:
            if event is not None:
                if event == Event.FOUND_CUBE:
                    self.__change_state(State.ORIENTING)
                    return
                            
            self.__do_action(
                lambda : self.__robot.turn_in_place(angle=u.Angle(degrees=360), speed=u.Angle(degrees=20)))
            
            
        
        elif self.__state == State.ORIENTING:
            
            if event is not None:
                
                if event == Event.CUBE_IS_CENTERED:
                    self.__change_state(State.APPROACHING)
                    return
                    
                if event == Event.LOST_CUBE:
                    self.__change_state(State.SEARCHING)
                    return
            
            if apx_cube is None:
                return
                
            x, y, sz = apx_cube
            
            if x < self.__width/2-self.__center_margin:
                self.__do_action(
                    lambda : self.__robot.turn_in_place(u.Angle(degrees=self.__turn_degrees), speed=u.Angle(degrees=20)))
                return
            
            elif x > self.__width/2+self.__center_margin:
                self.__do_action(
                    lambda : self.__robot.turn_in_place(u.Angle(degrees=-self.__turn_degrees), speed=u.Angle(degrees=20)))
                return
            
            elif y < self.__height/2-self.__center_margin:
                self.__do_action(
                    lambda : self.__robot.move_head(-self.__head_speed/10))
                return
            
            elif y > self.__height/2+self.__center_margin:
                self.__do_action(
                    lambda : self.__robot.move_head(self.__head_speed/10))
                return
                
            self.__raise_event(Event.CUBE_IS_CENTERED)
        
        elif self.__state == State.APPROACHING:
            if event == Event.LOST_CUBE or apx_cube is None:
                print("lost the cube")
                self.__change_state(State.SEARCHING)
                return
                
            if event == Event.CUBE_REACHED:
                self.__change_state(State.ATCUBE)
                return
                
            x, y, sz = apx_cube
            
            if sz > self.__close_enough_cube_size:
                self.__raise_event(Event.CUBE_REACHED)
                return
                
            self.__do_action(
                lambda : self.__robot.drive_straight(
                    u.Distance(distance_inches=self.__move_inches), 
                    u.Speed(speed_mmps=self.__approach_speed)))
        
        elif self.__state == State.ATCUBE:
            if event == Event.LOST_CUBE:
                self.__change_state(State.SEARCHING)
                
            self.__do_action(
                lambda : self.__robot.say_text(
                    "found the cube"
                ))
                
            return

async def run(robot: cozmo.robot.Robot):

    sk = CubeSeeker(robot)

    robot.world.image_annotator.annotation_enabled = True
    robot.world.image_annotator.add_annotator('box', BoxAnnotator)

    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = True
    robot.camera.enable_auto_exposure = True

    gain,exposure,mode = 390,3,1

    try:

        while True:
            event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)   #get camera image
            if event.image is not None:
                image = cv2.cvtColor(np.asarray(event.image), cv2.COLOR_BGR2RGB)

                if mode == 1:
                    robot.camera.enable_auto_exposure = True
                else:
                    robot.camera.set_manual_exposure(exposure,fixed_gain)

                #find the cube
                cube = find_cube(image, YELLOW_LOWER, YELLOW_UPPER)
                BoxAnnotator.cube = cube
                
                sk.receive_input(cube=cube)
                
            sk.step()

    except KeyboardInterrupt:
        print("")
        print("Exit requested by user")
    except cozmo.RobotBusy as e:
        print(e)
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    cozmo.run_program(run, use_viewer = True, force_viewer_on_top = True, show_viewer_controls=True)
