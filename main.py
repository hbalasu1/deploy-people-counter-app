"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
import sys
import time
import socket
import json
import cv2

import logging as log

import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network
from collections import deque
import numpy as np
import parser

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser

def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client
def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    dt = 0 # detected
    l = 0 # last counted
    t = 0 # total
    s = 0 # start time  
    request_id = 0 
    dq = deque(maxlen=30)
    # Initialise the class
    infer_network = Network()

    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    
    ### TODO: Load the model through `infer_network` ###
    
    n, c, h, w = infer_network.load_model(args.model,
                                          args.device, 
                                          request_id,
                                          args.cpu_extension)[1]                                       

    ### TODO: Handle the input stream ###
    single_image_mode = False

    # Checks for live feed
    if args.input == 'CAM':
        input_stream = 0

    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_stream = args.input

    # Checks for video file
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)
    
    if input_stream:
        cap.open(args.input)

    ### TODO: Loop until stream is over ###
    if not cap.isOpened():
        log.error("ERROR! Unable to open video source")
    
    wi = cap.get(3)
    hi = cap.get(4)

    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        # Start async inference

        ### TODO: Read from the video capture ###

        image = cv2.resize(frame, (w, h))

        ### TODO: Pre-process the image as needed ###
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))

        ### TODO: Start asynchronous inference for specified request ###
        inf_start = time.time()
        infer_network.exec_net(request_id, image)

        ### TODO: Wait for the result ###
        if infer_network.wait(request_id) == 0:
            infer_time = time.time() - inf_start
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output(request_id)
            cv2.putText(frame, "time: {:.3f}ms"\
                               .format(infer_time*1000) ,
                        (250, 25),
                        cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                        0.6,
                        (255, 0, 150),1)
            ### TODO: Extract any desired stats from the results ###
            count = 0
            for obj in result[0][0]:
                # Draw bounding box for object when it's probability is more than
                # the specified threshold
                if obj[2] >= prob_threshold:
                    xmin = int(obj[3] * wi)
                    ymin = int(obj[4] * hi)
                    xmax = int(obj[5] * wi)
                    ymax = int(obj[6] * hi)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
                    count = count + 1
            ### TODO: Calculate and send relevant information on 
            ### current_count, total_count and duration to the MQTT server 
            ### Topic "person": keys of "count" and "total" 
            ### Topic "person/duration": key of "duration"

            dq.append(count)
            dt=0
            # proportion of frames with a positive detection 
            if np.sum(dq)/30 > 0.1:
                dt = 1
            # Person is counted
            if dt > l:
                s = time.time()
                t = t + dt - l
                client.publish("person", json.dumps({"total": t}))

            # Person duration in the video is calculated
            if dt < l:
                d = int(time.time() - s)
                # Publish messages to the MQTT server
                client.publish("person/duration",json.dumps({"duration": d}))

            client.publish("person", json.dumps({"count": count}))
            l = dt

        wait_key = cv2.waitKey(60)
        if wait_key == 27:
            cap.release()
            cv2.destroyAllWindows()
            client.disconnect()
            break

        ### TODO: Send the frame to the FFMPEG server ###
        frame = cv2.resize(frame, (768, 432))

        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()
        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)

    cap.release()

    cv2.destroyAllWindows()
    
    client.disconnect()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)

if __name__ == '__main__':
    main()
