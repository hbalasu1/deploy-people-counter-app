#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """
    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None 
        self.net = None 
        self.input = None 
        self.output = None 
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model, device,requests, cpu_extension=None):
        
        ### TODO: Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        self.plugin = IECore() 
        self.net = IENetwork(model=model_xml, weights=model_bin)

        ### TODO: Add any necessary extensions 
        self.plugin.add_extension(cpu_extension, device_name="CPU")

        ### TODO: Check for supported layers 
        ### Check for any unsupported layers, and let the user
        ### know if anything is missing. Exit the program, if so.
        
        supported_layers = self.plugin.query_network(network=self.net, device_name="CPU")

        unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]

        if len(unsupported_layers) != 0:
            print("Not supported layers in model: ",unsupported_layers)
            print("Check whether extensions are available to add to IECore.")
            exit(1)


        ### TODO: Return the loaded inference plugin ###
        # Load the IENetwork into the plugin
        if requests == 0:
            # Loads network read from IR to the plugin
            self.exec_network = self.plugin.load_network(network=self.net,device_name="CPU")
        else:
            self.exec_network = self.plugin.load_network(network=self.net,device_name="CPU", num_requests=requests)
                    
        # Get the input layer
        self.input = next(iter(self.net.inputs))
        self.output = next(iter(self.net.outputs))
        ### Note: You may need to update the function parameters. ###
        return self.plugin, self.get_input_shape()

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return self.net.inputs[self.input].shape

    def exec_net(self, request_id, frame):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        self.infer_request = self.exec_network.start_async(
                                                           request_id=request_id,
                                                           inputs={self.input: frame}
                                                        )
        return self.exec_network      


    def wait(self,request_id):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return self.exec_network.requests[request_id].wait(-1)


    def get_output(self, request_id):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.exec_network.requests[request_id].outputs[self.output]

