#!/bin/bash

mycount=0;
while (( $mycount < 1)); do 
    ./blender-2.93.3-linux-x64/blender material_lib_v2.blend --background --python open6dor_renderer.py -- $mycount;
((mycount=$mycount+1));
done;

