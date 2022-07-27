docker run -d -it --rm --ipc=host --cap-add sys_ptrace -p127.0.0.1:2223:22 \
            --gpus all -e DISPLAY="$DISPLAY" \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
            --name lp_recognizer \
            doc.smartparking.kz/parking_detector:1.0