docker run -d -it --rm --ipc=host --cap-add sys_ptrace -p127.0.0.1:4040:22 \
            --gpus '"device=0"'\
            --name lp_recognizer \
            registry.infra.smartparking.kz/detector:dev