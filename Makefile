run:
	docker run -d -it --ipc=host --cap-add sys_ptrace -p0.0.0.0:2222:22 -v /mnt/sda1/china_release:/mnt/china_data -v /mnt/sdb1/LP_RECOGNIZER_DATA/data/KZ:/mnt/kz_data \
	--name trainer --gpus '"device=0,3"' registry.infra.smartparking.kz/trainer:1.0