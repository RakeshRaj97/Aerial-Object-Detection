#####################################################
# Makefile
#	Note that this makefile only works
#	if your image contains all necessary packages.
#####################################################

#########################
# Settings
#########################
SHELL := /bin/bash
.PHONY: docker train prune test
.NOTPARALLEL: docker train prune test
.SILENT: docker train prune test

#########################
# Global envs
#########################
DEBUG := 
SPACE :=
SPACE +=
DOCKER_ROOT := '/developer'
LN_PATH := ${HOME}/Documents/repos/lightnet


#########################
# Start docker
#########################
docker: gpu :=
docker: image := top/pytorch
docker: name := TOP-voc
docker: datapath := $(shell readlink .data)
docker: guard-gpu guard-datapath
	${DEBUG} docker run -it --rm --name ${name} -h ${name} \
		--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=${gpu} \
		-e DISPLAY -e TERM --net=host --ipc=host -e PYTHONIOENCODING=UTF-8 \
		-e VISDOM='' \
		-v ${PWD}:${DOCKER_ROOT}/project \
		-v ${datapath}:${DOCKER_ROOT}/project/data \
		-v ${LN_PATH}:${DOCKER_ROOT}/lightnet \
		${image} /bin/bash -c \
			" cd project; \
			pip install -e ${DOCKER_ROOT}/lightnet; \
			/bin/bash"


#########################
# Train
#########################
train: cfg := 
train: weight :=
train: gpu :=
train: image := top/pytorch
train: backup_root = backup/train
train: name := TOP-voc_train-${gpu}
train: backup := $(basename $(notdir ${cfg}))
train: datapath := $(shell readlink .data)
train: guard-cfg guard-weight guard-gpu guard-datapath
	${DEBUG} docker run --rm --name ${name} -h ${name} \
		--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=${gpu} \
		-e DISPLAY -e TERM --net=host --ipc=host -e PYTHONIOENCODING=UTF-8 \
		-e VISDOM='' \
		-v ${PWD}:${DOCKER_ROOT}/project \
		-v ${datapath}:${DOCKER_ROOT}/project/data \
		-v ${LN_PATH}:${DOCKER_ROOT}/lightnet \
		${image} /bin/bash -c \
			"cd project; \
			pip install -e ${DOCKER_ROOT}/lightnet; \
			./bin/train.py -cn ${cfg} -b ${backup_root}/${backup} -l ${backup_root}/${backup}/file.log ${weight}"


#########################
# Prune
#########################
prune: cfg := 
prune: percent :=
prune: weight :=
prune: gpu :=
prune: minprune := 1
prune: image := top/pytorch
prune: backup_root = backup/prune
prune: name := TOP-voc_prune-${gpu}
prune: backup := $(basename $(notdir ${cfg}))--$(subst ${SPACE},_,${percent})
prune: datapath := $(shell readlink .data)
prune: guard-cfg guard-percent guard-weight guard-gpu guard-datapath
	${DEBUG} docker run --rm --name ${name} -h ${name} \
		--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=${gpu} \
		-e DISPLAY -e TERM --net=host --ipc=host -e PYTHONIOENCODING=UTF-8 \
		-e VISDOM='' \
		-v ${PWD}:${DOCKER_ROOT}/project \
		-v ${datapath}:${DOCKER_ROOT}/project/data \
		-v ${LN_PATH}:${DOCKER_ROOT}/lightnet \
		${image} /bin/bash -c \
			"cd project; \
			pip install -e ${DOCKER_ROOT}/lightnet; \
			./bin/prune.py -cn ${cfg} -b ${backup_root}/${backup} -p ${percent} -m ${minprune} -l ${backup_root}/${backup}/file.log ${weight}"


#########################
# Test
#########################
test: cfg := 
test: weight :=
test: gpu :=
test: args :=
test: iou := 0.5
test: image := top/pytorch
test: name := TOP-voc_test-${gpu}
test: datapath := $(shell readlink .data)
test: guard-cfg guard-weight guard-gpu guard-datapath
	${DEBUG} docker run -it --rm --name ${name} -h ${name} \
		--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=${gpu} \
		-e DISPLAY -e TERM --net=host --ipc=host -e PYTHONIOENCODING=UTF-8 \
		-e VISDOM='' \
		-v ${PWD}:${DOCKER_ROOT}/project \
		-v ${datapath}:${DOCKER_ROOT}/project/data \
		-v ${LN_PATH}:${DOCKER_ROOT}/lightnet \
		${image} /bin/bash -c \
			"cd project; \
			pip install -e ${DOCKER_ROOT}/lightnet; \
			./bin/test.py -cn ${cfg} -i ${iou} ${args} ${weight}"


#########################
# Ensure variable is set
#########################
guard-%:
	@ if [ "${${*}}" = "" ]; then \
		echo "Variable '$*' not set"; \
		exit 1; \
	fi
