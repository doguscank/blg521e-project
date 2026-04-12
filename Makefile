GENERATIONS := 20
GENERATION_SIZE := 100
OUTPUT := artifacts/checkpoints/run_acceleration.json
PLOT := artifacts/plots/loss_acceleration.png
MODE := single
CHECKPOINT := $(OUTPUT)
GIF_PATH := artifacts/playback/run_acceleration.gif


.PHONY: train, playback, load

load:
	pip install -e '.[dev,notebook,viz]'

train:
	spce train --generations $(GENERATIONS) \
		--generation-size $(GENERATION_SIZE) \
		--output $(OUTPUT) \
		--plot $(PLOT) \
		--mode $(MODE)

playback:
	spce playback --checkpoint $(CHECKPOINT) \
		--output $(GIF_PATH)




