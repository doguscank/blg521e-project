GENERATIONS := 20
GENERATION_SIZE := 100
OUTPUT := artifacts/checkpoints/run.json
PLOT := artifacts/plots/loss.png
MODE := single
CHECKPOINT := $(OUTPUT)
GIF_PATH := artifacts/playback/run.gif


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




